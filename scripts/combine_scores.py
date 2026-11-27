"""Combine thread models' saved scores with simple thread features, and report test AUROC.

Reads the scores that `train_delta.py --save-scores` writes (outputs/<run>/scores-{validation,test}.jsonl)
for one or more runs on the same thread config (cmv_thread, cmv_thread_matched, cmv_thread_history, ...),
and reports test AUROC for:
  - each run alone, and the ensemble (the mean of the runs' scores, each standardized on validation);
  - the features alone: the post date, and the thread's shape (turns, characters, the OP's turns and
    their share of the turns), which the model cannot see (the date) or sees only up to 8,192 tokens;
  - the ensemble together with the date, and with every feature.
Each combination is a logistic regression fit on the validation split and scored once on the test
split. The interval is the 95% range over 1,000 bootstrap resamples of the test rows; the gains over
the ensemble are paired (the same resamples for both). Validation also chose each run's checkpoint,
so the fits may be a little optimistic there, but the test split is untouched.

Usage:
  python scripts/combine_scores.py --dataset-config cmv_thread_matched \
      --dataset-revision a1d18ad55440e3ff6de3878eb8a92f5498d5b7f1 \
      outputs/cmv_thread_matched-reva1d18ad outputs/cmv_thread_matched-reva1d18ad-seed43
"""
import argparse
import json
import os
import re

import numpy as np
from datasets import load_dataset

DATASET_REPO = "nlp-final-project11/persuasion-datasets"
BOOTSTRAP = 1000
SEED = 42
TURN = re.compile(r"^(OP|User \d+|Deleted user): ", re.M)  # the turn labels of cmv_named in create_dataset.py
OP_TURN = re.compile(r"^OP: ", re.M)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ensemble thread models and add the post date and thread shape.")
    ap.add_argument("runs", nargs="+", help="run folders that hold scores-validation.jsonl and scores-test.jsonl")
    ap.add_argument("--dataset-config", required=True, help="the config the runs were trained on")
    ap.add_argument("--dataset-revision", default="main", help="the revision the runs were trained on (default: %(default)s)")
    return ap.parse_args()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Chance a random delta thread outscores a random no-delta one (ties count half)."""
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iterations: int = 50) -> np.ndarray:
    """Weights (intercept first) of an L2-regularized logistic regression, by Newton's method."""
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    penalty = l2 * np.eye(len(w))
    penalty[0, 0] = 0  # no penalty on the intercept
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ w))
        gradient = X.T @ (p - y) + penalty @ w
        hessian = (X * (p * (1 - p))[:, None]).T @ X + penalty
        w -= np.linalg.solve(hessian, gradient)
    return w


def read_scores(run: str, split: str) -> list[dict]:
    path = os.path.join(run, f"scores-{split}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: train or score that run with train_delta.py --save-scores")
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def features(rows) -> dict[str, np.ndarray]:
    """Raw per-thread features: post date, and turns, characters and OP turns of the thread text."""
    turns = np.array([len(TURN.findall(reply)) for reply in rows["reply"]], float)
    op_turns = np.array([len(OP_TURN.findall(reply)) for reply in rows["reply"]], float)
    return {
        "date": np.array(rows["created_utc"], float),
        "log_turns": np.log1p(turns),
        "log_chars": np.log1p(np.array([len(reply) for reply in rows["reply"]], float)),
        "log_op_turns": np.log1p(op_turns),
        "op_share": op_turns / np.maximum(turns, 1),
    }


def main() -> None:
    args = parse_args()
    labels, columns, ids = {}, {}, {}
    for split in ("validation", "test"):
        rows = load_dataset(DATASET_REPO, args.dataset_config, revision=args.dataset_revision, split=split)
        first = read_scores(args.runs[0], split)
        index = [r["row"] for r in first]  # all rows, or the first 32 after a --smoke run
        rows = rows.select(index)
        labels[split] = np.array(rows["label"])
        ids[split] = rows["post_id"]
        columns[split] = features(rows)
        print(f"[DATA] {args.dataset_config}@{args.dataset_revision[:7]} {split}: {len(index)} rows, "
              f"{labels[split].mean():.3f} delta")

    # per run: scores in split order, checked against the dataset's labels and post ids
    run_scores = {"validation": [], "test": []}
    for run in args.runs:
        for split in ("validation", "test"):
            scores = read_scores(run, split)
            if [r["row"] for r in scores] != [r["row"] for r in read_scores(args.runs[0], split)]:
                raise SystemExit(f"{run}: its {split} rows differ from {args.runs[0]}'s")
            if [r.get("post_id") for r in scores] != list(ids[split]) or \
                    [r["label"] for r in scores] != labels[split].tolist():
                raise SystemExit(f"{run}: its {split} scores do not line up with {args.dataset_config}@"
                                 f"{args.dataset_revision}; was it trained on another config or revision?")
            run_scores[split].append(np.array([r["score"] for r in scores]))

    # standardize every input on validation, so runs and features are on one scale
    def standardize(values):
        mean, std = values["validation"].mean(), values["validation"].std() or 1.0
        return {split: (v - mean) / std for split, v in values.items()}

    inputs = {f"run {os.path.basename(run.rstrip('/'))}":
              standardize({s: run_scores[s][i] for s in run_scores}) for i, run in enumerate(args.runs)}
    runs = list(inputs)
    inputs["ensemble"] = {s: np.mean([inputs[r][s] for r in runs], axis=0) for s in ("validation", "test")}
    for name in columns["validation"]:
        inputs[name] = standardize({s: columns[s][name] for s in ("validation", "test")})

    shape = ["log_turns", "log_chars", "log_op_turns", "op_share"]
    combinations = {name: [name] for name in runs}
    combinations.update({
        "ensemble of the runs": ["ensemble"],
        "post date alone": ["date"],
        "thread shape alone (turns, characters, OP turns)": shape,
        "date + thread shape (no text)": ["date"] + shape,
        "ensemble + date": ["ensemble", "date"],
        "ensemble + date + thread shape": ["ensemble", "date"] + shape,
    })
    test_scores = {}
    for name, parts in combinations.items():
        X = {s: np.column_stack([inputs[p][s] for p in parts]) for s in ("validation", "test")}
        if len(parts) == 1:  # a model's score as it is; a feature's direction (e.g. fewer turns) chosen on validation
            flip = parts[0] in columns["validation"] and auroc(X["validation"][:, 0], labels["validation"]) < 0.5
            test_scores[name] = (-1.0 if flip else 1.0) * X["test"][:, 0]
        else:
            w = fit_logistic(X["validation"], labels["validation"])
            test_scores[name] = np.c_[np.ones(len(X["test"])), X["test"]] @ w

    y = labels["test"]
    rng = np.random.default_rng(SEED)
    samples = [idx for idx in (rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP)) if 0 < y[idx].sum() < len(idx)]
    boot = {name: np.array([auroc(s[idx], y[idx]) for idx in samples]) for name, s in test_scores.items()}

    print(f"\n=== TEST AUROC ({len(y)} threads; 95% bootstrap interval) ===")
    for name, s in test_scores.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        print(f"  {name:52s} {auroc(s, y):.4f}  [{lo:.3f}, {hi:.3f}]")
    print("\n=== GAIN OVER THE ENSEMBLE (paired bootstrap) ===")
    for name in ("ensemble + date", "ensemble + date + thread shape"):
        diff = boot[name] - boot["ensemble of the runs"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        gain = auroc(test_scores[name], y) - auroc(test_scores["ensemble of the runs"], y)
        print(f"  {name:52s} {gain:+.4f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
