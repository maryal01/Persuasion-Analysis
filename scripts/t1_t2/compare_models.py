"""T2 (ddo_switch): compare saved model scores, and combine a text model with the listener baseline.

scripts/train_delta.py saves the scores of a run per row (predictions_validation.parquet and
predictions_test.parquet in the run's outputs/ folder). This script reads them and answers two questions on
the test split:

  1. Is one model better than another? Every model gets its test AUROC with a 95% interval. Every
     comparison gets the AUROC difference with a 95% interval and a paired permutation test. Both resample
     whole debates, because the votes of one debate are not independent.
  2. Does the debate text add to what is known about the voter without it? The listener baseline
     (scripts/t1_t2/listener_baseline.py: the voter's voting record, profile, side before and year) is
     fitted on the training split as before. A logistic regression on two numbers, the text model's score
     and the listener model's score, is then fitted on the VALIDATION split and scored once on the test
     split. The test split is never used for fitting.

The voting record is not part of the audience description in the project proposal (that is the stance
profile). Rows that use it are an extension and are named so in the output.

Only two-class configs are supported (one score per row is needed).

Usage (from the repository root; the prediction files come from Gilbreth, see the analysis README):
    python scripts/t1_t2/compare_models.py --users ddo/users.json \
        --model "Text only=outputs/ddo_switch" \
        --model "Text + audience=outputs/ddo_switch-audience-self" \
        --output outputs/compare_models.json
Optional rows that have test scores only (controls), and extra comparisons by row name:
        --test-scores "Text + audience, swapped profiles=outputs/ddo_switch-audience-self/predictions_test_shuffled.parquet" \
        --compare "Text + audience" "Text + audience, swapped profiles"
"""
import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from listener_baseline import (SEED, Matrices, Rows, auroc, context_features, fit_best, load_profiles, load_rows,
                               profile_features, record_features, same_debate_pairs, scoring_for)

# Listener models that a text model is combined with: row name -> (short name, feature groups of the baseline).
LISTENER_MODELS = {
    "Listener: voting record": ("voting record", ["record"]),
    "Listener: voting record, earlier years only": ("voting record, earlier years only", ["record_past"]),
    "Listener: profile + voting record + side x year": ("all listener features", ["profile", "record", "context"]),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare saved T2 model scores and combine them with the listener baseline.")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FOLDER",
                    help="a run's outputs/ folder with predictions_validation.parquet and predictions_test.parquet; "
                         "repeat for several models. Later models are compared with the first.")
    ap.add_argument("--test-scores", action="append", default=[], metavar="NAME=FILE",
                    help="a predictions_test*.parquet file that gets a row but no combination (controls); repeatable")
    ap.add_argument("--compare", action="append", default=[], nargs=2, metavar=("ROW", "OTHER_ROW"),
                    help="an extra comparison, ROW minus OTHER_ROW, by row name; repeatable")
    ap.add_argument("--users", default="ddo/users.json", help="users.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_switch", help="two-class T2 config (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--permutations", type=int, default=2000, help="permutations per paired test")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def named(pairs: Sequence[str], what: str) -> Dict[str, str]:
    """Parse repeated NAME=PATH arguments, keeping their order."""
    out = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not name or not path or name in out:
            raise SystemExit(f"--{what} needs NAME=PATH with a new name each time, got {pair!r}")
        out[name] = path
    return out


def read_scores(path: str, rows: Rows) -> np.ndarray:
    """One score per row of a split, in the order of `rows`: the logit of the second class minus the first.

    `path` is a file written by scripts/train_delta.py. Its rows are matched on (debate, voter); the file
    must hold exactly the rows of the split, with the same labels.
    """
    table = pq.read_table(path, columns=["debate_id", "voter", "labels", "logits"]).to_pydict()
    logits = np.array(table["logits"], dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise SystemExit(f"{path}: expected two logits per row, found shape {logits.shape}")
    position = {key: i for i, key in enumerate(zip(table["debate_id"], table["voter"]))}
    keys = list(zip(rows["debate_id"], rows["voter"]))
    if len(position) != len(logits) or len(set(keys)) != len(keys):
        raise SystemExit(f"{path}: a (debate, voter) pair occurs twice, so rows cannot be matched")
    if len(keys) != len(position) or any(key not in position for key in keys):
        raise SystemExit(f"{path} has {len(position)} rows that do not match the {len(keys)} rows of the split; "
                         "is it from another config, split or dataset version?")
    order = np.array([position[key] for key in keys])
    if (np.array(table["labels"])[order] != np.array(rows["label"])).any():
        raise SystemExit(f"{path}: labels differ from the dataset's")
    return (logits[:, 1] - logits[:, 0])[order]


def listener_scores(groups: Dict[str, Matrices], parts: List[str], y: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Validation and test scores of the listener baseline on these feature groups: fitted on the training
    split, C chosen on validation AUROC, exactly as scripts/t1_t2/listener_baseline.py does."""
    x = tuple(hstack([groups[p][i] for p in parts]).tocsr() for i in range(3))
    c, _, test = fit_best(x, y["train"], y["validation"], scoring_for(2))
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x[0], y["train"])
    if not np.allclose(model.decision_function(x[2]), test):
        raise SystemExit("the refitted listener model differs from the baseline's; do not trust this run")
    return model.decision_function(x[1]), test


def combine(validation: Sequence[np.ndarray], test: Sequence[np.ndarray], y_validation: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Fit a logistic regression on the validation scores of several models and score the test rows with it.

    Returns the test scores and the weight of each model's score. The scores are standardised with their
    validation mean and standard deviation first, so the weights can be compared with each other.
    """
    x_validation, x_test = np.column_stack(validation), np.column_stack(test)
    mean, std = x_validation.mean(axis=0), x_validation.std(axis=0)
    std[std == 0] = 1.0
    model = LogisticRegression(class_weight="balanced", max_iter=5000).fit((x_validation - mean) / std, y_validation)
    return model.decision_function((x_test - mean) / std), [float(w) for w in model.coef_[0]]


def debate_groups(debates: Sequence[str]) -> List[np.ndarray]:
    """Row numbers of every debate, in the order the debates first appear."""
    index: Dict[str, List[int]] = {}
    for i, debate in enumerate(debates):
        index.setdefault(debate, []).append(i)
    return [np.array(rows) for rows in index.values()]


def bootstrap_aurocs(y: np.ndarray, scores: Dict[str, np.ndarray], groups: List[np.ndarray], resamples: int) -> Dict[str, np.ndarray]:
    """AUROC of every row of the table on the same resamples of whole debates, so that two rows can be
    compared resample by resample. The resamples are the listener baseline's (same seed, same order)."""
    rng = np.random.default_rng(SEED)
    draws = {name: np.empty(resamples) for name in scores}
    for r in range(resamples):
        rows = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for name, s in scores.items():
            draws[name][r] = auroc(y[rows], s[rows])
    return draws


def interval(values: np.ndarray) -> Tuple[float, float]:
    """95% interval of bootstrap values; a resample with an undefined value does not count."""
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def permutation_p(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: List[np.ndarray], permutations: int,
                  rng: np.random.Generator) -> float:
    """Two-sided paired permutation test of the AUROC difference between two models on the same rows.

    Under "the two models are interchangeable", swapping their scores changes nothing, so in each
    permutation the two scores are swapped for all rows of a debate with probability 1/2 (whole debates,
    as in the bootstrap). Scores are replaced by their ranks first: that puts both models on one scale
    and does not change a model's own AUROC.
    """
    a, b = rankdata(a), rankdata(b)
    observed = abs(auroc(y, a) - auroc(y, b))
    debate_of_row = np.empty(len(y), dtype=int)
    for g, rows in enumerate(groups):
        debate_of_row[rows] = g
    at_least_as_large = 0
    for _ in range(permutations):
        swap = (rng.random(len(groups)) < 0.5)[debate_of_row]
        difference = auroc(y, np.where(swap, b, a)) - auroc(y, np.where(swap, a, b))
        at_least_as_large += abs(difference) >= observed - 1e-12
    return (at_least_as_large + 1) / (permutations + 1)


def print_results(rows: Dict[str, Dict[str, Any]], comparisons: List[Dict[str, Any]], n_test: int, n_pairs: int) -> None:
    width = max(len(name) for name in rows)
    print(f"\n=== TEST AUROC ({n_test} rows; 95% interval from a bootstrap over debates) ===")
    print(f"  {'row':{width}s} {'val':>6s} {'AUROC':>6s}  {'95% interval':12s} {'same-debate pairs':>17s}")
    for name, r in rows.items():
        val = "-" if r["validation_auroc"] is None else f"{r['validation_auroc']:.3f}"
        low, high = r["test_auroc_ci95"]
        print(f"  {name:{width}s} {val:>6s} {r['test_auroc']:6.3f}  {f'{low:.3f}-{high:.3f}':12s} {r['same_debate_pair_accuracy']:17.1%}")
        if "weights" in r:
            print(f"  {'':{width}s}   weights of the standardised scores: "
                  + ", ".join(f"{part} {weight:+.2f}" for part, weight in r["weights"].items()))
    print("  (val: validation AUROC; '-' for a combination, which is fitted on the validation split)")
    print(f"  (same-debate pairs: {n_pairs} pairs of a switcher and a stayer from one debate with the same side before)")

    print("\n=== PAIRED COMPARISONS on the test split (first row minus second row) ===")
    for c in comparisons:
        low, high = c["difference_ci95"]
        print(f"  {c['row']}  minus  {c['other_row']}")
        print(f"      AUROC difference {c['difference']:+.3f} ({low:+.3f} to {high:+.3f}), permutation test p = {c['permutation_p']:.4f}")


def main() -> None:
    args = parse_args()
    model_folders, control_files = named(args.model, "model"), named(args.test_scores, "test-scores")
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision)
    if len(label_names) != 2:
        raise SystemExit(f"{args.dataset_config} has {len(label_names)} classes; this script needs a two-class config")
    y = {s: np.array(rows["label"]) for s, rows in splits.items()}
    print(f"[{args.dataset_config}] test: {len(y['test'])} rows, {int(y['test'].sum())} {label_names[1]}; "
          f"validation: {len(y['validation'])} rows, {int(y['validation'].sum())} {label_names[1]}")

    # Scores per row of the result table: validation (None for controls and combinations) and test.
    validation: Dict[str, Any] = {}
    test: Dict[str, np.ndarray] = {}
    for name, folder in model_folders.items():
        validation[name] = read_scores(os.path.join(folder, "predictions_validation.parquet"), splits["validation"])
        test[name] = read_scores(os.path.join(folder, "predictions_test.parquet"), splits["test"])
    for name, path in control_files.items():
        if name in test:
            raise SystemExit(f"two rows are named {name!r}")
        validation[name], test[name] = None, read_scores(path, splits["test"])

    voters = {v for rows in splits.values() for v in rows["voter"]}
    groups = {
        "context": context_features(splits, with_side=True),
        "profile": profile_features(splits, load_profiles(args.users, voters)),
        "record": record_features(splits, 2, earlier_years_only=False),
        "record_past": record_features(splits, 2, earlier_years_only=True),
    }
    for name, (_, parts) in LISTENER_MODELS.items():
        validation[name], test[name] = listener_scores(groups, parts, y)

    # Text model + listener model, fitted on the validation split. Each is compared with both of its parts.
    comparisons = [(name, next(iter(model_folders))) for name in list(model_folders)[1:]]
    weights = {}
    for model in model_folders:
        for listener, (short, _) in LISTENER_MODELS.items():
            name = f"{model} + {short}"
            test[name], w = combine([validation[model], validation[listener]], [test[model], test[listener]], y["validation"])
            validation[name], weights[name] = None, {model: w[0], listener: w[1]}
            comparisons += [(name, listener), (name, model)]
    for row, other in args.compare:
        if row not in test or other not in test:
            raise SystemExit(f"--compare: no row named {row!r} or {other!r}; rows are: {', '.join(test)}")
        comparisons.append((row, other))

    debates = splits["test"]["debate_id"]
    groups_of_rows = debate_groups(debates)
    draws = bootstrap_aurocs(y["test"], test, groups_of_rows, args.bootstrap)
    rows, n_pairs = {}, 0
    for name, scores in test.items():
        pair_accuracy, n_pairs = same_debate_pairs(y["test"], scores, debates, splits["test"]["before"])
        rows[name] = {"validation_auroc": None if validation[name] is None else auroc(y["validation"], validation[name]),
                      "test_auroc": auroc(y["test"], scores), "test_auroc_ci95": interval(draws[name]),
                      "same_debate_pair_accuracy": pair_accuracy}
        if name in weights:
            rows[name]["weights"] = weights[name]
    rng = np.random.default_rng(SEED)
    results = [{"row": row, "other_row": other,
                "difference": rows[row]["test_auroc"] - rows[other]["test_auroc"],
                "difference_ci95": interval(draws[row] - draws[other]),
                "permutation_p": permutation_p(y["test"], test[row], test[other], groups_of_rows, args.permutations, rng)}
               for row, other in comparisons]
    print_results(rows, results, len(y["test"]), n_pairs)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision},
                       "models": model_folders, "test_scores": control_files,
                       "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
                       "same_debate_pairs": n_pairs, "rows": rows, "comparisons": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
