"""T2 (DDO voters) baselines that use the listener and no debate text.

Predicts the label of a T2 config (ddo_switch: did a decided voter switch sides? ddo_undecided: where does
a tied voter end, tie / pro / con?) with class-weighted logistic regression on four groups of features:

  context      the voter's side before the debate and the debate's year; the year alone when every voter
               of the config has the same side before (ddo_undecided: all tied)
  profile      the voter's self-description in users.json: ideology, party, demographics, the 48 "big issues"
  record       how often the voter's votes in the training debates of the same config ended in each class
               (ddo_switch: stayed, switched; ddo_undecided: tie, pro, con); a training row never counts itself
  record_past  the same, counting only training debates from earlier years than the row's debate

Rows, labels, class names and splits come from the Hub dataset, so the test rows are the ones ModernBERT is
scored on. Only the id and label columns are read; the transcripts are not downloaded. The test split is
scored once per model and every test metric gets a 95% interval from a bootstrap over debates.

The metric follows the number of classes of the config:
  two classes    C is chosen on validation AUROC; test AUROC of the model's score
  more classes   C is chosen on validation macro-F1; test macro-F1 and accuracy of the predicted class, next
                 to a row that always predicts the largest training class

Two extra numbers show what a text-only model cannot do and what the profile really measures:
  same-debate pairs   among voters of one debate with the same side before, one of whom switched and one
                      stayed, how often the switcher gets the higher score (a text-only model gives both
                      the same input, so it scores 50%). Two classes only: it needs one score per voter.
  unseen voters       the profile model trained only on voters who never appear in validation or test,
                      so it cannot recognise a frequent voter by their answers

Usage (from the repository root; needs scikit-learn):
    python scripts/t1_t2/listener_baseline.py --users ddo/users.json
    python scripts/t1_t2/listener_baseline.py --users ddo/users.json --dataset-config ddo_undecided
    python scripts/t1_t2/listener_baseline.py --users ddo/users.json --output outputs/listener_baseline.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import ijson
import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.voter_record import record_counts  # noqa: E402

SPLITS = ("train", "validation", "test")
COLUMNS = ["debate_id", "voter", "before", "year", "label"]
PROFILE_FIELDS = ("political_ideology", "religious_ideology", "party", "president",
                  "gender", "education", "ethnicity", "income")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
SEED = 42
METRIC_TITLES = {"auroc": "AUROC", "macro_f1": "macro-F1", "accuracy": "accuracy"}

Rows = Dict[str, List[Any]]  # one split as columns: debate_id, voter, before, year, label (and extra ones asked for)
Matrices = Tuple[csr_matrix, csr_matrix, csr_matrix]  # train, validation, test
Metric = Callable[[np.ndarray, np.ndarray], float]  # (labels, model output) -> value


class Scoring(NamedTuple):
    """How a config is scored. `output` is what the metrics read from a fitted model: one score per row
    or one predicted class per row. C is chosen on the first metric."""
    output: Callable[[LogisticRegression, csr_matrix], np.ndarray]
    metrics: Dict[str, Metric]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Listener-only logistic regression baselines for a T2 config.")
    ap.add_argument("--users", default="ddo/users.json", help="users.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_switch",
                    help="T2 config: ddo_switch or ddo_undecided (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def load_rows(repo: str, config: str, revision: str, extra: Sequence[str] = ()) -> Tuple[Dict[str, Rows], List[str]]:
    """Id and label columns of every split (and any `extra` columns), plus the label names. Streams, so the
    transcripts stay on the Hub."""
    wanted = COLUMNS + [c for c in extra if c not in COLUMNS]
    ds = load_dataset(repo, config, revision=revision, columns=wanted, streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    names = ds["train"].features["label"].names
    splits = {}
    for split in SPLITS:
        columns: Rows = {c: [] for c in wanted}
        for row in ds[split]:
            for c in wanted:
                columns[c].append(row[c])
        splits[split] = columns
    return splits, names


def load_profiles(path: str, voters: set) -> Dict[str, Dict[str, str]]:
    """Profile fields of the given voters as {voter: {feature name: value}}; voters without a record are absent."""
    profiles = {}
    with open(path, "rb") as fh:
        for name, user in ijson.kvitems(fh, ""):
            if name not in voters:
                continue
            fields = {f: str(user.get(f)) for f in PROFILE_FIELDS}
            for issue, answer in (user.get("big_issues_dict") or {}).items():
                fields[f"issue:{issue}"] = str(answer)
            profiles[name] = fields
    return profiles


def one_hot(dicts: Dict[str, List[Dict[str, float]]]) -> Matrices:
    """Vectorize per-row feature dicts; the vocabulary comes from the training split."""
    vec = DictVectorizer(sparse=True)
    return (vec.fit_transform(dicts["train"]),) + tuple(vec.transform(dicts[s]) for s in SPLITS[1:])


def context_features(splits: Dict[str, Rows], with_side: bool) -> Matrices:
    """Side before x year, one-hot; the year alone if `with_side` is off (the side before is the same for all)."""
    def row(before: str, year: Optional[int]) -> Dict[str, float]:
        if not with_side:
            return {f"year={year}": 1.0}
        return {f"before={before}": 1.0, f"year={year}": 1.0, f"before={before}|year={year}": 1.0}

    return one_hot({s: [row(b, y) for b, y in zip(rows["before"], rows["year"])] for s, rows in splits.items()})


def profile_features(splits: Dict[str, Rows], profiles: Dict[str, Dict[str, str]]) -> Matrices:
    dicts = {s: [{f"{k}={v}": 1.0 for k, v in profiles[voter].items()} if voter in profiles else {"no_profile": 1.0}
                 for voter in rows["voter"]] for s, rows in splits.items()}
    return one_hot(dicts)


def record_features(splits: Dict[str, Rows], n_classes: int, earlier_years_only: bool) -> Matrices:
    """log(1 + votes that ended in the class) per class and a no-record flag, from the voter's votes in
    training debates. For ddo_switch the classes are stayed and switched, for ddo_undecided tie, pro and con.

    A training row never counts itself: with `earlier_years_only` only debates from earlier years count,
    otherwise the row is left out of its own voter's totals. Validation and test debates are never counted.
    The counting is scripts/common/voter_record.py's, which also writes the record as text for the models.
    """
    train = splits["train"]
    raw = {}
    for s, rows in splits.items():
        per_class = record_counts(train["voter"], train["year"], train["label"], rows["voter"], rows["year"],
                                  n_classes, earlier_years_only, own_labels=rows["label"] if s == "train" else None)
        raw[s] = np.column_stack([np.log1p(per_class), per_class.sum(axis=1) == 0])
    mean, std = raw["train"].mean(axis=0), raw["train"].std(axis=0)
    std[std == 0] = 1.0
    return tuple(csr_matrix((raw[s] - mean) / std) for s in SPLITS)


def model_rows(with_side: bool) -> Dict[str, List[str]]:
    """Rows of the result table: name -> feature groups. The context group is named after what it holds."""
    context, context_short = ("Side before x year", "side x year") if with_side else ("Year", "year")
    return {
        context: ["context"],
        "Profile": ["profile"],
        "Voting record, earlier years only": ["record_past"],
        "Voting record": ["record"],
        "Profile + voting record": ["profile", "record"],
        f"Profile + voting record + {context_short}": ["profile", "record", "context"],
    }


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of class 1; nan if only one class is present, which a bootstrap resample can produce."""
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")


def confusion(y: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """Number of rows per (true class, predicted class): one row per true class, one column per predicted class."""
    return np.bincount(y * n_classes + predicted, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def class_f1(y: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 of every class; 0 for a class that is neither present nor predicted."""
    counts = confusion(y, predicted, n_classes)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)  # 2 TP + FP + FN
    return np.divide(2 * np.diag(counts), denominator, out=np.zeros(n_classes), where=denominator > 0)


def scoring_for(n_classes: int) -> Scoring:
    """AUROC of the model's score for two classes; macro-F1 and accuracy of the predicted class for more."""
    if n_classes == 2:
        return Scoring(lambda model, x: model.decision_function(x), {"auroc": auroc})
    return Scoring(lambda model, x: model.predict(x),
                   {"macro_f1": lambda y, predicted: float(class_f1(y, predicted, n_classes).mean()),
                    "accuracy": lambda y, predicted: float((y == predicted).mean())})


def fit_best(x: Matrices, y_train: np.ndarray, y_val: np.ndarray, scoring: Scoring,
             train_mask: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray]:
    """Fit one model per C, keep the best validation value of the first metric of `scoring`.
    Returns (C, that validation value, the model's output on the test split).

    `train_mask` restricts training to the selected rows.
    """
    x_train = x[0] if train_mask is None else x[0][train_mask]
    y_train = y_train if train_mask is None else y_train[train_mask]
    select = next(iter(scoring.metrics.values()))
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x_train, y_train)
        val_value = select(y_val, scoring.output(model, x[1]))
        if best is None or val_value > best[1]:
            best = (c, val_value, scoring.output(model, x[2]))
    return best


def result_row(parts: List[str], n_features: int, c: Optional[float], val_value: float, test_output: np.ndarray,
               y_test: np.ndarray, scoring: Scoring) -> Dict[str, Any]:
    """One row of the result table: the model, the validation value it was chosen on, every test metric."""
    row = {"features": parts, "n_features": n_features, "C": c,
           f"validation_{next(iter(scoring.metrics))}": float(val_value)}
    row.update({f"test_{metric}": value(y_test, test_output) for metric, value in scoring.metrics.items()})
    return row


def evaluate(groups: Dict[str, Matrices], parts: List[str], y: Dict[str, np.ndarray], scoring: Scoring,
             train_mask: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit a model on the given feature groups. Returns its row of the result table and its test output."""
    x = tuple(hstack([groups[p][i] for p in parts]).tocsr() for i in range(3))
    c, val_value, output = fit_best(x, y["train"], y["validation"], scoring, train_mask)
    return result_row(parts, x[0].shape[1], c, val_value, output, y["test"], scoring), output


def constant_prediction(label: int, y: Dict[str, np.ndarray], scoring: Scoring) -> Tuple[Dict[str, Any], np.ndarray]:
    """Always predict `label`, with no features and nothing to fit. Returns the same pair as `evaluate`."""
    select = next(iter(scoring.metrics.values()))
    val_output, test_output = (np.full(len(y[s]), label) for s in SPLITS[1:])
    return result_row([], 0, None, select(y["validation"], val_output), test_output, y["test"], scoring), test_output


def same_debate_pairs(y: np.ndarray, scores: np.ndarray, debates: Sequence[str],
                      sides: Sequence[str]) -> Tuple[float, int]:
    """Share of (switcher, stayer) pairs from one debate and one side before whose switcher scores higher
    (ties count half), and the number of such pairs."""
    index = defaultdict(list)
    for i, key in enumerate(zip(debates, sides)):
        index[key].append(i)
    wins, pairs = 0.0, 0
    for rows in index.values():
        rows = np.array(rows)
        pos, neg = scores[rows][y[rows] == 1], scores[rows][y[rows] == 0]
        wins += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        pairs += len(pos) * len(neg)
    return (float(wins / pairs) if pairs else float("nan")), pairs


def bootstrap_by_debate(y: np.ndarray, outputs: Dict[str, np.ndarray], metrics: Dict[str, Metric],
                        debates: Sequence[str], resamples: int) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """95% interval of every test metric per model, resampling whole debates (their votes are not independent).
    A resample in which a metric is undefined does not count for that metric."""
    index = defaultdict(list)
    for i, d in enumerate(debates):
        index[d].append(i)
    groups = [np.array(v) for v in index.values()]
    rng = np.random.default_rng(SEED)
    draws = defaultdict(lambda: defaultdict(list))  # model -> metric -> values
    for _ in range(resamples):
        rows = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for name, output in outputs.items():
            for metric, value in metrics.items():
                draws[name][metric].append(value(y[rows], output[rows]))
    return {name: {metric: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                   for metric, v in by_metric.items()} for name, by_metric in draws.items()}


def print_table(results: Dict[str, Dict[str, Any]], metrics: Sequence[str], label_names: Sequence[str],
                n_rows: int) -> None:
    """One line per model: the validation value C was chosen on, every test metric with its interval, then
    the same-debate pairs (two classes) or the test F1 of each class (more classes)."""
    binary = len(label_names) == 2
    classes = f"{label_names[1]} vs {label_names[0]}" if binary else " / ".join(label_names)
    extra = ["same-debate pairs"] if binary else [f"F1 {n}" for n in label_names]
    print(f"\n=== TEST {' and '.join(METRIC_TITLES[m] for m in metrics)}, {classes} ({n_rows} rows) ===")
    print(f"  {'model':40s} {'features':>8s} {'C':>6s} {'val':>6s}"
          + "".join(f" {METRIC_TITLES[m]:>8s}  {'95% interval':12s}" for m in metrics)
          + "".join(f" {e:>{max(8, len(e))}s}" for e in extra))
    for name, r in results.items():
        c = "-" if r["C"] is None else f"{r['C']:g}"
        line = f"  {name:40s} {r['n_features']:8d} {c:>6s} {r['validation_' + metrics[0]]:6.3f}"
        for m in metrics:
            low, high = r[f"test_{m}_ci95"]
            interval = f"{low:.3f}-{high:.3f}"
            line += f" {r['test_' + m]:8.3f}  {interval:12s}"
        cells = ([f"{r['same_debate_pair_accuracy']:.1%}"] if binary
                 else [f"{r['test_f1_per_class'][n]:.3f}" for n in label_names])
        print(line + "".join(f" {cell:>{max(8, len(e))}s}" for cell, e in zip(cells, extra)))
    print(f"  (val: validation {METRIC_TITLES[metrics[0]]}, the value C is chosen on)")
    if binary:
        print(f"  (same-debate pairs: {r['same_debate_pairs']} pairs; a text-only model scores 50.0% on them)")


def main() -> None:
    args = parse_args()
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision)
    n_classes = len(label_names)
    scoring = scoring_for(n_classes)
    y = {s: np.array(rows["label"]) for s, rows in splits.items()}
    for s in SPLITS:
        counts = Counter(splits[s]["label"])
        print(f"[{args.dataset_config}] {s:10s} {len(y[s]):6d} rows, {len(set(splits[s]['debate_id'])):5d} debates  "
              + ", ".join(f"{n}={counts[i]}" for i, n in enumerate(label_names)))

    voters = {v for rows in splits.values() for v in rows["voter"]}
    profiles = load_profiles(args.users, voters)
    train_voters = set(splits["train"]["voter"])
    test_voters = splits["test"]["voter"]
    coverage = {
        "test_rows_with_profile": float(np.mean([v in profiles for v in test_voters])),
        "test_rows_with_training_votes": float(np.mean([v in train_voters for v in test_voters])),
    }
    earlier = defaultdict(lambda: 10**9)  # voter -> year of their first training vote
    for v, year in zip(splits["train"]["voter"], splits["train"]["year"]):
        if year is not None:
            earlier[v] = min(earlier[v], year)
    coverage["test_rows_with_earlier_training_votes"] = float(np.mean(
        [year is not None and earlier[v] < year for v, year in zip(test_voters, splits["test"]["year"])]))
    print(f"[COVERAGE] test rows whose voter has a profile record: {coverage['test_rows_with_profile']:.1%}; "
          f"whose voter votes in a training debate: {coverage['test_rows_with_training_votes']:.1%}; "
          f"in a training debate from an earlier year: {coverage['test_rows_with_earlier_training_votes']:.1%}")

    with_side = len(set(splits["train"]["before"])) > 1  # ddo_undecided: every voter is tied before
    groups = {
        "context": context_features(splits, with_side),
        "profile": profile_features(splits, profiles),
        "record": record_features(splits, n_classes, earlier_years_only=False),
        "record_past": record_features(splits, n_classes, earlier_years_only=True),
    }
    results, test_outputs = {}, {}
    if n_classes > 2:  # the floor for metrics of the predicted class; a constant score has AUROC 0.5 by definition
        largest = Counter(splits["train"]["label"]).most_common(1)[0][0]
        name = f"Always {label_names[largest]}"
        results[name], test_outputs[name] = constant_prediction(largest, y, scoring)
    for name, parts in model_rows(with_side).items():
        results[name], test_outputs[name] = evaluate(groups, parts, y, scoring)

    # The profile again, trained only on voters who never appear in validation or test.
    held_out_voters = set(splits["validation"]["voter"]) | set(test_voters)
    unseen = np.array([v not in held_out_voters for v in splits["train"]["voter"]])
    name = "Profile, trained on unseen voters only"
    results[name], test_outputs[name] = evaluate(groups, ["profile"], y, scoring, train_mask=unseen)
    results[name]["training_rows"] = int(unseen.sum())

    intervals = bootstrap_by_debate(y["test"], test_outputs, scoring.metrics, splits["test"]["debate_id"],
                                    args.bootstrap)
    for name, r in results.items():
        for metric, interval in intervals[name].items():
            r[f"test_{metric}_ci95"] = interval
        if n_classes == 2:
            r["same_debate_pair_accuracy"], r["same_debate_pairs"] = same_debate_pairs(
                y["test"], test_outputs[name], splits["test"]["debate_id"], splits["test"]["before"])
        else:
            r["test_f1_per_class"] = dict(zip(label_names, class_f1(y["test"], test_outputs[name], n_classes).tolist()))
            # Rows: true class, columns: predicted class, both in the order of label_names.
            r["test_confusion"] = confusion(y["test"], test_outputs[name], n_classes).tolist()
    print_table(results, list(scoring.metrics), label_names, len(y["test"]))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision},
                       "label_names": label_names, "coverage": coverage, "bootstrap_resamples": args.bootstrap,
                       "models": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
