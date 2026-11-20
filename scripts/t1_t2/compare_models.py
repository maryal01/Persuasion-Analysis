"""T2 (ddo_switch, ddo_undecided): compare saved model scores, and combine a text model with the listener baseline.

scripts/train_delta.py saves the scores of a run per row (predictions_validation.parquet and
predictions_test.parquet in the run's outputs/ folder). This script reads them and answers two questions on
the test split:

  1. Is one model better than another? Every model gets its test metric with a 95% interval. Every
     comparison gets the difference with a 95% interval and a paired permutation test. Both resample
     whole debates, because the votes of one debate are not independent.
  2. Two classes only: does the debate text add to what is known about the voter without it? The listener
     baseline (scripts/t1_t2/listener_baseline.py: the voter's voting record, profile, side before and
     year) is fitted on the training split as before. A logistic regression on two numbers, the text
     model's score and the listener model's score, is then fitted on the VALIDATION split and scored once
     on the test split. The test split is never used for fitting.

The metric follows the number of classes of the config, as in the listener baseline:
  two classes    AUROC of the model's score (the logit of the second class minus the first)
  more classes   macro-F1 of the predicted class (the largest logit), with accuracy and the F1 of every
                 class next to it. No combination is fitted. The rows of the listener baseline are added to
                 compare with: the largest training class, the year, the profile and the voting record.

Several seeds of one model: give their folders (or files) separated by commas. The row then shows the
metric averaged over the seeds, with the standard deviation between seeds. Its 95% interval is that of
the average (the seeds are averaged inside every bootstrap resample), so it describes the test votes,
not the spread between seeds. A combination is fitted once per seed. In a comparison, seeds are paired
in the order given; a row with one set of scores is compared with every seed of the other row.

--subset COLUMN reports every row and comparison a second time, on the test rows where that true/false
column of the dataset is true. Nothing is fitted again: the models are the ones trained on all rows.

The voting record is not part of the audience description in the project proposal (that is the stance
profile). Rows that use it are an extension and are named so in the output.

Usage (from the repository root; the prediction files come from Gilbreth, see the analysis README):
    python scripts/t1_t2/compare_models.py --users ddo/users.json \
        --model "Text only=outputs/ddo_switch" \
        --model "Text + audience=outputs/ddo_switch-audience-self" \
        --output outputs/compare_models.json
Three seeds per model:
        --model "Text only=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44"
Optional rows that have test scores only (controls), and extra comparisons by row name:
        --test-scores "Text + audience, swapped profiles=outputs/ddo_switch-audience-self/predictions_test_shuffled.parquet" \
        --compare "Text + audience" "Text + audience, swapped profiles"
Three classes, also on the voters who marked a debater on another criterion:
    python scripts/t1_t2/compare_models.py --users ddo/users.json --dataset-config ddo_undecided \
        --model "Text only=outputs/ddo_undecided" \
        --model "Text + audience=outputs/ddo_undecided-audience-self" \
        --subset filled_other_criteria --output outputs/compare_models_undecided.json
"""
import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from listener_baseline import (METRIC_TITLES, SEED, SPLIT_KINDS, Matrices, Metric, Rows, class_f1, confusion,
                               constant_prediction, context_features, evaluate, fit_best, load_profiles, load_rows,
                               model_rows, profile_features, record_features, same_debate_pairs, scoring_for)

# Two classes: listener models that a text model is combined with: row name -> (short name, feature groups
# of the baseline).
LISTENER_MODELS = {
    "Listener: voting record": ("voting record", ["record"]),
    "Listener: voting record, earlier years only": ("voting record, earlier years only", ["record_past"]),
    "Listener: profile + voting record + side x year": ("all listener features", ["profile", "record", "context"]),
}

# What the metric reads for one row of the result table, one array per seed: a score per test row (two
# classes) or a predicted class per test row (more classes).
Scores = List[np.ndarray]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare saved T2 model scores and combine them with the listener baseline.")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FOLDER[,FOLDER...]",
                    help="a run's outputs/ folder with predictions_validation.parquet and predictions_test.parquet; "
                         "several folders separated by commas are seeds of one model. Repeat for several models. "
                         "Later models are compared with the first.")
    ap.add_argument("--test-scores", action="append", default=[], metavar="NAME=FILE[,FILE...]",
                    help="a predictions_test*.parquet file that gets a row but no combination (controls); "
                         "several files separated by commas are seeds; repeatable")
    ap.add_argument("--compare", action="append", default=[], nargs=2, metavar=("ROW", "OTHER_ROW"),
                    help="an extra comparison, ROW minus OTHER_ROW, by row name; repeatable")
    ap.add_argument("--subset", metavar="COLUMN",
                    help="a true/false column of the dataset, e.g. filled_other_criteria: every row and comparison "
                         "is reported a second time on the test rows where it is true")
    ap.add_argument("--users", default="ddo/users.json", help="users.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_switch", help="T2 config: ddo_switch or ddo_undecided (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--split", choices=SPLIT_KINDS, default="random",
                    help="the split the models were trained on: random = the one stored on the Hub; temporal = "
                         "train on debates through 2015, test on later ones (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--permutations", type=int, default=2000, help="permutations per paired test")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def named(pairs: Sequence[str], what: str) -> Dict[str, List[str]]:
    """Parse repeated NAME=PATH[,PATH...] arguments, keeping their order: name -> one path per seed."""
    out = {}
    for pair in pairs:
        name, _, paths = pair.partition("=")
        seeds = [path for path in paths.split(",") if path]
        if not name or not seeds or name in out:
            raise SystemExit(f"--{what} needs NAME=PATH[,PATH...] with a new name each time, got {pair!r}")
        out[name] = seeds
    return out


def read_logits(path: str, rows: Rows, n_classes: int) -> np.ndarray:
    """The logits of a split, one line per row in the order of `rows`.

    `path` is a file written by scripts/train_delta.py. Its rows are matched on (debate, voter); the file
    must hold exactly the rows of the split, with the same labels and one logit per class.
    """
    table = pq.read_table(path, columns=["debate_id", "voter", "labels", "logits"]).to_pydict()
    logits = np.array(table["logits"], dtype=float)
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise SystemExit(f"{path}: expected {n_classes} logits per row, found shape {logits.shape}")
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
    return logits[order]


def read_scores(path: str, rows: Rows, n_classes: int) -> np.ndarray:
    """What the metric reads, per row of a split in the order of `rows`. Two classes: one score, the logit
    of the second class minus the first. More classes: the predicted class, the one with the largest logit."""
    logits = read_logits(path, rows, n_classes)
    return logits[:, 1] - logits[:, 0] if n_classes == 2 else logits.argmax(axis=1)


def listener_features(splits: Dict[str, Rows], n_classes: int, users: str, with_side: bool) -> Dict[str, Matrices]:
    """The feature groups of the listener baseline, built as scripts/t1_t2/listener_baseline.py builds them."""
    voters = {v for rows in splits.values() for v in rows["voter"]}
    return {
        "context": context_features(splits, with_side=with_side),
        "profile": profile_features(splits, load_profiles(users, voters)),
        "record": record_features(splits, n_classes, earlier_years_only=False),
        "record_past": record_features(splits, n_classes, earlier_years_only=True),
    }


def listener_scores(groups: Dict[str, Matrices], parts: List[str], y: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Validation and test scores of the listener baseline on these feature groups: fitted on the training
    split, C chosen on validation AUROC, exactly as scripts/t1_t2/listener_baseline.py does."""
    x = tuple(hstack([groups[p][i] for p in parts]).tocsr() for i in range(3))
    c, _, test = fit_best(x, y["train"], y["validation"], scoring_for(2))
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x[0], y["train"])
    if not np.allclose(model.decision_function(x[2]), test):
        raise SystemExit("the refitted listener model differs from the baseline's; do not trust this run")
    return model.decision_function(x[1]), test


def reference_rows(splits: Dict[str, Rows], y: Dict[str, np.ndarray], label_names: List[str],
                   users: str) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """More classes: the rows of the listener baseline, to compare the models with. They use no debate text:
    always the largest training class, and logistic regression on the year, the profile and the voting
    record, fitted and chosen exactly as scripts/t1_t2/listener_baseline.py does.

    Returns, per row name, the validation value of the main metric and the predicted class of every test row.
    """
    n_classes = len(label_names)
    scoring = scoring_for(n_classes)
    main = next(iter(scoring.metrics))
    with_side = len(set(splits["train"]["before"])) > 1  # ddo_undecided: every voter is tied before
    groups = listener_features(splits, n_classes, users, with_side)
    largest = Counter(splits["train"]["label"]).most_common(1)[0][0]
    fitted = {f"Always {label_names[largest]}": constant_prediction(largest, y, scoring)}
    for name, parts in model_rows(with_side).items():
        fitted[f"Listener: {name[0].lower()}{name[1:]}"] = evaluate(groups, parts, y, scoring)
    return ({name: row[f"validation_{main}"] for name, (row, _) in fitted.items()},
            {name: output for name, (_, output) in fitted.items()})


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


def bootstrap_values(y: np.ndarray, scores: Dict[str, Scores], groups: List[np.ndarray], resamples: int,
                     metrics: Dict[str, Metric]) -> Dict[str, Dict[str, np.ndarray]]:
    """Every metric of every row of the table, averaged over its seeds, on the same resamples of whole
    debates, so that two rows can be compared resample by resample. The resamples are the listener
    baseline's (same seed, same order). Returns metric -> row name -> one value per resample."""
    rng = np.random.default_rng(SEED)
    draws = {metric: {name: np.empty(resamples) for name in scores} for metric in metrics}
    for r in range(resamples):
        rows = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        y_rows = y[rows]
        for name, per_seed in scores.items():
            resampled = [s[rows] for s in per_seed]
            for metric, value in metrics.items():
                draws[metric][name][r] = np.mean([value(y_rows, s) for s in resampled])
    return draws


def interval(values: np.ndarray) -> Tuple[float, float]:
    """95% interval of bootstrap values; a resample with an undefined value does not count."""
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def paired_seeds(a: Scores, b: Scores, what: str) -> Tuple[Scores, Scores]:
    """Line up the seeds of two rows: seed by seed in the order given, or a row with one set of scores
    against every seed of the other row."""
    if len(a) != len(b) and min(len(a), len(b)) != 1:
        raise SystemExit(f"cannot compare {what}: one row has {len(a)} seeds and the other {len(b)}")
    longest = max(len(a), len(b))
    return (a * longest if len(a) == 1 else a), (b * longest if len(b) == 1 else b)


def permutation_p(y: np.ndarray, a: Scores, b: Scores, groups: List[np.ndarray], permutations: int,
                  rng: np.random.Generator, metric: Metric, ranks: bool, what: str = "these rows") -> float:
    """Two-sided paired permutation test of the difference in `metric` between two rows on the same test rows.

    Under "the two models are interchangeable", swapping their outputs changes nothing, so in each
    permutation the two outputs are swapped for all rows of a debate with probability 1/2 (whole debates,
    as in the bootstrap). With `ranks` (two classes) scores are replaced by their ranks first: that puts
    both models on one scale and does not change a model's own AUROC. Predicted classes (more classes) are
    swapped as they are. With several seeds the swap is applied to every pair of seeds at once, and the
    statistic is the difference averaged over the seeds.
    """
    if ranks:
        a, b = [rankdata(s) for s in a], [rankdata(s) for s in b]
    a, b = paired_seeds(a, b, what)

    def difference(first: Scores, second: Scores) -> float:
        return float(np.mean([metric(y, f) - metric(y, s) for f, s in zip(first, second)]))

    observed = abs(difference(a, b))
    debate_of_row = np.empty(len(y), dtype=int)
    for g, rows in enumerate(groups):
        debate_of_row[rows] = g
    at_least_as_large = 0
    for _ in range(permutations):
        swap = (rng.random(len(groups)) < 0.5)[debate_of_row]
        swapped = difference([np.where(swap, s_b, s_a) for s_a, s_b in zip(a, b)],
                             [np.where(swap, s_a, s_b) for s_a, s_b in zip(a, b)])
        at_least_as_large += abs(swapped) >= observed - 1e-12
    return (at_least_as_large + 1) / (permutations + 1)


def score_rows(y: np.ndarray, test: Dict[str, Scores], comparisons: List[Tuple[str, str]], debates: Sequence[str],
               sides: Sequence[str], label_names: List[str], resamples: int, permutations: int,
               keep: Optional[np.ndarray] = None) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], int]:
    """Every row of the result table and every comparison, on the test rows or on those selected by `keep`.

    Returns the table (row name -> numbers), the comparisons, and the number of same-debate pairs (two
    classes; 0 otherwise).
    """
    if keep is not None:
        y = y[keep]
        test = {name: [scores[keep] for scores in per_seed] for name, per_seed in test.items()}
        debates = [debate for debate, kept in zip(debates, keep) if kept]
        sides = [side for side, kept in zip(sides, keep) if kept]
    n_classes = len(label_names)
    metrics = scoring_for(n_classes).metrics
    main = next(iter(metrics))  # AUROC for two classes, macro-F1 for more
    groups = debate_groups(debates)
    draws = bootstrap_values(y, test, groups, resamples, metrics)
    rows, n_pairs = {}, 0
    for name, per_seed in test.items():
        values = [metrics[main](y, scores) for scores in per_seed]
        rows[name] = {"seeds": len(per_seed),
                      f"test_{main}": float(np.mean(values)), f"test_{main}_per_seed": values,
                      f"test_{main}_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                      f"test_{main}_ci95": interval(draws[main][name])}
        for metric in list(metrics)[1:]:
            rows[name][f"test_{metric}"] = float(np.mean([metrics[metric](y, scores) for scores in per_seed]))
            rows[name][f"test_{metric}_ci95"] = interval(draws[metric][name])
        if n_classes == 2:
            pairs = [same_debate_pairs(y, scores, debates, sides) for scores in per_seed]
            n_pairs = pairs[0][1]
            rows[name]["same_debate_pair_accuracy"] = float(np.mean([accuracy for accuracy, _ in pairs]))
        else:
            per_class = np.mean([class_f1(y, scores, n_classes) for scores in per_seed], axis=0)
            rows[name]["test_f1_per_class"] = dict(zip(label_names, per_class.tolist()))
            # Per seed. Rows: true class, columns: predicted class, both in the order of label_names.
            rows[name]["test_confusion_per_seed"] = [confusion(y, scores, n_classes).tolist() for scores in per_seed]
    rng = np.random.default_rng(SEED)
    results = [{"row": row, "other_row": other,
                "difference": rows[row][f"test_{main}"] - rows[other][f"test_{main}"],
                "difference_ci95": interval(draws[main][row] - draws[main][other]),
                "permutation_p": permutation_p(y, test[row], test[other], groups, permutations, rng, metrics[main],
                                               ranks=n_classes == 2, what=f"{row!r} with {other!r}")}
               for row, other in comparisons]
    return rows, results, n_pairs


def print_results(rows: Dict[str, Dict[str, Any]], comparisons: List[Dict[str, Any]], where: str, n_pairs: int,
                  label_names: List[str]) -> None:
    binary = len(label_names) == 2
    main = "auroc" if binary else "macro_f1"
    title = METRIC_TITLES[main]
    extra = ["same-debate pairs"] if binary else ["accuracy"] + [f"F1 {name}" for name in label_names]
    width, column = max(len(name) for name in rows), max(6, len(title))
    print(f"\n=== TEST {title} ({where}; 95% interval from a bootstrap over debates) ===")
    print(f"  {'row':{width}s} {'seeds':>5s} {'val':>6s} {title:>{column}s} {'std':>6s}  {'95% interval':12s}"
          + "".join(f" {e:>{max(8, len(e))}s}" for e in extra))
    for name, r in rows.items():
        val = "-" if r.get(f"validation_{main}") is None else f"{r[f'validation_{main}']:.3f}"
        std = "-" if r[f"test_{main}_std"] is None else f"{r[f'test_{main}_std']:.3f}"
        low, high = r[f"test_{main}_ci95"]
        cells = ([f"{r['same_debate_pair_accuracy']:.1%}"] if binary
                 else [f"{r['test_accuracy']:.3f}"] + [f"{r['test_f1_per_class'][n]:.3f}" for n in label_names])
        print(f"  {name:{width}s} {r['seeds']:5d} {val:>6s} {r[f'test_{main}']:{column}.3f} {std:>6s}  {f'{low:.3f}-{high:.3f}':12s}"
              + "".join(f" {cell:>{max(8, len(e))}s}" for cell, e in zip(cells, extra)))
        if r["seeds"] > 1:
            print(f"  {'':{width}s}   per seed: " + ", ".join(f"{value:.3f}" for value in r[f"test_{main}_per_seed"]))
        if "weights" in r:
            print(f"  {'':{width}s}   weights of the standardised scores: "
                  + ", ".join(f"{part} {weight:+.2f}" for part, weight in r["weights"].items()))
    print(f"  (val: validation {title} on all validation rows; '-' for a row without validation scores, such as a")
    print("   control or a combination, which is fitted on the validation split)")
    print(f"  (with several seeds: {title}, val and the other columns are averages over the seeds, std is the standard")
    print("   deviation between seeds, and the interval is that of the average)")
    if binary:
        print(f"  (same-debate pairs: {n_pairs} pairs of a switcher and a stayer from one debate with the same side before)")

    print(f"\n=== PAIRED COMPARISONS on the test split, {where} (first row minus second row) ===")
    for c in comparisons:
        low, high = c["difference_ci95"]
        print(f"  {c['row']}  minus  {c['other_row']}")
        print(f"      {title} difference {c['difference']:+.3f} ({low:+.3f} to {high:+.3f}), permutation test p = {c['permutation_p']:.4f}")


def main() -> None:
    args = parse_args()
    model_folders, control_files = named(args.model, "model"), named(args.test_scores, "test-scores")
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision,
                                    extra=[args.subset] if args.subset else [], split=args.split)
    n_classes = len(label_names)
    main_metric = next(iter(scoring_for(n_classes).metrics))
    measure = scoring_for(n_classes).metrics[main_metric]
    y = {s: np.array(rows["label"]) for s, rows in splits.items()}
    print(f"[{args.dataset_config}] " + "; ".join(
        f"{s}: {len(y[s])} rows, " + ", ".join(f"{int((y[s] == i).sum())} {name}" for i, name in enumerate(label_names) if i or n_classes > 2)
        for s in ("test", "validation")))

    # What the metric reads, per row of the result table, one array per seed. Validation scores exist for
    # the models and, with two classes, for the listener models; the validation value of every row is kept.
    validation: Dict[str, Scores] = {}
    validation_value: Dict[str, Optional[float]] = {}
    test: Dict[str, Scores] = {}
    for name, folders in model_folders.items():
        validation[name] = [read_scores(os.path.join(f, "predictions_validation.parquet"), splits["validation"], n_classes) for f in folders]
        validation_value[name] = float(np.mean([measure(y["validation"], scores) for scores in validation[name]]))
        test[name] = [read_scores(os.path.join(f, "predictions_test.parquet"), splits["test"], n_classes) for f in folders]
    for name, paths in control_files.items():
        if name in test:
            raise SystemExit(f"two rows are named {name!r}")
        validation_value[name], test[name] = None, [read_scores(path, splits["test"], n_classes) for path in paths]

    comparisons = [(name, next(iter(model_folders))) for name in list(model_folders)[1:]]
    weights = {}
    if n_classes == 2:
        groups = listener_features(splits, n_classes, args.users, with_side=True)
        for name, (_, parts) in LISTENER_MODELS.items():
            if name in test:
                raise SystemExit(f"two rows are named {name!r}")
            listener_validation, listener_test = listener_scores(groups, parts, y)
            validation[name], test[name] = [listener_validation], [listener_test]
            validation_value[name] = float(measure(y["validation"], listener_validation))
        # Text model + listener model, fitted on the validation split, once per seed of the text model.
        # Each combination is compared with both of its parts.
        for model in model_folders:
            for listener, (short, _) in LISTENER_MODELS.items():
                name = f"{model} + {short}"
                fitted = [combine([v, validation[listener][0]], [t, test[listener][0]], y["validation"])
                          for v, t in zip(validation[model], test[model])]
                validation_value[name], test[name] = None, [scores for scores, _ in fitted]
                weights[name] = {model: float(np.mean([w[0] for _, w in fitted])), listener: float(np.mean([w[1] for _, w in fitted]))}
                comparisons += [(name, listener), (name, model)]
    else:
        reference_validation, reference_test = reference_rows(splits, y, label_names, args.users)
        for name, predicted in reference_test.items():
            if name in test:
                raise SystemExit(f"two rows are named {name!r}")
            validation_value[name], test[name] = float(reference_validation[name]), [predicted]
    for row, other in args.compare:
        if row not in test or other not in test:
            raise SystemExit(f"--compare: no row named {row!r} or {other!r}; rows are: {', '.join(test)}")
        comparisons.append((row, other))
    for row, other in comparisons:  # refuse rows whose seeds cannot be paired before the slow part starts
        paired_seeds(test[row], test[other], f"{row!r} with {other!r}")

    debates, sides = splits["test"]["debate_id"], splits["test"]["before"]
    rows, results, n_pairs = score_rows(y["test"], test, comparisons, debates, sides, label_names, args.bootstrap, args.permutations)
    # the validation value first, then the test numbers, then the weights of a combination
    rows = {name: {"seeds": r.pop("seeds"), f"validation_{main_metric}": validation_value[name], **r,
                   **({"weights": weights[name]} if name in weights else {})} for name, r in rows.items()}
    print_results(rows, results, f"{len(y['test'])} rows", n_pairs, label_names)
    out = {"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision,
                       "split": args.split},
           "label_names": label_names, "metric": main_metric,
           "models": model_folders, "test_scores": control_files,
           "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
           "same_debate_pairs": n_pairs, "rows": rows, "comparisons": results}

    if args.subset:
        values = splits["test"][args.subset]
        if any(value not in (True, False) for value in values):
            raise SystemExit(f"--subset: {args.subset} is not a true/false column")
        keep = np.array(values, dtype=bool)
        if not keep.any():
            raise SystemExit(f"--subset: {args.subset} is true on no test row")
        kept_debates = len({debate for debate, kept in zip(debates, keep) if kept})
        subset_rows, subset_results, subset_pairs = score_rows(y["test"], test, comparisons, debates, sides, label_names,
                                                               args.bootstrap, args.permutations, keep=keep)
        where = f"{int(keep.sum())} rows from {kept_debates} debates where {args.subset} is true"
        print(f"\n##### The same on the test rows where {args.subset} is true ({keep.mean():.1%} of the test rows); "
              "nothing is fitted again #####")
        print("  classes in this subset: " + ", ".join(f"{int((y['test'][keep] == i).sum())} {name}" for i, name in enumerate(label_names)))
        print_results(subset_rows, subset_results, where, subset_pairs, label_names)
        out["subset"] = {"column": args.subset, "test_rows": int(keep.sum()), "test_debates": kept_debates,
                         "class_counts": {name: int((y["test"][keep] == i).sum()) for i, name in enumerate(label_names)},
                         "same_debate_pairs": subset_pairs, "rows": subset_rows, "comparisons": subset_results}

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
