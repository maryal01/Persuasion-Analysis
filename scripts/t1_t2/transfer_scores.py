"""RQ2 transfer: saved model scores on one platform's test split of the common format, next to the length-only baseline.

scripts/t1_t2/compare_models.py compares the transfer runs on the DDO test votes. It cannot read the CMV
test pairs (they have no voter), and it has no row for the length-only baseline. This script covers both,
for either platform:

  - One row per model: AUROC with a 95% interval, and on CMV the pairwise accuracy (how often the reply
    that earned the delta scores above the other reply of its pair), as scripts/train_delta.py reports it.
    Several seeds of one model: folders separated by commas, averaged as in compare_models.py.
  - The length-only baseline of scripts/t1_t2/length_baseline.py as rows, fitted on each platform and on
    both: "Length only, fitted on DDO", "... on CMV", "... on both".
  - Paired comparisons, every later model with the first and any --compare ROW OTHER_ROW: the difference,
    its 95% interval on the same resamples, and a paired permutation test.

The bootstrap and the permutation test resample and swap whole debates (DDO) or posts (CMV), because the
rows of one debate or post are not independent. They are the functions of compare_models.py, so a DDO row
has the same numbers here as there.

Reads the parquet files that scripts/t1_t2/build_common.py writes (ids, labels, and the texts whose words
the length baseline counts) and the predictions_test.parquet of each run. Nothing is trained and nothing
is read from the Hub. A model's score is the logit of the second class minus the first.

Usage (from the repository root, after build_common.py; the prediction files come from Gilbreth):
    python scripts/t1_t2/transfer_scores.py --platform cmv \
        --model "CMV only=outputs/cmv_common" \
        --model "DDO only, scored on CMV=outputs/cmv_common-by-ddo_common" \
        --compare "DDO only, scored on CMV" "Length only, fitted on DDO" \
        --output outputs/transfer_scores_cmv.json
"""
import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

from compare_models import Scores, bootstrap_values, debate_groups, interval, named, permutation_p
from length_baseline import PLATFORMS, TRAINING_SETS, fit, load
from listener_baseline import SEED, auroc

PLATFORM_NAMES = {"ddo": "DDO", "cmv": "CMV", "both": "both"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--platform", choices=list(PLATFORMS), required=True, help="the platform whose test split is scored")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FOLDER[,FOLDER...]",
                    help="outputs/ folder of a run with predictions_test.parquet for this platform's test split; "
                         "several folders separated by commas are seeds; repeatable. Later models are compared with the first.")
    ap.add_argument("--compare", action="append", default=[], nargs=2, metavar=("ROW", "OTHER_ROW"),
                    help="one more paired comparison, by row name; repeatable")
    ap.add_argument("--data", default="outputs/common", help="folder written by build_common.py (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates or posts")
    ap.add_argument("--permutations", type=int, default=2000, help="permutations per paired test")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def read_scores(path: str, split_path: str) -> np.ndarray:
    """A model's score for every test row. The file must hold the rows of the split in its order: the
    labels and every id column that the two files share have to be equal row by row."""
    predictions = pq.read_table(path)
    shared = [c for c in predictions.column_names if c in pq.read_schema(split_path).names]
    if not shared:
        raise SystemExit(f"{path} shares no id column with {split_path}, so its rows cannot be checked")
    reference = pq.read_table(split_path, columns=shared + ["label"])
    if predictions.num_rows != reference.num_rows:
        raise SystemExit(f"{path} has {predictions.num_rows} rows, the test split {reference.num_rows}; "
                         "is it from another config or dataset version?")
    for column, other in [("labels", "label")] + [(c, c) for c in shared]:
        if predictions[column].to_pylist() != reference[other].to_pylist():
            raise SystemExit(f"{path}: column {column!r} differs from the test split's, so the rows are not the same")
    logits = np.array(predictions["logits"].to_pylist(), dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise SystemExit(f"{path}: expected 2 logits per row, found shape {logits.shape}")
    return logits[:, 1] - logits[:, 0]


def complete_pairs(y: np.ndarray, pairs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Row numbers of the persuaded and of the not persuaded reply of every pair that has both."""
    rows: Dict[int, Dict[int, int]] = {}
    for i, (label, pair) in enumerate(zip(y, pairs)):
        rows.setdefault(pair, {})[int(label)] = i
    both = np.array([(r[1], r[0]) for r in rows.values() if len(r) == 2])
    return both[:, 0], both[:, 1]


def mean_outcome(_: np.ndarray, outcomes: np.ndarray) -> float:
    """Pairwise accuracy from one outcome per pair; the first argument (labels) is unused."""
    return float(outcomes.mean())


def main() -> None:
    args = parse_args()
    folders = named(args.model, "model")
    config, group = PLATFORMS[args.platform]
    data = {name: {split: load(args.data, c, split, g) for split in ("train", "test")} for name, (c, g) in PLATFORMS.items()}
    test_rows = data[args.platform]["test"]
    y = test_rows["y"]
    groups = debate_groups(test_rows["group"])
    split_path = os.path.join(args.data, config, "test.parquet")
    print(f"[{config}] test: {len(y)} rows, {int(y.sum())} persuaded, {len(groups)} {group.split('_')[0]}s")

    test: Dict[str, Scores] = {name: [read_scores(os.path.join(folder, "predictions_test.parquet"), split_path)
                                      for folder in seeds] for name, seeds in folders.items()}
    for name, parts in TRAINING_SETS.items():
        row = f"Length only, fitted on {PLATFORM_NAMES[name]}"
        if row in test:
            raise SystemExit(f"a model cannot be named {row!r}")
        model, mean, std = fit([data[p]["train"] for p in parts])
        test[row] = [model.decision_function((test_rows["x"] - mean) / std)]
    first = next(iter(folders))
    comparisons = [(name, first) for name in list(folders)[1:]] + [tuple(pair) for pair in args.compare]
    for pair in comparisons:
        for name in pair:
            if name not in test:
                raise SystemExit(f"--compare: no row named {name!r}; rows: {list(test)}")

    # metric -> (labels, one array per seed and row, groups to resample, the metric, compare on ranks?)
    measures: Dict[str, Tuple[np.ndarray, Dict[str, Scores], List[np.ndarray], Any, bool]] = {
        "auroc": (y, test, groups, auroc, True)}
    if test_rows["pair"]:
        won, lost = complete_pairs(y, test_rows["pair"])
        outcomes = {name: [np.where(s[won] > s[lost], 1.0, np.where(s[won] == s[lost], 0.5, 0.0)) for s in per_seed]
                    for name, per_seed in test.items()}
        pair_groups = debate_groups([test_rows["group"][i] for i in won])
        measures["pairwise_acc"] = (np.zeros(len(won)), outcomes, pair_groups, mean_outcome, False)
        print(f"  {len(won)} complete pairs")

    rows: Dict[str, Dict[str, Any]] = {name: {"seeds": len(per_seed)} for name, per_seed in test.items()}
    results: List[Dict[str, Any]] = [{"row": row, "other_row": other} for row, other in comparisons]
    for metric, (labels, values, by_group, function, ranks) in measures.items():
        draws = bootstrap_values(labels, values, by_group, args.bootstrap, {metric: function})[metric]
        for name, per_seed in values.items():
            per_seed_values = [function(labels, v) for v in per_seed]
            rows[name].update({metric: float(np.mean(per_seed_values)), f"{metric}_per_seed": per_seed_values,
                               f"{metric}_std": float(np.std(per_seed_values, ddof=1)) if len(per_seed_values) > 1 else None,
                               f"{metric}_ci95": list(interval(draws[name]))})
        rng = np.random.default_rng(SEED)
        for result in results:
            row, other = result["row"], result["other_row"]
            result.update({f"{metric}_difference": rows[row][metric] - rows[other][metric],
                           f"{metric}_difference_ci95": list(interval(draws[row] - draws[other])),
                           f"{metric}_permutation_p": permutation_p(labels, values[row], values[other], by_group,
                                                                    args.permutations, rng, function, ranks=ranks,
                                                                    what=f"{row!r} with {other!r}")})

    width = max(len(name) for name in rows)
    unit = "debates" if args.platform == "ddo" else "posts"
    print(f"\n=== TEST, {config} ({len(y)} rows; 95% intervals from a bootstrap over {unit}) ===")
    print(f"  {'row':{width}s} {'seeds':>5s}  " + "  ".join(f"{metric:>12s}  {'95% interval':12s}" for metric in measures))
    for name, r in rows.items():
        cells = []
        for metric in measures:
            low, high = r[f"{metric}_ci95"]
            cells.append(f"{r[metric]:12.3f}  {f'{low:.3f}-{high:.3f}':12s}")
        print(f"  {name:{width}s} {r['seeds']:5d}  " + "  ".join(cells))
    print(f"\n=== PAIRED COMPARISONS (first row minus second row; the permutation test swaps whole {unit}) ===")
    for result in results:
        print(f"  {result['row']}  minus  {result['other_row']}")
        for metric in measures:
            low, high = result[f"{metric}_difference_ci95"]
            print(f"      {metric} difference {result[f'{metric}_difference']:+.3f} ({low:+.3f} to {high:+.3f}), "
                  f"permutation test p = {result[f'{metric}_permutation_p']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"data": args.data, "platform": args.platform, "config": config, "models": folders,
                       "test_rows": int(len(y)), "test_groups": len(groups), "bootstrap_resamples": args.bootstrap,
                       "permutations": args.permutations, "rows": rows, "comparisons": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
