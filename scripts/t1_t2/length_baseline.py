"""RQ2 transfer: a length-only baseline for the common-format configs.

The proposal asks for a baseline that sees only how long the texts are, to rule out that a model which
transfers between CMV and DDO does so on surface cues. This fits a class-weighted logistic regression on
two numbers per row, the words of the arguments and the words of the claim (both as log(1 + n)), on the
training split of one platform or of both, and scores it on the test split of each platform.

Reads the parquet files that scripts/t1_t2/build_common.py writes. Intervals are 95%, from a bootstrap
over debates (DDO) or posts (CMV), because rows of one debate or post are not independent.

Usage (from the repository root, after build_common.py):
    python scripts/t1_t2/length_baseline.py --data outputs/common --output outputs/length_baseline.json
"""
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
PLATFORMS = {"ddo": ("ddo_common", "debate_id"), "cmv": ("cmv_common", "post_id")}  # config, the column rows are grouped by


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Length-only baseline for the CMV/DDO transfer experiments.")
    ap.add_argument("--data", default="outputs/common", help="folder written by build_common.py (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates or posts")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def load(folder: str, config: str, split: str, group: str) -> Dict[str, Any]:
    """Length features, labels and groups of one split. Only the word counts of the texts are kept."""
    table = pq.read_table(os.path.join(folder, config, f"{split}.parquet")).to_pydict()
    x = np.log1p(np.array([[len(reply.split()), len(original.split())]
                           for reply, original in zip(table["reply"], table["original"])], dtype=float))
    return {"x": x, "y": np.array(table["label"]), "group": table[group], "pair": table.get("pair_id")}


def pairwise_accuracy(scores: np.ndarray, y: np.ndarray, pairs: List[int]) -> float:
    """Share of (persuaded, not persuaded) pairs in which the persuaded row has the higher score; ties count half."""
    by_pair: Dict[int, Dict[int, float]] = {}
    for score, label, pair in zip(scores, y, pairs):
        by_pair.setdefault(pair, {})[int(label)] = float(score)
    complete = [p for p in by_pair.values() if len(p) == 2]
    return float(np.mean([1.0 if p[1] > p[0] else 0.5 if p[1] == p[0] else 0.0 for p in complete]))


def interval(y: np.ndarray, scores: np.ndarray, groups: List[Any], resamples: int) -> Tuple[float, float]:
    """95% bootstrap interval of the AUROC, resampling whole groups."""
    index: Dict[Any, List[int]] = {}
    for i, g in enumerate(groups):
        index.setdefault(g, []).append(i)
    members = [np.array(rows) for rows in index.values()]
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(resamples):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(set(y[rows])) == 2:
            values.append(roc_auc_score(y[rows], scores[rows]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    args = parse_args()
    data = {name: {split: load(args.data, config, split, group) for split in ("train", "test")}
            for name, (config, group) in PLATFORMS.items()}
    training_sets = {"ddo": ["ddo"], "cmv": ["cmv"], "both": ["ddo", "cmv"]}
    results: Dict[str, Dict[str, Any]] = {}
    print(f"{'trained on':11s} {'tested on':10s} {'rows':>6s} {'AUROC':>6s}  {'95% interval':13s} {'pairwise':>8s}  weights (arguments, claim)")
    for name, parts in training_sets.items():
        x = np.concatenate([data[p]["train"]["x"] for p in parts])
        y = np.concatenate([data[p]["train"]["y"] for p in parts])
        mean, std = x.mean(axis=0), x.std(axis=0)
        model = LogisticRegression(class_weight="balanced", max_iter=1000).fit((x - mean) / std, y)
        results[name] = {"train_rows": int(len(y)), "weights": [float(w) for w in model.coef_[0]], "test": {}}
        for platform in PLATFORMS:
            test = data[platform]["test"]
            scores = model.decision_function((test["x"] - mean) / std)
            row = {"rows": int(len(test["y"])), "auroc": float(roc_auc_score(test["y"], scores)),
                   "auroc_ci95": list(interval(test["y"], scores, test["group"], args.bootstrap)),
                   "pairwise_acc": pairwise_accuracy(scores, test["y"], test["pair"]) if test["pair"] else None}
            results[name]["test"][platform] = row
            pairwise = "-" if row["pairwise_acc"] is None else f"{row['pairwise_acc']:.3f}"
            low, high = row["auroc_ci95"]
            print(f"{name:11s} {platform:10s} {row['rows']:6d} {row['auroc']:6.3f}  {f'{low:.3f}-{high:.3f}':13s} {pairwise:>8s}  "
                  f"{results[name]['weights'][0]:+.2f}, {results[name]['weights'][1]:+.2f}")
    print("(AUROC: how often a persuaded row scores above one that was not; pairwise: the same within a CMV pair;")
    print(" weights of the standardised log word counts: a positive one means longer texts go with persuasion)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"data": args.data, "features": ["log1p words of the arguments", "log1p words of the claim"],
                       "bootstrap_resamples": args.bootstrap, "trained_on": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
