"""T2 (ddo_undecided): which side wins over a tied voter? The question of Longpre et al. (2019).

Longpre et al. keep only the voters who were persuaded and predict the side, Pro or Con, that they end
on. Their "from the middle" case is a voter who was tied before the debate and picked a side after it.
This script scores the saved `ddo_undecided` models on that question: on the test votes that left the
tie, a model's answer is the side with the larger logit, and the metric is accuracy, as in their paper.
Nothing is trained here. The models were trained on all tied voters, with the tie as a third class.

One row per model (several seeds as folders separated by commas), and one for always answering the side
that is more common among the training votes that left the tie. Every row gets a 95% interval from a
bootstrap over debates. Every later model is compared with the first, and every model with the constant
row: difference, interval on the same resamples, and a paired permutation test that swaps whole debates,
all as in scripts/t1_t2/compare_models.py.

Usage (from the repository root; streams ids and labels from the Hub, so no transcripts are downloaded):
    python scripts/t1_t2/which_side.py \
        --model "Text only=outputs/ddo_undecided,outputs/ddo_undecided-seed43,outputs/ddo_undecided-seed44" \
        --model "Text + audience=outputs/ddo_undecided-audience-self,outputs/ddo_undecided-audience-self-seed43,outputs/ddo_undecided-audience-self-seed44" \
        --output outputs/which_side.json
"""
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from compare_models import Scores, bootstrap_values, debate_groups, interval, named, paired_seeds, permutation_p, read_logits
from listener_baseline import SEED, load_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FOLDER[,FOLDER...]",
                    help="outputs/ folder of a trained run with predictions_test.parquet; several folders "
                         "separated by commas are seeds; repeatable. Later models are compared with the first.")
    ap.add_argument("--tie-class", default="tie", help="name of the class for a voter who stays tied (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_undecided", help="config of the tied voters (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--permutations", type=int, default=2000, help="permutations per paired test")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    return float((y == predicted).mean())


def main() -> None:
    args = parse_args()
    folders = named(args.model, "model")
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision)
    if args.tie_class not in label_names or len(label_names) != 3:
        raise SystemExit(f"{args.dataset_config} has the classes {label_names}; expected {args.tie_class!r} and two sides")
    tie = label_names.index(args.tie_class)
    sides = [i for i in range(len(label_names)) if i != tie]
    y_train, y_test = np.array(splits["train"]["label"]), np.array(splits["test"]["label"])
    keep = y_test != tie
    y = y_test[keep]
    debates = [debate for debate, kept in zip(splits["test"]["debate_id"], keep) if kept]
    groups = debate_groups(debates)
    train_counts = {label_names[s]: int((y_train == s).sum()) for s in sides}
    test_counts = {label_names[s]: int((y == s).sum()) for s in sides}
    print(f"[{args.dataset_config}] votes that left the tie: train {sum(train_counts.values())} ({train_counts}), "
          f"test {len(y)} of {len(y_test)} ({test_counts}) from {len(groups)} debates")

    majority = max(sides, key=lambda s: train_counts[label_names[s]])
    constant = f"Always {label_names[majority]}"
    test: Dict[str, Scores] = {constant: [np.full(len(y), majority)]}
    for name, seeds in folders.items():
        if name in test:
            raise SystemExit(f"two rows are named {name!r}")
        test[name] = []
        for folder in seeds:
            logits = read_logits(os.path.join(folder, "predictions_test.parquet"), splits["test"], len(label_names))[keep]
            test[name].append(np.array(sides)[logits[:, sides].argmax(axis=1)])  # the side with the larger logit

    draws = bootstrap_values(y, test, groups, args.bootstrap, {"accuracy": accuracy})["accuracy"]
    rows: Dict[str, Dict[str, Any]] = {}
    for name, per_seed in test.items():
        values = [accuracy(y, predicted) for predicted in per_seed]
        rows[name] = {
            "seeds": len(per_seed), "accuracy": float(np.mean(values)), "accuracy_per_seed": values,
            "accuracy_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "accuracy_ci95": list(interval(draws[name])),
            # how often a vote for each side is found, and how often each side is answered
            "recall": {label_names[s]: float(np.mean([(p[y == s] == s).mean() for p in per_seed])) for s in sides},
            "answered": {label_names[s]: float(np.mean([(p == s).mean() for p in per_seed])) for s in sides},
        }

    first = next(iter(folders))
    pairs = [(name, first) for name in list(folders)[1:]] + [(name, constant) for name in folders]
    rng = np.random.default_rng(SEED)
    comparisons: List[Dict[str, Any]] = []
    for row, other in pairs:
        a, b = paired_seeds(test[row], test[other], f"{row!r} with {other!r}")
        comparisons.append({
            "row": row, "other_row": other,
            "difference": float(np.mean([accuracy(y, f) - accuracy(y, s) for f, s in zip(a, b)])),
            "difference_ci95": list(interval(draws[row] - draws[other])),
            "permutation_p": permutation_p(y, test[row], test[other], groups, args.permutations, rng, accuracy,
                                           ranks=False, what=f"{row!r} with {other!r}"),
        })

    width = max(len(name) for name in rows)
    print(f"\n=== TEST accuracy, which side ({len(y)} votes that left the tie; 95% interval from a bootstrap over debates) ===")
    print(f"  {'row':{width}s} {'seeds':>5s} {'accuracy':>8s} {'std':>6s}  {'95% interval':12s}  "
          + "  ".join(f"found {label_names[s]:>4s}" for s in sides) + f"  answers {label_names[sides[0]]}")
    for name, r in rows.items():
        std = "-" if r["accuracy_std"] is None else f"{r['accuracy_std']:.3f}"
        low, high = r["accuracy_ci95"]
        print(f"  {name:{width}s} {r['seeds']:5d} {r['accuracy']:8.3f} {std:>6s}  {f'{low:.3f}-{high:.3f}':12s}  "
              + "  ".join(f"{r['recall'][label_names[s]]:10.3f}" for s in sides) + f"  {r['answered'][label_names[sides[0]]]:10.3f}")
        if r["seeds"] > 1:
            print(f"  {'':{width}s} per seed: " + ", ".join(f"{v:.3f}" for v in r["accuracy_per_seed"]))
    print("\n=== PAIRED COMPARISONS (first row minus second row) ===")
    for c in comparisons:
        low, high = c["difference_ci95"]
        print(f"  {c['row']}  minus  {c['other_row']}\n      accuracy difference {c['difference']:+.3f} "
              f"({low:+.3f} to {high:+.3f}), permutation test p = {c['permutation_p']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision},
                       "label_names": label_names, "tie_class": args.tie_class, "models": folders,
                       "train_votes_that_left_the_tie": train_counts, "test_votes_that_left_the_tie": test_counts,
                       "test_debates": len(groups), "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
                       "rows": rows, "comparisons": comparisons}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
