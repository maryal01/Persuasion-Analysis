"""T2 text + audience with filled profiles: the scores by what the filled profile changed for the voter.

`--audience filled` (scripts/common/audience.py) keeps the labels a voter reported and adds the T3 model's
labels where the voter reported none. This script scores saved models separately on these test votes:

  no reported profile        the voters the proposal names: their line was "Voter profile: unknown"
      blank, now inferred        and is an inferred profile with filled profiles
      still unknown              and stays unknown: the voter wrote nothing the T3 model could read
  some reported profile      the voter reported at least one label
      some labels added          and the T3 model added the missing ones
      unchanged                  and the line is the same with filled profiles

Every subset gets the table and the paired comparisons of scripts/t1_t2/compare_models.py, by its
functions: the metric with a 95% interval from a bootstrap over debates, and for every later model its
difference with the first with an interval and a paired permutation test. Nothing is trained or fitted
again. A score inside a subset ranks the votes of that subset only, so the subsets do not add up to the
score on all votes. No voter name is printed or written.

Usage (from the repository root; reads ids, labels and the two profile tables from the Hub):
    python scripts/t1_t2/filled_subsets.py --audience-revision t3 \
        --model "Text + audience=outputs/ddo_switch-audience-self" \
        --model "Text + audience, filled profiles=outputs/ddo_switch-audience-filled" \
        --model "Text only=outputs/ddo_switch" \
        --output outputs/filled_subsets.json
Three seeds per model: folders separated by commas, as in compare_models.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from compare_models import Scores, named, print_results, read_scores, score_rows
from listener_baseline import load_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.audience import PROFILES_REVISION, UNKNOWN_PROFILE, load_profiles, profiles_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=FOLDER[,FOLDER...]",
                    help="outputs/ folder of a run with predictions_test.parquet; several folders separated by "
                         "commas are seeds; repeatable. Later models are compared with the first.")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_switch", help="T2 config (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--audience-revision", default=PROFILES_REVISION,
                    help="branch that holds the two profile tables (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--permutations", type=int, default=2000, help="permutations per paired test")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    folders = named(args.model, "model")
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision)
    rows = splits["test"]
    y = np.array(rows["label"])
    test: Dict[str, Scores] = {name: [read_scores(os.path.join(folder, "predictions_test.parquet"), rows, len(label_names))
                                      for folder in seeds] for name, seeds in folders.items()}
    first = next(iter(folders))
    comparisons = [(name, first) for name in list(folders)[1:]]

    reported = profiles_for(rows["voter"], load_profiles("self", args.dataset_repo, args.audience_revision)[0])
    filled = profiles_for(rows["voter"], load_profiles("filled", args.dataset_repo, args.audience_revision)[0])
    blank = np.array([line == UNKNOWN_PROFILE for line in reported])
    changed = np.array([a != b for a, b in zip(reported, filled)])
    subsets = {"no reported profile": blank, "blank, now inferred": blank & changed, "still unknown": blank & ~changed,
               "some reported profile": ~blank, "some labels added": ~blank & changed, "unchanged": ~blank & ~changed}

    out: Dict[str, Any] = {"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision},
                           "audience_revision": args.audience_revision, "label_names": label_names, "models": folders,
                           "test_rows": len(y), "bootstrap_resamples": args.bootstrap, "permutations": args.permutations,
                           "subsets": {}}
    for name, keep in subsets.items():
        counts = {label: int((y[keep] == i).sum()) for i, label in enumerate(label_names)}
        debates = len({debate for debate, kept in zip(rows["debate_id"], keep) if kept})
        print(f"\n##### {name}: {int(keep.sum())} test votes ({keep.mean():.1%}) from {debates} debates; "
              + ", ".join(f"{n} {label}" for label, n in counts.items()) + " #####")
        if min(counts.values()) == 0:
            print("  not scored: a class has no vote in this subset")
            continue
        table, results, n_pairs = score_rows(y, test, comparisons, rows["debate_id"], rows["before"], label_names,
                                             args.bootstrap, args.permutations, keep=keep)
        print_results(table, results, f"{int(keep.sum())} votes, {name}", n_pairs, label_names)
        out["subsets"][name] = {"test_rows": int(keep.sum()), "test_debates": debates, "class_counts": counts,
                                "same_debate_pairs": n_pairs, "rows": table, "comparisons": results}

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
