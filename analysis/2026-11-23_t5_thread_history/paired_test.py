"""Paired comparison of T5 history runs on the same test posts.

Loads the best checkpoint of each run (from its result_*.json written by scripts/t5/train.py), scores the test split again, saves the
scores (test_scores.npz) and compares every run with a reference run: AUROC difference with a 95% interval and a one-sided p value
from a bootstrap over OPs (the same resampled OPs for both runs).

Usage (from the repository root, one GPU, about 10 minutes for 5 runs; needs a Hugging Face login):
    python analysis/2026-11-23_t5_thread_history/paired_test.py --runs-dir <out_dir>/2026-11-23_t5_history_modernbert \
        --reference result_cmv_thread_history_post_len2048_seed0_grouped.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "t5"))
import train  # noqa: E402
from t5_common import posts_by_author  # noqa: E402


def run_args(result: dict) -> argparse.Namespace:
    """The training arguments of a run that matter for scoring. Results written before the key history_input existed get the chunk
    settings from the checkpoint name (..._<pooling>_<items>x<length>_seed...)."""
    if "history_input" in result:
        pooling, items, item_length = result["history_input"]
    elif result["mode"] == "post_history_chunks":
        pooling, size = Path(result["best_model"]).name.split("post_history_chunks_")[1].split("_")[:2]
        items, item_length = (int(x) for x in size.split("x"))
    else:
        pooling, items, item_length = None, 0, 0
    return argparse.Namespace(config=result["config"], mode=result["mode"], hub_revision=result["hub"]["commit"], smoke=False,
                              max_length=result["max_length"], batch_size=4, grad_accum=1, group_by_length=True, seed=result["seed"],
                              lr=result["lr"], head_lr=result["head_lr"], epochs=1, output_dir=Path("/nonexistent"),
                              history_pooling=pooling, history_items=items, item_length=item_length)


def score(result: dict, device: torch.device) -> Dict[str, np.ndarray]:
    args = run_args(result)
    data = train.prepare_data(args)
    trainer = train.Trainer(args, data, device, bf16=result["bf16"])
    trainer.model.load_state_dict(torch.load(result["best_model"], map_location=device))
    scores, labels = trainer.predict("test")
    return {"scores": scores, "labels": labels, "authors": np.array(data.test.authors)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, labels: np.ndarray, authors: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """AUROC(b) - AUROC(a) on the test posts, 95% interval and share of resamples with a difference <= 0 (OP bootstrap)."""
    groups = posts_by_author(list(authors))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n):
        posts = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if labels[posts].min() == labels[posts].max():
            continue
        differences.append(roc_auc_score(labels[posts], b[posts]) - roc_auc_score(labels[posts], a[posts]))
    differences = np.array(differences)
    return {"difference": float(roc_auc_score(labels, b) - roc_auc_score(labels, a)),
            "ci95": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
            "p_not_better": float((differences <= 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="result_*.json of the reference run")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = {}
    for path in sorted(args.runs_dir.glob("result_*.json")):
        cache = path.with_name(path.stem.replace("result_", "test_scores_") + ".npz")
        if not cache.exists():
            np.savez(cache, **score(json.loads(path.read_text()), device))
        runs[path.name] = dict(np.load(cache))
    reference = runs[args.reference]
    report = {}
    for name, run in runs.items():
        if not (run["labels"] == reference["labels"]).all() or not (run["authors"] == reference["authors"]).all():
            raise SystemExit(f"{name}: test posts differ from the reference")
        report[name] = {"test_auroc": float(roc_auc_score(run["labels"], run["scores"])),
                        "vs_reference": paired_bootstrap(reference["scores"], run["scores"], run["labels"], run["authors"])}
        r = report[name]["vs_reference"]
        print(f"{name:75s} AUROC {report[name]['test_auroc']:.4f}  diff {r['difference']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"p(not better) {r['p_not_better']:.3f}")
    (args.runs_dir / "paired_test.json").write_text(json.dumps({"reference": args.reference, "runs": report}, indent=1))


if __name__ == "__main__":
    main()
