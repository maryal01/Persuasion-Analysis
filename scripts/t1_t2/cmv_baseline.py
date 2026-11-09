"""T1 shared baseline: score the CMV test split with the published ModernBERT model and cache it.

Reproduces the published original+reply model (nlp-final-project11/nlp-project-modernbert-cmv) on the
pinned `cmv` test split, checks the published 66.1% pairwise accuracy / 0.629 AUROC, and caches per-row
scores and numeric features (no text) for the T1 analyses that compare against it:

  scripts/t1_t2/evaluate_cmv_links.py       (reads <out-dir>/predictions.csv and reproduction.json)
  scripts/t1_t2/train_cmv_reply_only.py     (paired comparison with the cached scores)

It also holds the helpers those scripts share: pinned revisions, pairing by `pair_id`, win scores,
post-cluster bootstrap and reply scoring.

Usage:
  python scripts/t1_t2/cmv_baseline.py --out-dir outputs/cmv_length

Pairs are matched by the dataset's `pair_id` column, never by row order. Inputs mirror
scripts/train_delta.py: tok(original, reply, truncation="longest_first", max_length=4096), and a
reply's score is logit[delta] - logit[nodelta], with label ids read from the model config. The
headline metrics reuse `pairwise_accuracy` and `binary_auroc` from scripts/train_delta.py.
predictions.csv holds ids, scores and numeric features only. The Hub token is read from the
environment (HF_TOKEN) by huggingface_hub and is never printed or stored.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede `import torch`

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
from train_delta import binary_auroc, pairwise_accuracy  # noqa: E402  (team metric definitions)


DATASET_REPO = "nlp-final-project11/persuasion-datasets"
DATASET_CONFIG = "cmv"
# t1-t2 branch head on 2026-11-09; the cmv config last changed 2026-10-26, before the model upload
DATASET_REVISION = "e236e8d98785ad4c80a3af422c656061f373f825"
MODEL_REPO = "nlp-final-project11/nlp-project-modernbert-cmv"
# main head on 2026-11-09 (adds only the model card to the 2026-10-27 weights upload e4c86c70)
MODEL_REVISION = "407888b2c6395a52498173ab686b91fdce3f9b73"
MAX_LENGTH = 4096  # as scripts/train_delta.py and the model card
PUBLISHED_PAIRWISE_ACC = 0.661  # RESULTS.md, rounded to 0.1 points
PUBLISHED_AUROC = 0.629  # RESULTS.md, rounded to 0.001
PAIRWISE_TOLERANCE = 0.003  # 0.3 points, about 3 of 1,019 pairs
AUROC_TOLERANCE = 0.005
EXPECTED_TEST_PAIRS = 1019

LINK_RE = re.compile(r"https?://|www\.|\]\(")
QUOTE_LINE_RE = re.compile(r"^\s*(>|&gt;)", re.MULTILINE)

PREDICTION_FIELDS = [
    "row", "pair_id", "post_id", "label", "logit_nodelta", "logit_delta", "score",
    "input_tokens", "truncated", "reply_words", "reply_chars", "reply_tokens", "original_tokens",
    "n_links", "n_quote_lines",
]



# --------------------------------------------------------------------------------------------
# Pairing and per-pair quantities (pure functions, unit-tested)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    pair_id: int
    post_id: str
    delta_row: int
    nodelta_row: int


def build_pairs(pair_ids: Sequence[int], post_ids: Sequence[str], labels: Sequence[int],
                delta_label: int) -> list[Pair]:
    """Group rows by pair_id (row order is ignored). Raises ValueError unless every pair has exactly
    one delta row and one no-delta row from the same post."""
    rows_by_pair: dict[int, list[int]] = defaultdict(list)
    for row, pair_id in enumerate(pair_ids):
        rows_by_pair[int(pair_id)].append(row)
    pairs, problems = [], []
    for pair_id, rows in sorted(rows_by_pair.items()):
        delta_rows = [r for r in rows if int(labels[r]) == delta_label]
        other_rows = [r for r in rows if int(labels[r]) != delta_label]
        if len(rows) != 2 or len(delta_rows) != 1 or len(other_rows) != 1:
            problems.append(f"pair {pair_id}: {len(delta_rows)} delta / {len(other_rows)} other rows")
            continue
        if post_ids[delta_rows[0]] != post_ids[other_rows[0]]:
            problems.append(f"pair {pair_id}: rows come from different posts")
            continue
        pairs.append(Pair(pair_id, str(post_ids[delta_rows[0]]), delta_rows[0], other_rows[0]))
    if problems:
        raise ValueError(f"{len(problems)} malformed pairs, e.g. " + "; ".join(problems[:5]))
    return pairs


def win_score(delta_values: np.ndarray, nodelta_values: np.ndarray) -> np.ndarray:
    """1 where the delta side is larger, 0.5 on ties, 0 otherwise (the team's tie convention)."""
    delta_values, nodelta_values = np.asarray(delta_values, float), np.asarray(nodelta_values, float)
    return (delta_values > nodelta_values) + 0.5 * (delta_values == nodelta_values)


def length_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shorter / longer length, in [0, 1]; 1 when both are 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    longer = np.maximum(a, b)
    return np.where(longer > 0, np.minimum(a, b) / np.where(longer > 0, longer, 1), 1.0)


def cluster_bootstrap_means(cluster_ids: Sequence[str], columns: dict[str, np.ndarray],
                            n_boot: int, seed: int, chunk: int = 2000) -> dict[str, np.ndarray]:
    """Post-cluster bootstrap: resample clusters (posts) with replacement, keeping all of a post's
    pairs together, and return n_boot replicate means of each per-pair column."""
    unique, inverse = np.unique(np.asarray(cluster_ids), return_inverse=True)
    n_clusters = len(unique)
    counts = np.bincount(inverse, minlength=n_clusters).astype(float)
    sums = {k: np.bincount(inverse, weights=np.asarray(v, float), minlength=n_clusters)
            for k, v in columns.items()}
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {k: [] for k in columns}
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        weights = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=size).astype(float)
        denom = weights @ counts
        for k in columns:
            out[k].append((weights @ sums[k]) / denom)
    return {k: np.concatenate(v) for k, v in out.items()}


def percentile_ci(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(replicates, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def rank_average(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank_average(np.asarray(x)), rank_average(np.asarray(y)))[0, 1])


def text_features(reply: str) -> dict[str, int]:
    return {"reply_words": len(reply.split()), "reply_chars": len(reply),
            "n_links": len(LINK_RE.findall(reply)), "n_quote_lines": len(QUOTE_LINE_RE.findall(reply))}


def is_git_ignored(path: Path) -> bool:
    try:
        return subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True).returncode == 0
    except OSError:
        return False



# --------------------------------------------------------------------------------------------
# Score the test split + reproduction check
# --------------------------------------------------------------------------------------------

def pick_device(requested: str):
    import torch
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[DEVICE] MPS not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def score_rows(model, tok, originals: list[str], replies: list[str], rows: Sequence[int], device,
               label_ids: tuple[int, int], log_every: int = 200) -> dict[int, tuple[float, float, int]]:
    """Batch size 1 (no padding, exactly as each example was seen alone). Returns
    {row: (logit_nodelta, logit_delta, n_input_tokens)}."""
    import torch
    nodelta_id, delta_id = label_ids
    out = {}
    start = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            enc = tok(originals[row], replies[row], truncation="longest_first", max_length=MAX_LENGTH,
                      return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0].float().cpu()
            out[row] = (float(logits[nodelta_id]), float(logits[delta_id]), int(enc["input_ids"].shape[1]))
            if log_every and (i % log_every == 0 or i == len(rows)):
                print(f"  [{device.type}] {i}/{len(rows)} rows, {time.time() - start:.0f}s", flush=True)
    return out


def run_predict(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_dataset(DATASET_REPO, DATASET_CONFIG, revision=args.dataset_revision, split="test")
    originals, replies = list(test["original"]), list(test["reply"])
    labels, pair_ids, post_ids = list(test["label"]), list(test["pair_id"]), list(test["post_id"])
    dataset_names = test.features["label"].names

    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=args.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO, revision=args.model_revision, dtype=torch.float32)
    model.eval()

    # label mapping from the model config, cross-checked against the dataset's ClassLabel
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    delta_id, nodelta_id = label2id["delta"], label2id["nodelta"]
    if dataset_names[delta_id] != "delta" or dataset_names[nodelta_id] != "nodelta":
        raise ValueError(f"label mismatch: model {model.config.label2id} vs dataset {dataset_names}")

    pairs = build_pairs(pair_ids, post_ids, labels, delta_label=delta_id)
    print(f"[DATA] {len(test)} rows, {len(pairs)} pairs, {len(set(post_ids))} posts "
          f"(dataset {args.dataset_revision[:8]}, model {args.model_revision[:8]})")
    full_run = args.limit_pairs is None
    if not full_run:
        pairs = pairs[: args.limit_pairs]
        print(f"[DATA] --limit-pairs {args.limit_pairs}: smoke run, the reproduction check is skipped")
    elif len(pairs) != EXPECTED_TEST_PAIRS:
        print(f"[WARN] expected {EXPECTED_TEST_PAIRS} pairs, found {len(pairs)}")
    rows = sorted(r for p in pairs for r in (p.delta_row, p.nodelta_row))

    device = pick_device(args.device)
    print(f"[DEVICE] {device}; torch {torch.__version__}, transformers {transformers.__version__}")
    model.to(device)
    t0 = time.time()
    scored = score_rows(model, tok, originals, replies, rows, device, (nodelta_id, delta_id))
    elapsed = time.time() - t0

    # Cross-device check: re-score a random sample on CPU and compare with the MPS scores
    device_check = None
    if device.type != "cpu" and args.cpu_check_rows > 0:
        sample = sorted(np.random.default_rng(0).choice(rows, size=min(args.cpu_check_rows, len(rows)),
                                                        replace=False).tolist())
        print(f"[CHECK] re-scoring {len(sample)} random rows on CPU")
        model.to("cpu")
        cpu = score_rows(model, tok, originals, replies, sample, torch.device("cpu"),
                         (nodelta_id, delta_id), log_every=0)
        diffs = [abs((scored[r][1] - scored[r][0]) - (cpu[r][1] - cpu[r][0])) for r in sample]
        device_check = {"rows": len(sample), "max_abs_score_diff": float(max(diffs)),
                        "mean_abs_score_diff": float(np.mean(diffs))}
        print(f"[CHECK] |score(mps) - score(cpu)|: max {device_check['max_abs_score_diff']:.2e}, "
              f"mean {device_check['mean_abs_score_diff']:.2e}")

    reply_tokens = [len(x) for x in tok([replies[r] for r in rows], add_special_tokens=False)["input_ids"]]
    original_tokens = [len(x) for x in tok([originals[r] for r in rows], add_special_tokens=False)["input_ids"]]
    records = []
    for r, n_reply_tok, n_orig_tok in zip(rows, reply_tokens, original_tokens):
        logit_n, logit_d, n_input = scored[r]
        full_len = n_reply_tok + n_orig_tok + 3  # [CLS] original [SEP] reply [SEP]
        records.append({"row": r, "pair_id": int(pair_ids[r]), "post_id": post_ids[r], "label": int(labels[r]),
                        "logit_nodelta": logit_n, "logit_delta": logit_d, "score": logit_d - logit_n,
                        "input_tokens": n_input, "truncated": int(full_len > MAX_LENGTH),
                        "reply_tokens": n_reply_tok, "original_tokens": n_orig_tok, **text_features(replies[r])})
    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows({k: rec[k] for k in PREDICTION_FIELDS} for rec in records)

    # Headline metrics with the team's own functions, plus an independent per-pair recount
    scores = np.array([rec["score"] for rec in records])
    rec_labels = np.array([rec["label"] for rec in records])
    team_pairwise = pairwise_accuracy(scores, (rec_labels == delta_id).astype(int), [rec["pair_id"] for rec in records])
    team_auroc = binary_auroc(scores, (rec_labels == delta_id).astype(int))
    by_row = {rec["row"]: rec["score"] for rec in records}
    recount = float(np.mean(win_score([by_row[p.delta_row] for p in pairs], [by_row[p.nodelta_row] for p in pairs])))
    if abs(recount - team_pairwise) > 1e-12:
        raise AssertionError(f"pair recount {recount} disagrees with train_delta.pairwise_accuracy {team_pairwise}")
    preds = np.array([rec["logit_delta"] > rec["logit_nodelta"] for rec in records])
    truth = rec_labels == delta_id

    acc_ok = abs(team_pairwise - PUBLISHED_PAIRWISE_ACC) <= PAIRWISE_TOLERANCE
    auroc_ok = abs(team_auroc - PUBLISHED_AUROC) <= AUROC_TOLERANCE
    result = {
        "passed": bool(full_run and acc_ok and auroc_ok),
        "full_run": full_run,
        "pairs": len(pairs), "rows": len(records), "posts": len({p.post_id for p in pairs}),
        "pairwise_acc": team_pairwise, "auroc": team_auroc,
        "published_pairwise_acc": PUBLISHED_PAIRWISE_ACC, "published_auroc": PUBLISHED_AUROC,
        "pairwise_tolerance": PAIRWISE_TOLERANCE, "auroc_tolerance": AUROC_TOLERANCE,
        "row_accuracy": float((preds == truth).mean()),
        "share_predicted_delta": float(preds.mean()),
        "truncated_rows": int(sum(rec["truncated"] for rec in records)),
        "device": str(device), "device_check": device_check, "inference_seconds": round(elapsed, 1),
        "dataset": {"repo": DATASET_REPO, "config": DATASET_CONFIG, "revision": args.dataset_revision},
        "model": {"repo": MODEL_REPO, "revision": args.model_revision, "label2id": model.config.label2id},
        "max_length": MAX_LENGTH,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    (out_dir / "reproduction.json").write_text(json.dumps(result, indent=2))

    print("\n=== REPRODUCTION CHECK ===")
    print(f"  pairwise acc : {team_pairwise:.4f}  (published {PUBLISHED_PAIRWISE_ACC:.3f}, "
          f"tolerance ±{PAIRWISE_TOLERANCE:.3f}) -> {'OK' if acc_ok else 'MISMATCH'}")
    print(f"  AUROC        : {team_auroc:.4f}  (published {PUBLISHED_AUROC:.3f}, "
          f"tolerance ±{AUROC_TOLERANCE:.3f}) -> {'OK' if auroc_ok else 'MISMATCH'}")
    print(f"  row accuracy : {result['row_accuracy']:.4f}; predicted delta for {result['share_predicted_delta']:.1%} of rows")
    print(f"  pairs {len(pairs)}, posts {result['posts']}, truncated rows {result['truncated_rows']}, "
          f"inference {elapsed:.0f}s on {device}")
    print(f"  wrote {out_dir / 'predictions.csv'} and {out_dir / 'reproduction.json'}")
    if not full_run:
        print("  RESULT: smoke run only (not a reproduction)")
        return 0
    print(f"  RESULT: {'PASSED' if result['passed'] else 'FAILED; investigate before using the cached scores'}")
    return 0 if result["passed"] else 1

# --------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="outputs/cmv_length", help="git-ignored output directory (default: %(default)s)")
    ap.add_argument("--dataset-revision", default=DATASET_REVISION, help="dataset commit (default: pinned)")
    ap.add_argument("--model-revision", default=MODEL_REVISION, help="model commit (default: pinned)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="auto = MPS if available, else CPU")
    ap.add_argument("--cpu-check-rows", type=int, default=40, help="rows re-scored on CPU to compare devices (0 = skip)")
    ap.add_argument("--limit-pairs", type=int, default=None, help="smoke run on the first N pairs (no reproduction check)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run_predict(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
