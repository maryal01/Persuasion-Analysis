"""T1 length-controlled evaluation of the trained CMV ModernBERT model (no retraining).

Research question: can the model pick the delta reply when the delta and no-delta replies of a
pair have similar lengths? The published "longer reply wins" baseline (55.3%) shows length alone
carries signal; this script measures pairwise accuracy on pairs whose length ratio
(shorter / longer reply) is >= 0.80, >= 0.90, >= 0.95, and < 0.80.

Two stages, run in order:

  predict  Score every test row with the published checkpoint (pinned revisions), cache per-row
           scores and numeric features (no text) to <out-dir>/predictions.csv, and check that the
           published 66.1% pairwise accuracy / 0.629 AUROC are reproduced.
  analyze  Length-controlled pairwise accuracy with post-cluster bootstrap CIs (primary), Wilson
           intervals (descriptive), the longer-reply baseline per subset, accuracy by length
           ratio, and error breakdowns. Refuses to run unless `predict` passed the check.

Usage:
  python scripts/t1_t2/evaluate_cmv_length.py predict --out-dir outputs/cmv_length
  python scripts/t1_t2/evaluate_cmv_length.py analyze --out-dir outputs/cmv_length

Pairs are matched by the dataset's `pair_id` column, never by row order. Inputs mirror
scripts/train_delta.py: tok(original, reply, truncation="longest_first", max_length=4096), and a
reply's score is logit[delta] - logit[nodelta], with label ids read from the model config. The
headline metrics reuse `pairwise_accuracy` and `binary_auroc` from scripts/train_delta.py.

Private text never leaves the git-ignored output directory: predictions.csv holds ids, scores and
numeric features only; `analyze --export-error-text` writes reply text for inspection, and only
into an output directory that git ignores. The Hub token is read from the environment (HF_TOKEN)
by huggingface_hub and is never printed or stored.
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
from typing import Callable, Sequence

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

LENGTH_UNITS = ("words", "tokens", "chars")
PRIMARY_UNIT = "words"  # whitespace words reproduce the published 55.3% longer-reply baseline
SUBSETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "all": lambda r: np.ones_like(r, dtype=bool),
    "ratio >= 0.80": lambda r: r >= 0.80,
    "ratio >= 0.90": lambda r: r >= 0.90,
    "ratio >= 0.95": lambda r: r >= 0.95,
    "ratio < 0.80": lambda r: r < 0.80,
}
RATIO_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0000001)

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


def wilson_interval(mean: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion (fractional successes from ties are allowed).
    Assumes independent pairs, so it is descriptive only here."""
    if n == 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (mean + z * z / (2 * n)) / denom
    half = z * np.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


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


def weighted_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                      iters: int = 50) -> tuple[float, float]:
    """Intercept and slope of a one-predictor logistic regression by Newton's method; y may be
    fractional (tie = 0.5) and w are per-observation weights (bootstrap multiplicities)."""
    X = np.column_stack([np.ones_like(x, dtype=float), np.asarray(x, float)])
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (y - p))
        hess = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def text_features(reply: str) -> dict[str, int]:
    return {"reply_words": len(reply.split()), "reply_chars": len(reply),
            "n_links": len(LINK_RE.findall(reply)), "n_quote_lines": len(QUOTE_LINE_RE.findall(reply))}


# --------------------------------------------------------------------------------------------
# Stage 1: predict + reproduction check
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
    print(f"  RESULT: {'PASSED' if result['passed'] else 'FAILED; investigate before running analyze'}")
    return 0 if result["passed"] else 1


# --------------------------------------------------------------------------------------------
# Stage 2: length-controlled analysis
# --------------------------------------------------------------------------------------------

def load_pair_table(out_dir: Path) -> dict[str, np.ndarray]:
    """predictions.csv -> per-pair arrays (d_* = delta reply, n_* = no-delta reply)."""
    with open(out_dir / "predictions.csv") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads((out_dir / "reproduction.json").read_text())
    delta_id = int({k.lower(): v for k, v in meta["model"]["label2id"].items()}["delta"])
    pairs = build_pairs([int(r["pair_id"]) for r in rows], [r["post_id"] for r in rows],
                        [int(r["label"]) for r in rows], delta_label=delta_id)
    t: dict[str, np.ndarray] = {"pair_id": np.array([p.pair_id for p in pairs]),
                                "post_id": np.array([p.post_id for p in pairs])}
    numeric = ["score", "truncated", "reply_words", "reply_chars", "reply_tokens", "original_tokens",
               "n_links", "n_quote_lines"]
    for col in numeric:
        t["d_" + col] = np.array([float(rows[p.delta_row][col]) for p in pairs])
        t["n_" + col] = np.array([float(rows[p.nodelta_row][col]) for p in pairs])
    t["model_win"] = win_score(t["d_score"], t["n_score"])
    t["score_gap"] = t["d_score"] - t["n_score"]
    for unit in LENGTH_UNITS:
        d, n = t[f"d_reply_{unit}"], t[f"n_reply_{unit}"]
        t[f"ratio_{unit}"] = length_ratio(d, n)
        t[f"longer_win_{unit}"] = win_score(d, n)
    return t


def accuracy_row(name: str, mask: np.ndarray, t: dict[str, np.ndarray], unit: str,
                 n_boot: int, seed: int) -> dict:
    model, base = t["model_win"][mask], t[f"longer_win_{unit}"][mask]
    posts = t["post_id"][mask]
    row = {"subset": name, "pairs": int(mask.sum()), "posts": int(len(set(posts)))}
    if row["pairs"] == 0:
        return row
    boot = cluster_bootstrap_means(posts, {"model": model, "base": base, "diff": model - base}, n_boot, seed)
    row.update(
        model_acc=float(model.mean()), model_ci=percentile_ci(boot["model"]),
        model_wilson=wilson_interval(float(model.mean()), row["pairs"]),
        longer_acc=float(base.mean()), longer_ci=percentile_ci(boot["base"]),
        diff=float(model.mean() - base.mean()), diff_ci=percentile_ci(boot["diff"]),
        share_diff_le_0=float((boot["diff"] <= 0).mean()),
    )
    return row


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{fmt_pct(ci[0])}, {fmt_pct(ci[1])}]"


def subset_table_md(rows: list[dict]) -> str:
    lines = ["| Subset | Pairs | Posts | ModernBERT % | 95% cluster CI | Wilson CI (descr.) | Longer reply % | 95% cluster CI | Diff (pts) | 95% cluster CI |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["pairs"] == 0:
            lines.append(f"| {r['subset']} | 0 | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(f"| {r['subset']} | {r['pairs']} | {r['posts']} | {fmt_pct(r['model_acc'])} | {fmt_ci(r['model_ci'])} | "
                     f"{fmt_ci(r['model_wilson'])} | {fmt_pct(r['longer_acc'])} | {fmt_ci(r['longer_ci'])} | "
                     f"{100 * r['diff']:+.1f} | [{100 * r['diff_ci'][0]:+.1f}, {100 * r['diff_ci'][1]:+.1f}] |")
    return "\n".join(lines)


def group_table_md(title: str, rows: list[dict]) -> str:
    lines = [f"**{title}**", "", "| Group | Pairs | Posts | ModernBERT % | 95% cluster CI |", "|---|---|---|---|---|"]
    for r in rows:
        if r["pairs"]:
            lines.append(f"| {r['subset']} | {r['pairs']} | {r['posts']} | {fmt_pct(r['model_acc'])} | {fmt_ci(r['model_ci'])} |")
        else:
            lines.append(f"| {r['subset']} | 0 | 0 | – | – |")
    return "\n".join(lines)


def is_git_ignored(path: Path) -> bool:
    try:
        return subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True).returncode == 0
    except OSError:
        return False


def run_analyze(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    meta = json.loads((out_dir / "reproduction.json").read_text())
    if not meta.get("passed") and not args.allow_failed_reproduction:
        print(f"[STOP] reproduction check did not pass (pairwise {meta.get('pairwise_acc')}, "
              f"AUROC {meta.get('auroc')}, full_run {meta.get('full_run')}). Fix that first.")
        return 2
    t = load_pair_table(out_dir)
    n_boot, seed = args.n_boot, args.seed
    md: list[str] = []
    metrics: dict = {"reproduction": {k: meta[k] for k in ("pairwise_acc", "auroc", "pairs", "posts")},
                     "n_boot": n_boot, "seed": seed}

    # 1) Length-controlled subsets, primary unit + robustness units
    metrics["subsets"] = {}
    for unit in LENGTH_UNITS:
        rows = [accuracy_row(name, fn(t[f"ratio_{unit}"]), t, unit, n_boot, seed) for name, fn in SUBSETS.items()]
        metrics["subsets"][unit] = rows
        label = "primary" if unit == PRIMARY_UNIT else "robustness"
        md += [f"### Length = {unit} ({label})", "", subset_table_md(rows), ""]

    # 2) Exploratory: accuracy by ratio bin (all pairs) and logistic trend
    ratio = t[f"ratio_{PRIMARY_UNIT}"]
    bin_rows = []
    for lo, hi in zip(RATIO_BINS[:-1], RATIO_BINS[1:]):
        name = f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}"
        bin_rows.append(accuracy_row(name, (ratio >= lo) & (ratio < hi), t, PRIMARY_UNIT, n_boot, seed))
    metrics["ratio_bins"] = bin_rows
    intercept, slope = weighted_logistic(ratio, t["model_win"])
    unique_posts, inverse = np.unique(t["post_id"], return_inverse=True)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(args.n_boot_logistic):
        w_post = rng.multinomial(len(unique_posts), np.full(len(unique_posts), 1 / len(unique_posts)))
        slopes.append(weighted_logistic(ratio, t["model_win"], w_post[inverse])[1])
    metrics["logistic"] = {"intercept": intercept, "slope_per_unit_ratio": slope,
                           "slope_ci": percentile_ci(np.array(slopes)), "n_boot": args.n_boot_logistic}
    md += ["### Accuracy by word-length ratio (exploratory, all pairs)", "", subset_table_md(bin_rows), "",
           f"Logistic regression of pair correctness on word-length ratio: slope {slope:+.2f} per unit of ratio, "
           f"95% post-cluster CI [{metrics['logistic']['slope_ci'][0]:+.2f}, {metrics['logistic']['slope_ci'][1]:+.2f}] "
           f"({args.n_boot_logistic} resamples).", ""]

    # 3) Error analysis
    d_w, n_w = t[f"d_reply_{PRIMARY_UNIT}"], t[f"n_reply_{PRIMARY_UNIT}"]
    model_picks_longer = np.where(t["score_gap"] > 0, d_w > n_w, n_w > d_w).astype(float)
    model_picks_longer[d_w == n_w] = 0.5
    log_len_ratio = np.log((d_w + 1) / (n_w + 1))
    err: dict = {
        "model_picks_longer_reply": float(model_picks_longer.mean()),
        "delta_is_longer": float(t[f"longer_win_{PRIMARY_UNIT}"].mean()),
        "spearman_scoregap_vs_log_word_ratio": spearman(t["score_gap"], log_len_ratio),
        "median_abs_gap_correct": float(np.median(np.abs(t["score_gap"][t["model_win"] == 1]))),
        "median_abs_gap_wrong": float(np.median(np.abs(t["score_gap"][t["model_win"] == 0]))),
    }
    subsets_longer = {name: float(model_picks_longer[fn(ratio)].mean()) for name, fn in SUBSETS.items() if fn(ratio).any()}
    err["model_picks_longer_by_subset"] = subsets_longer
    md += ["### Error analysis (word lengths)", "",
           f"- The model picks the longer reply in {fmt_pct(err['model_picks_longer_reply'])}% of pairs; "
           f"the delta reply is the longer one in {fmt_pct(err['delta_is_longer'])}%.",
           "- Model picks the longer reply, by subset: "
           + ", ".join(f"{k}: {fmt_pct(v)}%" for k, v in subsets_longer.items()) + ".",
           f"- Spearman correlation of score gap (delta − no-delta) with log word ratio (delta / no-delta): "
           f"{err['spearman_scoregap_vs_log_word_ratio']:+.3f}.",
           f"- Median |score gap|: correct pairs {err['median_abs_gap_correct']:.3f}, wrong pairs {err['median_abs_gap_wrong']:.3f}.", ""]

    def groups(title: str, named_masks: list[tuple[str, np.ndarray]]) -> None:
        rows = [accuracy_row(name, mask, t, PRIMARY_UNIT, n_boot, seed) for name, mask in named_masks]
        err.setdefault("breakdowns", {})[title] = rows
        md.extend([group_table_md(title, rows), ""])

    groups("Which reply is longer (words)", [("delta reply longer", d_w > n_w), ("no-delta reply longer", d_w < n_w),
                                              ("same length", d_w == n_w)])
    trunc = (t["d_truncated"] + t["n_truncated"]) > 0
    groups("Truncation (> 4,096 tokens)", [("neither reply truncated", ~trunc), ("a reply truncated", trunc)])
    d_link, n_link = t["d_n_links"] > 0, t["n_n_links"] > 0
    groups("Links", [("no links", ~d_link & ~n_link), ("delta reply only", d_link & ~n_link),
                     ("no-delta reply only", ~d_link & n_link), ("both", d_link & n_link)])
    quotes = (t["d_n_quote_lines"] + t["n_n_quote_lines"]) > 0
    groups("Quoted lines ('>')", [("no quotes", ~quotes), ("a reply quotes", quotes)])
    orig = t["d_original_tokens"]
    q1, q2 = np.percentile(orig, [100 / 3, 200 / 3])
    groups("Original post length (tokens, terciles)", [(f"short (< {q1:.0f})", orig < q1),
                                                       (f"medium ({q1:.0f}–{q2:.0f})", (orig >= q1) & (orig < q2)),
                                                       (f"long (>= {q2:.0f})", orig >= q2)])
    pairs_per_post = {p: c for p, c in zip(*np.unique(t["post_id"], return_counts=True))}
    multi = np.array([pairs_per_post[p] > 1 for p in t["post_id"]])
    groups("Pairs per post", [("post with 1 pair", ~multi), ("post with 2+ pairs", multi)])
    gap = np.abs(t["score_gap"])
    g = np.percentile(gap, [25, 50, 75])
    groups("|Score gap| quartile (confidence)", [("Q1 (least confident)", gap < g[0]), ("Q2", (gap >= g[0]) & (gap < g[1])),
                                                 ("Q3", (gap >= g[1]) & (gap < g[2])), ("Q4 (most confident)", gap >= g[2])])
    metrics["error_analysis"] = err

    # Most confident errors: ids and numbers only
    wrong = np.where(t["model_win"] == 0)[0]
    worst = wrong[np.argsort(-gap[wrong])][: args.top_errors]
    with open(out_dir / "confident_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "post_id", "score_gap", "delta_words", "nodelta_words", "word_ratio", "truncated"])
        for i in worst:
            w.writerow([int(t["pair_id"][i]), t["post_id"][i], f"{t['score_gap'][i]:.4f}", int(d_w[i]), int(n_w[i]),
                        f"{ratio[i]:.3f}", int(trunc[i])])

    if args.export_error_text:
        target = out_dir / "confident_errors_text.jsonl"
        if not is_git_ignored(target):
            print(f"[STOP] {target} is not git-ignored; refusing to write private text there")
            return 2
        from datasets import load_dataset
        test = load_dataset(DATASET_REPO, DATASET_CONFIG, revision=meta["dataset"]["revision"], split="test")
        keep = {int(t["pair_id"][i]) for i in worst}
        delta_id = int({k.lower(): v for k, v in meta["model"]["label2id"].items()}["delta"])
        by_pair: dict[int, dict] = defaultdict(dict)
        for ex in test:
            if int(ex["pair_id"]) in keep:
                by_pair[int(ex["pair_id"])]["original"] = ex["original"]
                by_pair[int(ex["pair_id"])]["delta" if ex["label"] == delta_id else "nodelta"] = ex["reply"]
        with open(target, "w") as fh:
            for i in worst:
                fh.write(json.dumps({"pair_id": int(t["pair_id"][i]), "score_gap": float(t["score_gap"][i]),
                                     **by_pair[int(t["pair_id"][i])]}) + "\n")
        print(f"[TEXT] wrote {len(worst)} examples with text to {target} (git-ignored; keep local)")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out_dir / 'metrics.json'}, {out_dir / 'summary.md'}, {out_dir / 'confident_errors.csv'}")
    return 0


# --------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("predict", help="score the test split and check the published numbers")
    p.add_argument("--out-dir", default="outputs/cmv_length", help="git-ignored output directory (default: %(default)s)")
    p.add_argument("--dataset-revision", default=DATASET_REVISION, help="dataset commit (default: pinned)")
    p.add_argument("--model-revision", default=MODEL_REVISION, help="model commit (default: pinned)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="auto = MPS if available, else CPU")
    p.add_argument("--cpu-check-rows", type=int, default=40, help="rows re-scored on CPU to compare devices (0 = skip)")
    p.add_argument("--limit-pairs", type=int, default=None, help="smoke run on the first N pairs (no reproduction check)")

    a = sub.add_parser("analyze", help="length-controlled analysis from cached predictions")
    a.add_argument("--out-dir", default="outputs/cmv_length", help="directory written by `predict` (default: %(default)s)")
    a.add_argument("--n-boot", type=int, default=10000, help="post-cluster bootstrap resamples (default: %(default)s)")
    a.add_argument("--n-boot-logistic", type=int, default=2000, help="resamples for the logistic slope CI (default: %(default)s)")
    a.add_argument("--seed", type=int, default=577, help="bootstrap seed (default: %(default)s)")
    a.add_argument("--top-errors", type=int, default=25, help="confident errors listed by id (default: %(default)s)")
    a.add_argument("--export-error-text", action="store_true", help="also write their text locally (git-ignored dir only)")
    a.add_argument("--allow-failed-reproduction", action="store_true", help="analyze even if `predict` did not reproduce")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_predict(args) if args.stage == "predict" else run_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
