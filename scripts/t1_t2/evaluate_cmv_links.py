"""T1 link-bias analysis of the trained CMV ModernBERT model (no retraining).

Question: is the frozen model sensitive to URL presence or content, i.e. could links be a shortcut
in its persuasion predictions? Three stages, all on the pinned `cmv` test split:

  score      Scores every test reply with the published model
             (nlp-final-project11/nlp-project-modernbert-cmv), checks that this reproduces the
             published 66.1% pairwise accuracy / 0.629 AUROC, and caches per-reply scores and numeric
             features (no text) in --cache-dir. Run once; the other stages read the cache.
  observe    Experiment A (observational). Groups pairs
             by which reply has a URL (neither / both / delta only / no-delta only), and reports pairwise
             accuracy with post-cluster bootstrap CIs, the joint view with length matching (word ratio
             >= 0.80), possible confounders per group, and the URL/label association in each split.
  intervene  Experiment B (controlled edits). Re-scores every reply that contains a URL in three
             variants: original (must reproduce the cached score), URL spans replaced by a fixed
             placeholder, and URL spans removed. Reports score-margin changes, preference flips and
             pairwise-accuracy changes with paired post-cluster bootstrap CIs.

Usage:
  python scripts/t1_t2/evaluate_cmv_links.py score     --cache-dir outputs/cmv_length
  python scripts/t1_t2/evaluate_cmv_links.py observe   --out-dir outputs/cmv_links
  python scripts/t1_t2/evaluate_cmv_links.py intervene --out-dir outputs/cmv_links

URL spans (see find_url_spans):
  markdown link  [anchor](target) with an http(s):// or www. target, as one span.
                 placeholder -> [anchor'](URL); remove -> anchor'  (anchor' = anchor with its own URLs
                 replaced / removed the same way)
  bare URL       http(s)://... or www.... ; balanced parentheses are allowed inside (Wikipedia links),
                 trailing punctuation .,;:!?'"* is not part of the URL. placeholder -> URL; remove -> ''
Not detected: bare domains without scheme or www (e.g. "example.com"), relative reddit links
("/r/...", "](/u/...)"), and URLs broken by whitespace.

Scoring mirrors scripts/train_delta.py: tok(original, reply, truncation="longest_first",
max_length=4096), score = logit[delta] - logit[nodelta], with label ids read from the model config.
Pairs are matched by the dataset's `pair_id` column, never by row order, and the headline metrics
reuse `pairwise_accuracy` and `binary_auroc` from scripts/train_delta.py. Outputs hold ids and numbers
only; reply text is written only with `intervene --export-flip-text`, and only into a git-ignored
directory. The Hub token is read from the environment (HF_TOKEN) by huggingface_hub and is never
printed or stored.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede `import torch`

import argparse
import csv
import difflib
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

PLACEHOLDER = "URL"
URL_START = r"(?:https?://|www\.)"
BARE_URL_RE = re.compile(URL_START + r"(?:[^\s()<>\[\]]|\([^\s()<>\[\]]*\))+", re.IGNORECASE)
MD_LINK_RE = re.compile(
    r"\[(?P<anchor>[^\[\]\n]*)\]\(\s*(?P<target>" + URL_START + r"(?:[^\s()]|\([^\s()]*\))+)"
    r"(?:\s+\"[^\"\n]*\")?\s*\)", re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?'\"*"
BARE_PREFIXES = ("http://", "https://", "www.")

CATEGORIES = ("neither", "both", "delta only", "no-delta only")
VARIANTS = ("placeholder", "remove")
SCENARIOS = ("edit all URL replies", "edit delta reply only", "edit no-delta reply only")
REPRO_TOLERANCE = 1e-3  # max |rescored - cached| score for unedited replies


# --------------------------------------------------------------------------------------------
# Pairing and per-pair quantities (pure functions)
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


def run_score(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(args.cache_dir)
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
# URL detection and edits (pure functions)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UrlSpan:
    kind: str  # "markdown" or "bare"
    start: int
    end: int
    anchor_start: int = -1  # markdown only
    anchor_end: int = -1


def find_bare_urls(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """(start, end) of plain URLs, with trailing punctuation excluded."""
    spans = []
    for m in BARE_URL_RE.finditer(text):
        start, end = m.start(), m.end()
        while end > start and text[end - 1] in TRAILING_PUNCT:
            end -= 1
        if text[start:end].lower() in BARE_PREFIXES:
            continue
        spans.append((start + offset, end + offset))
    return spans


def find_url_spans(text: str) -> list[UrlSpan]:
    """Markdown links with a URL target, then plain URLs outside them, sorted by position."""
    spans = [UrlSpan("markdown", m.start(), m.end(), m.start("anchor"), m.end("anchor"))
             for m in MD_LINK_RE.finditer(text)]
    md_ranges = [(s.start, s.end) for s in spans]
    for start, end in find_bare_urls(text):
        if not any(s < end and start < e for s, e in md_ranges):
            spans.append(UrlSpan("bare", start, end))
    return sorted(spans, key=lambda s: s.start)


def _replace_bare(text: str, replacement: str) -> str:
    out, pos = [], 0
    for start, end in find_bare_urls(text):
        out += [text[pos:start], replacement]
        pos = end
    return "".join(out) + text[pos:]


def edit_urls(text: str, variant: str, placeholder: str = PLACEHOLDER) -> str:
    """variant 'original' (unchanged), 'placeholder' or 'remove'; see the module docstring."""
    if variant == "original":
        return text
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    replacement = placeholder if variant == "placeholder" else ""
    out, pos = [], 0
    for span in find_url_spans(text):
        out.append(text[pos:span.start])
        if span.kind == "bare":
            out.append(replacement)
        else:
            anchor = _replace_bare(text[span.anchor_start:span.anchor_end], replacement)
            out.append(f"[{anchor}]({placeholder})" if variant == "placeholder" else anchor)
        pos = span.end
    return "".join(out) + text[pos:]


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _diff_within_spans(original: str, edited: str, spans: Sequence[UrlSpan]) -> bool:
    """Every non-equal block of a character diff lies inside a URL span. Sound, but can reject a
    valid edit when difflib aligns a span character with identical text just outside the span."""
    ops = difflib.SequenceMatcher(None, original, edited, autojunk=False).get_opcodes()
    return all(tag == "equal" or any(s.start <= i1 and i2 <= s.end for s in spans) for tag, i1, i2, _, _ in ops)


def _gaps_preserved(original: str, edited: str, spans: Sequence[UrlSpan], placeholder: str = PLACEHOLDER) -> bool:
    """Alignment-free check: the text between URL spans appears in `edited` verbatim and in order
    (first gap at the start, last gap at the end), and each edited stretch, after dropping the
    placeholder and markdown brackets, is a subsequence of the original span(s) it replaces."""
    groups: list[list[int]] = []  # adjacent spans (empty gap) are checked together
    for s in sorted(spans, key=lambda s: s.start):
        if groups and groups[-1][1] == s.start:
            groups[-1][1] = s.end
        else:
            groups.append([s.start, s.end])
    if not groups:
        return original == edited
    gaps = [original[:groups[0][0]]] + [original[a[1]:b[0]] for a, b in zip(groups, groups[1:])] + [original[groups[-1][1]:]]
    if not (edited.startswith(gaps[0]) and edited.endswith(gaps[-1])):
        return False
    cursor, end_limit = len(gaps[0]), len(edited) - len(gaps[-1])
    if end_limit < cursor:
        return False
    for i, (start, end) in enumerate(groups):
        if i + 1 < len(groups):
            pos = edited.find(gaps[i + 1], cursor, end_limit)
            if pos < 0:
                return False
            stretch, cursor = edited[cursor:pos], pos + len(gaps[i + 1])
        else:
            stretch = edited[cursor:end_limit]
        residue = stretch.replace(placeholder, "") if placeholder else stretch
        residue = re.sub(r"[\[\]()]", "", residue)
        if not _is_subsequence(residue, original[start:end]):
            return False
    return True


def changes_within_spans(original: str, edited: str, spans: Sequence[UrlSpan], placeholder: str = PLACEHOLDER) -> bool:
    """Independent check that `edited` differs from `original` only inside URL spans. Passes if
    either sound check passes: the character diff, or the alignment-free gap check."""
    return _diff_within_spans(original, edited, spans) or _gaps_preserved(original, edited, spans, placeholder)


def link_category(delta_has_url: bool, nodelta_has_url: bool) -> str:
    if delta_has_url and nodelta_has_url:
        return "both"
    if delta_has_url:
        return "delta only"
    if nodelta_has_url:
        return "no-delta only"
    return "neither"


def group_mean_replicates(clusters: Sequence[str], values: np.ndarray, member: np.ndarray,
                          n_boot: int, seed: int) -> np.ndarray:
    """Post-cluster bootstrap replicates of mean(values[member]) when the bootstrap resamples all
    clusters (so a group's size varies across replicates). nan where a replicate has no members."""
    member = np.asarray(member, float)
    b = cluster_bootstrap_means(clusters, {"num": np.asarray(values, float) * member, "den": member},
                                    n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b["den"] > 0, b["num"] / np.where(b["den"] > 0, b["den"], 1), np.nan)


def nan_ci(replicates: np.ndarray) -> tuple[float, float]:
    r = replicates[~np.isnan(replicates)]
    return percentile_ci(r) if len(r) else (float("nan"), float("nan"))


# --------------------------------------------------------------------------------------------
# Shared loading
# --------------------------------------------------------------------------------------------

def load_test_with_cache(cache_dir: Path):
    """Test split at the revision recorded by the `score` stage, its cached scores, and verified pairs."""
    from datasets import load_dataset
    meta = json.loads((cache_dir / "reproduction.json").read_text())
    if not meta.get("passed"):
        raise SystemExit(f"[STOP] {cache_dir}/reproduction.json did not pass; rerun the `score` stage")
    test = load_dataset(DATASET_REPO, DATASET_CONFIG, revision=meta["dataset"]["revision"], split="test")
    replies, originals = list(test["reply"]), list(test["original"])
    labels, pair_ids, post_ids = list(test["label"]), list(test["pair_id"]), list(test["post_id"])
    delta_id = int({k.lower(): v for k, v in meta["model"]["label2id"].items()}["delta"])
    with open(cache_dir / "predictions.csv") as fh:
        cached = {int(r["row"]): r for r in csv.DictReader(fh)}
    if sorted(cached) != list(range(len(test))):
        raise SystemExit("[STOP] cached predictions do not cover every test row")
    for row, rec in cached.items():  # same rows, same order, same text as when the scores were cached
        if (int(rec["pair_id"]) != pair_ids[row] or int(rec["label"]) != labels[row]
                or int(rec["reply_chars"]) != len(replies[row]) or int(rec["reply_words"]) != len(replies[row].split())):
            raise SystemExit(f"[STOP] cached row {row} does not match the dataset")
    pairs = build_pairs(pair_ids, post_ids, labels, delta_label=delta_id)
    return meta, test, originals, replies, cached, pairs, delta_id


def pair_arrays(pairs, cached, replies) -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {"pair_id": np.array([p.pair_id for p in pairs]),
                                "post_id": np.array([p.post_id for p in pairs])}
    for side, attr in (("d", "delta_row"), ("n", "nodelta_row")):
        rows = [getattr(p, attr) for p in pairs]
        t[f"{side}_row"] = np.array(rows)
        t[f"{side}_score"] = np.array([float(cached[r]["score"]) for r in rows])
        for col in ("reply_words", "reply_tokens", "original_tokens", "n_quote_lines", "truncated"):
            t[f"{side}_{col}"] = np.array([float(cached[r][col]) for r in rows])
        spans = [find_url_spans(replies[r]) for r in rows]
        t[f"{side}_n_urls"] = np.array([len(s) for s in spans], float)
        t[f"{side}_has_url"] = t[f"{side}_n_urls"] > 0
    t["category"] = np.array([link_category(a, b) for a, b in zip(t["d_has_url"], t["n_has_url"])])
    t["model_win"] = win_score(t["d_score"], t["n_score"])
    t["longer_win"] = win_score(t["d_reply_words"], t["n_reply_words"])
    t["word_ratio"] = length_ratio(t["d_reply_words"], t["n_reply_words"])
    return t


def acc_row(name: str, mask: np.ndarray, t: dict, n_boot: int, seed: int) -> dict:
    """Pairs, posts, ModernBERT and longer-reply accuracy, their difference, post-cluster CIs."""
    row = {"group": name, "pairs": int(mask.sum()), "posts": int(len(set(t["post_id"][mask])))}
    if not row["pairs"]:
        return row
    m, b = t["model_win"][mask], t["longer_win"][mask]
    boot = cluster_bootstrap_means(t["post_id"][mask], {"m": m, "b": b, "d": m - b}, n_boot, seed)
    row.update(model_acc=float(m.mean()), model_ci=percentile_ci(boot["m"]),
               longer_acc=float(b.mean()), longer_ci=percentile_ci(boot["b"]),
               diff=float(m.mean() - b.mean()), diff_ci=percentile_ci(boot["d"]))
    return row


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:.1f}"


def ci_str(ci, scale: float = 100, signed: bool = False, digits: int = 1) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"[{f.format(scale * ci[0])}, {f.format(scale * ci[1])}]"


def acc_table(rows: list[dict]) -> str:
    lines = ["| Group | Pairs | Posts | ModernBERT % | 95% CI | Longer reply % | 95% CI | Diff (pts) | 95% CI |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["pairs"]:
            lines.append(f"| {r['group']} | 0 | 0 | – | – | – | – | – | – |")
            continue
        lines.append(f"| {r['group']} | {r['pairs']} | {r['posts']} | {pct(r['model_acc'])} | {ci_str(r['model_ci'])} | "
                     f"{pct(r['longer_acc'])} | {ci_str(r['longer_ci'])} | {100 * r['diff']:+.1f} | {ci_str(r['diff_ci'], signed=True)} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Experiment A: observational
# --------------------------------------------------------------------------------------------

def run_observe(args: argparse.Namespace) -> int:
    from datasets import load_dataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, _, _, replies, cached, pairs, delta_id = load_test_with_cache(Path(args.cache_dir))
    t = pair_arrays(pairs, cached, replies)
    n_boot, seed = args.n_boot, args.seed
    md, metrics = [], {"n_boot": n_boot, "seed": seed, "placeholder": PLACEHOLDER}

    # A1. accuracy by link category
    rows = [acc_row("all pairs", np.ones(len(pairs), bool), t, n_boot, seed)]
    rows += [acc_row(c, t["category"] == c, t, n_boot, seed) for c in CATEGORIES]
    metrics["by_category"] = rows
    md += ["### A1. Pairwise accuracy by link category", "", acc_table(rows), ""]

    # A2. joint with length matching
    matched = t["word_ratio"] >= 0.80
    joint = []
    for c in CATEGORIES:
        for name, m in (("ratio >= 0.80", matched), ("ratio < 0.80", ~matched)):
            joint.append(acc_row(f"{c}, {name}", (t["category"] == c) & m, t, n_boot, seed))
    metrics["by_category_and_length"] = joint
    md += ["### A2. Link category x word-length matching", "", acc_table(joint), ""]

    # A3. possible confounders per category
    conf = []
    for c in CATEGORIES:
        m = t["category"] == c
        if not m.any():
            continue
        quotes = (t["d_n_quote_lines"] + t["n_n_quote_lines"])[m] > 0
        conf.append({
            "group": c, "pairs": int(m.sum()),
            "median_words_delta": float(np.median(t["d_reply_words"][m])),
            "median_words_nodelta": float(np.median(t["n_reply_words"][m])),
            "share_delta_longer": float((t["d_reply_words"][m] > t["n_reply_words"][m]).mean()),
            "share_ratio_ge_080": float(matched[m].mean()),
            "median_urls_per_url_reply": float(np.median(np.r_[t["d_n_urls"][m & t["d_has_url"]], t["n_n_urls"][m & t["n_has_url"]]]))
            if (m & (t["d_has_url"] | t["n_has_url"])).any() else float("nan"),
            "share_pairs_with_quote": float(quotes.mean()),
            "median_original_tokens": float(np.median(t["d_original_tokens"][m])),
            "mean_score_delta": float(t["d_score"][m].mean()), "mean_score_nodelta": float(t["n_score"][m].mean()),
        })
    metrics["confounders"] = conf
    md += ["### A3. Observable characteristics by link category", "",
           "| Group | Pairs | Median words (delta / no-delta) | Delta reply longer % | Word ratio >= 0.80 % | Median URLs per URL reply | Pairs with a quote % | Median post tokens | Mean score (delta / no-delta) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in conf:
        urls = r["median_urls_per_url_reply"]
        urls_str = "–" if np.isnan(urls) else f"{urls:.0f}"
        md.append(f"| {r['group']} | {r['pairs']} | {r['median_words_delta']:.0f} / {r['median_words_nodelta']:.0f} | "
                  f"{pct(r['share_delta_longer'])} | {pct(r['share_ratio_ge_080'])} | {urls_str} | "
                  f"{pct(r['share_pairs_with_quote'])} | {r['median_original_tokens']:.0f} | "
                  f"{r['mean_score_delta']:+.2f} / {r['mean_score_nodelta']:+.2f} |")
    md.append("")

    # A4. URL presence vs label in each split (what the model could learn from training data)
    split_rates = {}
    revision = meta["dataset"]["revision"]
    for split in ("train", "validation", "test"):
        ds = load_dataset(DATASET_REPO, DATASET_CONFIG, revision=revision, split=split)
        has = np.array([bool(find_url_spans(r)) for r in ds["reply"]])
        lab = np.array(list(ds["label"]))
        split_rates[split] = {"replies": int(len(lab)),
                              "url_rate_delta": float(has[lab == delta_id].mean()),
                              "url_rate_nodelta": float(has[lab != delta_id].mean())}
    metrics["url_rate_by_split"] = split_rates
    md += ["### A4. Share of replies with a URL, by label and split", "",
           "| Split | Replies | Delta replies with URL % | No-delta replies with URL % |", "|---|---|---|---|"]
    md += [f"| {s} | {v['replies']} | {pct(v['url_rate_delta'])} | {pct(v['url_rate_nodelta'])} |" for s, v in split_rates.items()]
    md.append("")

    # A5. reply-level score: URL vs no-URL replies, within each label (post-cluster CI)
    clusters = np.r_[t["post_id"], t["post_id"]]
    scores = np.r_[t["d_score"], t["n_score"]]
    has_url = np.r_[t["d_has_url"], t["n_has_url"]]
    is_delta = np.r_[np.ones(len(pairs), bool), np.zeros(len(pairs), bool)]
    a5 = {}
    md += ["### A5. Mean reply score (logit delta − logit no-delta), URL vs no-URL replies", "",
           "| Label | Replies with URL | Mean score with URL | Replies without | Mean score without | Difference | 95% CI |",
           "|---|---|---|---|---|---|---|"]
    for name, lab_mask in (("delta", is_delta), ("no-delta", ~is_delta)):
        with_url = group_mean_replicates(clusters, scores, lab_mask & has_url, n_boot, seed)
        without = group_mean_replicates(clusters, scores, lab_mask & ~has_url, n_boot, seed)
        diff = float(scores[lab_mask & has_url].mean() - scores[lab_mask & ~has_url].mean())
        a5[name] = {"n_url": int((lab_mask & has_url).sum()), "mean_url": float(scores[lab_mask & has_url].mean()),
                    "n_no_url": int((lab_mask & ~has_url).sum()), "mean_no_url": float(scores[lab_mask & ~has_url].mean()),
                    "diff": diff, "diff_ci": nan_ci(with_url - without)}
        r = a5[name]
        md.append(f"| {name} | {r['n_url']} | {r['mean_url']:+.3f} | {r['n_no_url']} | {r['mean_no_url']:+.3f} | "
                  f"{diff:+.3f} | {ci_str(r['diff_ci'], scale=1, signed=True, digits=3)} |")
    metrics["score_by_url_presence"] = a5
    md.append("")

    with open(out_dir / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "post_id", "category", "delta_urls", "nodelta_urls", "delta_score", "nodelta_score",
                    "model_win", "delta_words", "nodelta_words", "word_ratio"])
        for i in range(len(pairs)):
            w.writerow([int(t["pair_id"][i]), t["post_id"][i], t["category"][i], int(t["d_n_urls"][i]), int(t["n_n_urls"][i]),
                        f"{t['d_score'][i]:.6f}", f"{t['n_score'][i]:.6f}", t["model_win"][i],
                        int(t["d_reply_words"][i]), int(t["n_reply_words"][i]), f"{t['word_ratio'][i]:.4f}"])
    (out_dir / "observe_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "observe_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {out_dir}/observe_metrics.json, observe_summary.md, pairs.csv")
    return 0


# --------------------------------------------------------------------------------------------
# Experiment B: controlled URL edits
# --------------------------------------------------------------------------------------------

def run_intervene(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, _, originals, replies, cached, pairs, delta_id = load_test_with_cache(Path(args.cache_dir))
    t = pair_arrays(pairs, cached, replies)
    url_rows = sorted(r for r in range(len(replies)) if find_url_spans(replies[r]))
    print(f"[DATA] {len(url_rows)} of {len(replies)} replies contain a URL")

    # Build and verify the edited replies before touching the model
    variants: dict[str, list[str]] = {v: list(replies) for v in VARIANTS}
    for r in url_rows:
        spans = find_url_spans(replies[r])
        for v in VARIANTS:
            edited = edit_urls(replies[r], v)
            if edited == replies[r] or find_url_spans(edited) or not changes_within_spans(replies[r], edited, spans):
                raise SystemExit(f"[STOP] edit check failed for row {r}, variant {v}")
            variants[v][r] = edited
    print(f"[CHECK] {len(url_rows)} x {len(VARIANTS)} edits: each differs from the original only inside URL spans "
          f"and contains no detected URL")

    rev = meta["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, revision=rev, dtype=torch.float32).eval()
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    label_ids = (label2id["nodelta"], label2id["delta"])
    device = pick_device(args.device)
    model.to(device)
    print(f"[DEVICE] {device}")

    t0 = time.time()
    rescored = score_rows(model, tok, originals, replies, url_rows, device, label_ids)
    repro_diff = max(abs((d - n) - float(cached[r]["score"])) for r, (n, d, _) in rescored.items())
    print(f"[CHECK] unedited URL replies rescored: max |score - cached| = {repro_diff:.2e}")
    if repro_diff > REPRO_TOLERANCE:
        raise SystemExit(f"[STOP] original predictions not reproduced (tolerance {REPRO_TOLERANCE})")
    edited_scores = {v: score_rows(model, tok, originals, variants[v], url_rows, device, label_ids) for v in VARIANTS}
    elapsed = time.time() - t0

    def n_tokens(text_list: list[str], rows: list[int]) -> dict[int, int]:
        ids = tok([text_list[r] for r in rows], add_special_tokens=False)["input_ids"]
        return {r: len(x) for r, x in zip(rows, ids)}
    tokens = {"original": n_tokens(replies, url_rows), **{v: n_tokens(variants[v], url_rows) for v in VARIANTS}}

    # score lookup per variant (unchanged rows reuse the cached score)
    base = np.array([float(cached[r]["score"]) for r in range(len(replies))])
    score = {"original": base}
    for v in VARIANTS:
        s = base.copy()
        for r, (n, d, _) in edited_scores[v].items():
            s[r] = d - n
        score[v] = s

    n_boot, seed = args.n_boot, args.seed
    md: list[str] = []
    metrics: dict = {"url_replies": len(url_rows), "repro_max_abs_diff": repro_diff, "placeholder": PLACEHOLDER,
                     "device": str(device), "seconds": round(elapsed, 1), "n_boot": n_boot, "seed": seed}

    # B1. reply level: score change and own-class flips, by label
    rows = np.array(url_rows)
    post_of_row = {}
    for p in pairs:
        post_of_row[p.delta_row] = post_of_row[p.nodelta_row] = p.post_id
    clusters = np.array([post_of_row[r] for r in rows])
    is_delta = np.array([int(cached[r]["label"]) == delta_id for r in rows])
    md += ["### B1. Reply-level change in score (logit delta − logit no-delta) after editing URLs", "",
           "| Variant | Replies | Label | Mean tokens removed | Mean score change | 95% CI | Replies whose predicted class flips |",
           "|---|---|---|---|---|---|---|"]
    b1 = {}
    for v in VARIANTS:
        change = score[v][rows] - base[rows]
        removed = np.array([tokens["original"][r] - tokens[v][r] for r in rows], float)
        flips = np.sign(score[v][rows]) != np.sign(base[rows])
        res = {}
        for name, m in (("all", np.ones(len(rows), bool)), ("delta", is_delta), ("no-delta", ~is_delta)):
            reps = group_mean_replicates(clusters, change, m, n_boot, seed)
            res[name] = {"replies": int(m.sum()), "mean_tokens_removed": float(removed[m].mean()),
                         "mean_change": float(change[m].mean()), "ci": nan_ci(reps),
                         "class_flips": int(flips[m].sum())}
            md.append(f"| {v} | {res[name]['replies']} | {name} | {res[name]['mean_tokens_removed']:.1f} | "
                      f"{res[name]['mean_change']:+.3f} | {ci_str(res[name]['ci'], scale=1, signed=True, digits=3)} | "
                      f"{res[name]['class_flips']} ({pct(flips[m].mean())}%) |")
        side_diff = (group_mean_replicates(clusters, change, is_delta, n_boot, seed)
                     - group_mean_replicates(clusters, change, ~is_delta, n_boot, seed))
        res["delta_minus_nodelta"] = {"diff": float(change[is_delta].mean() - change[~is_delta].mean()), "ci": nan_ci(side_diff)}
        res["spearman_change_vs_tokens_removed"] = spearman(change, removed)
        res["share_score_decreased"] = float((change < 0).mean())
        b1[v] = res
    metrics["reply_level"] = b1
    md.append("")
    for v in VARIANTS:
        r = b1[v]
        md.append(f"- {v}: delta − no-delta difference in mean score change {r['delta_minus_nodelta']['diff']:+.3f} "
                  f"{ci_str(r['delta_minus_nodelta']['ci'], scale=1, signed=True, digits=3)}; score fell for "
                  f"{pct(r['share_score_decreased'])}% of URL replies; Spearman(change, tokens removed) "
                  f"{r['spearman_change_vs_tokens_removed']:+.3f}.")
    md.append("")

    # B2. pair level: accuracy change and preference flips, per scenario
    d_row, n_row = t["d_row"], t["n_row"]
    win_before = win_score(base[d_row], base[n_row])
    gap_before = base[d_row] - base[n_row]
    b2, flip_records = {}, []
    md += ["### B2. Pairwise accuracy after editing (paired post-cluster bootstrap)", "",
           "| Variant | Scenario | Pairs affected | Accuracy before % | After % | Change (pts) | 95% CI | Preference flips | Correct→wrong | Wrong→correct |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for scen in SCENARIOS:
            d_s = score[v][d_row] if scen != "edit no-delta reply only" else base[d_row]
            n_s = score[v][n_row] if scen != "edit delta reply only" else base[n_row]
            affected = {"edit all URL replies": t["d_has_url"] | t["n_has_url"],
                        "edit delta reply only": t["d_has_url"],
                        "edit no-delta reply only": t["n_has_url"]}[scen]
            win_after = win_score(d_s, n_s)
            gap_after = d_s - n_s
            flip = np.sign(gap_after) != np.sign(gap_before)
            res = {}
            for subset, m in (("all pairs", np.ones(len(pairs), bool)), ("affected pairs", affected)):
                boot = cluster_bootstrap_means(t["post_id"][m], {"chg": win_after[m] - win_before[m]}, n_boot, seed)
                res[subset] = {"pairs": int(m.sum()), "acc_before": float(win_before[m].mean()),
                               "acc_after": float(win_after[m].mean()),
                               "change": float((win_after[m] - win_before[m]).mean()), "change_ci": percentile_ci(boot["chg"]),
                               "flips": int(flip[m].sum()), "correct_to_wrong": int((flip & (win_before == 1))[m].sum()),
                               "wrong_to_correct": int((flip & (win_before == 0))[m].sum())}
                r = res[subset]
                md.append(f"| {v} | {scen} ({subset}) | {r['pairs']} | {pct(r['acc_before'])} | {pct(r['acc_after'])} | "
                          f"{100 * r['change']:+.1f} | {ci_str(r['change_ci'], signed=True)} | {r['flips']} | "
                          f"{r['correct_to_wrong']} | {r['wrong_to_correct']} |")
            b2.setdefault(v, {})[scen] = res
            if scen == "edit all URL replies":
                cat_rows = {}
                for c in CATEGORIES[1:]:
                    m = t["category"] == c
                    boot = cluster_bootstrap_means(t["post_id"][m], {"chg": win_after[m] - win_before[m]}, n_boot, seed)
                    cat_rows[c] = {"pairs": int(m.sum()), "acc_before": float(win_before[m].mean()),
                                   "acc_after": float(win_after[m].mean()), "change_ci": percentile_ci(boot["chg"]),
                                   "flips": int(flip[m].sum()),
                                   "correct_to_wrong": int((flip & (win_before == 1))[m].sum()),
                                   "wrong_to_correct": int((flip & (win_before == 0))[m].sum())}
                b2[v]["by_category"] = cat_rows
                for i in np.where(flip)[0]:
                    flip_records.append({"variant": v, "pair_id": int(t["pair_id"][i]), "post_id": t["post_id"][i],
                                         "category": t["category"][i], "gap_before": float(gap_before[i]),
                                         "gap_after": float(gap_after[i]), "correct_before": float(win_before[i]),
                                         "correct_after": float(win_after[i]),
                                         "delta_tokens_removed": tokens["original"].get(int(d_row[i]), 0) - tokens[v].get(int(d_row[i]), 0),
                                         "nodelta_tokens_removed": tokens["original"].get(int(n_row[i]), 0) - tokens[v].get(int(n_row[i]), 0)})
    metrics["pair_level"] = b2
    md += ["", "**Edit all URL replies, by link category**", "",
           "| Variant | Category | Pairs | Accuracy before % | After % | Change 95% CI | Flips | Correct→wrong | Wrong→correct |",
           "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for c, r in b2[v]["by_category"].items():
            md.append(f"| {v} | {c} | {r['pairs']} | {pct(r['acc_before'])} | {pct(r['acc_after'])} | "
                      f"{ci_str(r['change_ci'], signed=True)} | {r['flips']} | {r['correct_to_wrong']} | {r['wrong_to_correct']} |")
    md.append("")

    with open(out_dir / "url_replies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "pair_id", "post_id", "label", "n_urls", "tokens_original", "tokens_placeholder", "tokens_remove",
                    "score_cached", "score_rescored", "score_placeholder", "score_remove"])
        for r in url_rows:
            n, d, _ = rescored[r]
            w.writerow([r, cached[r]["pair_id"], post_of_row[r], cached[r]["label"], len(find_url_spans(replies[r])),
                        tokens["original"][r], tokens["placeholder"][r], tokens["remove"][r],
                        f"{base[r]:.6f}", f"{d - n:.6f}", f"{score['placeholder'][r]:.6f}", f"{score['remove'][r]:.6f}"])
    with open(out_dir / "flips.csv", "w", newline="") as fh:
        fields = list(flip_records[0]) if flip_records else ["variant", "pair_id"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flip_records)

    if args.export_flip_text and flip_records:
        target = out_dir / "flips_text.jsonl"
        if not is_git_ignored(target):
            print(f"[STOP] {target} is not git-ignored; refusing to write private text there")
            return 2
        by_id = {p.pair_id: p for p in pairs}
        with open(target, "w") as fh:
            for rec in flip_records:
                p = by_id[rec["pair_id"]]
                fh.write(json.dumps({**rec, "original": originals[p.delta_row],
                                     "delta": replies[p.delta_row], "nodelta": replies[p.nodelta_row],
                                     f"delta_{rec['variant']}": variants[rec["variant"]][p.delta_row],
                                     f"nodelta_{rec['variant']}": variants[rec["variant"]][p.nodelta_row]}) + "\n")
        print(f"[TEXT] wrote {len(flip_records)} flipped pairs with text to {target} (git-ignored; keep local)")

    (out_dir / "intervene_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "intervene_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {out_dir}/intervene_metrics.json, intervene_summary.md, url_replies.csv, flips.csv")
    return 0


# --------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("score", help="score the test split with the published model and cache the scores")
    s.add_argument("--cache-dir", default="outputs/cmv_length", help="git-ignored directory the scores are written to (default: %(default)s)")
    s.add_argument("--dataset-revision", default=DATASET_REVISION, help="dataset commit (default: pinned)")
    s.add_argument("--model-revision", default=MODEL_REVISION, help="model commit (default: pinned)")
    s.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="auto = MPS if available, else CPU")
    s.add_argument("--cpu-check-rows", type=int, default=40, help="rows re-scored on CPU to compare devices (0 = skip)")
    s.add_argument("--limit-pairs", type=int, default=None, help="smoke run on the first N pairs (no reproduction check)")
    for name, help_text in (("observe", "Experiment A: observational link analysis (cached scores)"),
                            ("intervene", "Experiment B: score URL-edited replies")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out-dir", default="outputs/cmv_links", help="git-ignored output directory (default: %(default)s)")
        p.add_argument("--cache-dir", default="outputs/cmv_length", help="cached scores from the score stage, read only (default: %(default)s)")
        p.add_argument("--n-boot", type=int, default=10000, help="post-cluster bootstrap resamples (default: %(default)s)")
        p.add_argument("--seed", type=int, default=577, help="bootstrap seed (default: %(default)s)")
        if name == "intervene":
            p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="auto = MPS if available, else CPU")
            p.add_argument("--export-flip-text", action="store_true", help="write flipped pairs' text (git-ignored dir only)")
    args = ap.parse_args(argv)
    if args.stage != "score" and Path(args.out_dir).resolve() == Path(args.cache_dir).resolve():
        ap.error("--out-dir must differ from --cache-dir (the cached scores are read only)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return {"score": run_score, "observe": run_observe, "intervene": run_intervene}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
