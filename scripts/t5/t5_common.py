"""Helpers shared by the T5 scripts (baselines.py, train.py): data loading with the date filter, feature standardization, metrics and an OP-level bootstrap.

Both scripts read the same files written by build_dataset.py through load_rows(), so the logistic-regression baselines and
ModernBERT always see the same splits and the same date filter.
"""
import datetime
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

Row = Dict[str, Any]  # e.g. {"post_id": "3d3hj8", "author": "name", "created_utc": 1436766200, "label": 0, "text": "...", "features": [0.1, ...]}
Ci = Tuple[float, float]  # (lower, upper) of a 95% interval


def post_datetime(created_utc: int) -> datetime.datetime:
    """UTC datetime of a Unix timestamp."""
    return datetime.datetime.fromtimestamp(created_utc, datetime.timezone.utc)


# Text that does not belong to the original post: the moderator footnote that is appended to the thread, paragraphs the OP added later
# ("Edit: ...", "Update: ...", "EDIT 2 -", "**Edit**" ...) and delta symbols. Only paragraphs that START with an edit/update marker are
# removed; the word elsewhere in a paragraph is left alone.
_FOOTNOTE = re.compile(r"(?is)\n\s*(?:[_\-*]{3,}\s*\n\s*)?(?:&gt;|>)?\s*\*?\s*(?:hello, users of cmv! )?this is a footnote from (?:the cmv moderators|your moderators).*$")
_EDIT_PARAGRAPH = re.compile(r"(?im)^[\s>*_#]*(?:edit|update)s?\s*(?:#?\d+|[ivx]{1,4}\b|\([^)\n]{0,40}\)|\[[^\]\n]{0,30}\])?(?:\s+(?:for|to)\s+\w+)?\s*(?:[*_]+|[:\-\u2013\u2014~/]|$)(?s:.*?)(?:\n\s*\n|\Z)")
_DELTA_SYMBOL = re.compile(r"[\u2206\u0394]|&amp;#8710;|!delta", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Post text as it was before the discussion: without the moderator footnote, the OP's later edit/update paragraphs and delta symbols."""
    text = _FOOTNOTE.sub("", text)
    text = _EDIT_PARAGRAPH.sub("\n\n", text)  # keep a paragraph break where a paragraph was removed
    text = _DELTA_SYMBOL.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_rows(path: Path) -> List[Row]:
    """Read one split written by build_dataset.py; `text` is clean_text() of the stored post, the stored text stays in `text_raw`."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    for r in rows:
        r["text_raw"], r["text"] = r["text"], clean_text(r["text"])
    return rows


def check_label(rows: List[Row], data_dir: Path) -> None:
    """Stop if the dataset does not carry the DeltaBot label (older datasets used the flair-based `delta` flag, which misses many awarded deltas)."""
    if not rows or "label_deltabot" not in rows[0] or any(r["label"] != r["label_deltabot"] for r in rows):
        raise SystemExit(f"{data_dir} does not use the DeltaBot label; rebuild it with scripts/t5/build_dataset.py")


def standardizer(train_features: np.ndarray, names: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
    """z-score function fitted on the training matrix; the heavy-tailed LIWC word count is log-transformed first."""
    def prep(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        if "liwc_wc" in names:
            x[:, list(names).index("liwc_wc")] = np.log1p(x[:, list(names).index("liwc_wc")])
        return x
    base = prep(train_features)
    mean, std = base.mean(0), base.std(0) + 1e-6
    return lambda x: (prep(x) - mean) / std


def metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Ranking and threshold metrics for the delta class (threshold 0.5): AUROC, PR-AUC (chance = delta rate), accuracy, balanced accuracy, F1, macro-F1."""
    pred = (scores >= 0.5).astype(int)
    return {"auroc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores)),
            "accuracy": float(accuracy_score(labels, pred)), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "f1_delta": float(f1_score(labels, pred, zero_division=0)), "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0))}


def bootstrap_by_group(scores: np.ndarray, labels: np.ndarray, groups: Sequence[str], n_resamples: int = 1000, seed: int = 0) -> Dict[str, Ci]:
    """95% percentile intervals of AUROC and PR-AUC when whole OPs (all posts of one author) are resampled, since posts of one OP are not independent."""
    members: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    arrays = [np.array(v) for v in members.values()]
    rng = np.random.default_rng(seed)
    auroc, pr_auc = [], []
    for _ in range(n_resamples):
        idx = np.concatenate([arrays[j] for j in rng.integers(0, len(arrays), len(arrays))])
        if len(set(labels[idx])) == 2:
            auroc.append(roc_auc_score(labels[idx], scores[idx]))
            pr_auc.append(average_precision_score(labels[idx], scores[idx]))
    return {"auroc": (float(np.percentile(auroc, 2.5)), float(np.percentile(auroc, 97.5))),
            "pr_auc": (float(np.percentile(pr_auc, 2.5)), float(np.percentile(pr_auc, 97.5)))}
