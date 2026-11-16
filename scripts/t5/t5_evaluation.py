"""Metrics for the T5 models: ranking and threshold metrics for the delta class, and an OP-level bootstrap."""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

Interval = Tuple[float, float]  # lower and upper end of a 95% interval


def metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """AUROC and PR-AUC of the scores (PR-AUC of a model without information = share of delta posts), and accuracy,
    balanced accuracy, F1 and macro-F1 at the threshold 0.5."""
    predictions = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_delta": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def posts_by_author(authors: Sequence[str]) -> List[np.ndarray]:
    """For each author, the indices of that author's posts."""
    indices: Dict[str, List[int]] = {}
    for index, author in enumerate(authors):
        indices.setdefault(author, []).append(index)
    return [np.array(posts) for posts in indices.values()]


def bootstrap_by_author(scores: np.ndarray, labels: np.ndarray, authors: Sequence[str],
                        n_resamples: int = 1000, seed: int = 0) -> Dict[str, Interval]:
    """95% intervals of AUROC and PR-AUC when whole authors (with all their posts) are resampled.
    Posts of one author are not independent, so resampling single posts would give intervals that are too narrow."""
    author_posts = posts_by_author(authors)
    rng = np.random.default_rng(seed)
    auroc: List[float] = []
    pr_auc: List[float] = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(author_posts), len(author_posts))
        posts = np.concatenate([author_posts[i] for i in chosen])
        if len(set(labels[posts])) < 2:
            continue
        auroc.append(roc_auc_score(labels[posts], scores[posts]))
        pr_auc.append(average_precision_score(labels[posts], scores[posts]))
    return {"auroc": interval(auroc), "pr_auc": interval(pr_auc)}


def interval(values: List[float]) -> Interval:
    """The 2.5th and 97.5th percentile of the values."""
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
