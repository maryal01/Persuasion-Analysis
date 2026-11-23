"""T5 with the OP's history: logistic regression baselines on the Hub configs with a history (built by Manish, see
analysis/2026-11-23_t5_thread_history).

In these configs `original` is "<history items> [SEP] <post>". Only `original` is used: `reply` is the discussion under the
post, which is not part of T5 (post + history). Inputs compared: posting date, number of history items, TF-IDF of the post,
of the history, of both. The post and the history are cleaned with clean_text (footnote, edit paragraphs, delta symbols).
Penalty and C are chosen on validation AUROC like baselines.py; test AUROC/PR-AUC get 95% intervals from a bootstrap over OPs.

Usage (from the repository root; one core, a few minutes per config; needs a Hugging Face login):
    python scripts/t5/history_baselines.py --config cmv_thread_own_history --output-dir <out_dir>/history_baselines
"""
import argparse
import datetime
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from t5_common import HUB_REPO, SPLIT_NAMES, bootstrap_by_author, clean_text, metrics

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
PENALTIES = ("l1", "l2")
SEPARATOR = " [SEP] "


@dataclass
class HistorySplit:
    authors: List[str]
    years: np.ndarray       # year of the post with the fraction of the year
    n_history: np.ndarray   # number of history items
    posts: List[str]        # cleaned post
    histories: List[str]    # cleaned history ("" without history)
    labels: np.ndarray


def split_original(original: str) -> Tuple[str, str]:
    """(history, post) of an `original` field; the history is "" when the row has none."""
    history, post = original.rsplit(SEPARATOR, 1)
    return ("" if history.strip() == "[no history]" else history), post


def to_split(table) -> HistorySplit:
    parts = [split_original(original) for original in table["original"]]
    dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in table["created_utc"]]
    return HistorySplit(authors=table["author"], years=np.array([d.year + d.timetuple().tm_yday / 366 for d in dates]),
                        n_history=np.array(table["n_history"], dtype=float),
                        posts=[clean_text(post) for _, post in parts], histories=[clean_text(history) for history, _ in parts],
                        labels=np.array(table["label"]))


# --- design matrices: one tuple (train, validation, test) per input -----------------------------------------------------

def dense_column(splits: List[HistorySplit], values) -> Tuple[np.ndarray, ...]:
    """One standardized column computed by values(split), with the training mean and deviation."""
    columns = [np.asarray(values(split), dtype=float).reshape(-1, 1) for split in splits]
    mean, std = columns[0].mean(), columns[0].std() or 1.0
    return tuple((column - mean) / std for column in columns)


def tfidf(splits: List[HistorySplit], field: str) -> Tuple[Any, ...]:
    """TF-IDF of one text field; the vocabulary is learned on the training split."""
    vectorizer = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True).fit(getattr(splits[0], field))
    return tuple(vectorizer.transform(getattr(split, field)) for split in splits)


def side_by_side(*inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return tuple(hstack([csr_matrix(m) for m in matrices]).tocsr() for matrices in zip(*inputs))


# --- fitting ----------------------------------------------------------------------------------------------------------------

def fit_logistic(matrices: Tuple[Any, ...], splits: List[HistorySplit]) -> Dict[str, Any]:
    """Best (penalty, C) on validation AUROC; validation and test metrics of that model."""
    train, validation, test = splits
    fitted = []
    for penalty, c in product(PENALTIES, C_GRID):
        model = LogisticRegression(penalty=penalty, C=c, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0)
        model.fit(matrices[0], train.labels)
        fitted.append((roc_auc_score(validation.labels, model.predict_proba(matrices[1])[:, 1]), penalty, c, model))
    _, penalty, c, model = max(fitted, key=lambda candidate: candidate[0])
    test_scores = model.predict_proba(matrices[2])[:, 1]
    return {"penalty": penalty, "C": c, "val": metrics(model.predict_proba(matrices[1])[:, 1], validation.labels),
            "test": metrics(test_scores, test.labels), "test_ci95": bootstrap_by_author(test_scores, test.labels, test.authors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Hub config with '<history> [SEP] <post>' in `original`")
    parser.add_argument("--hub-revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(HUB_REPO, args.config, revision=args.hub_revision)
    splits = [to_split(dataset[name]) for name in SPLIT_NAMES]
    results: Dict[str, Any] = {
        "config": args.config, "revision": args.hub_revision,
        "n": {name: len(split.labels) for name, split in zip(SPLIT_NAMES, splits)},
        "delta_rate": {name: float(split.labels.mean()) for name, split in zip(SPLIT_NAMES, splits)},
        "share_without_history": {name: float((split.n_history == 0).mean()) for name, split in zip(SPLIT_NAMES, splits)},
    }
    date = dense_column(splits, lambda split: split.years)
    n_history = dense_column(splits, lambda split: np.log1p(split.n_history))
    posts, histories = tfidf(splits, "posts"), tfidf(splits, "histories")
    inputs = {
        "date": date,
        "n_history": n_history,
        "post": posts,
        "history": histories,
        "post_history": side_by_side(posts, histories),
        "post_history_date": side_by_side(posts, histories, date),
    }
    for name, matrices in inputs.items():
        results[name] = fit_logistic(matrices, splits)
        test, (low, high) = results[name]["test"], results[name]["test_ci95"]["auroc"]
        print(f"{args.config:24s} {name:18s} val {results[name]['val']['auroc']:.3f}  test AUROC {test['auroc']:.3f} "
              f"[{low:.3f},{high:.3f}]  PR-AUC {test['pr_auc']:.3f}  bal.acc {test['balanced_accuracy']:.3f}  "
              f"{results[name]['penalty']} C={results[name]['C']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.config}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
