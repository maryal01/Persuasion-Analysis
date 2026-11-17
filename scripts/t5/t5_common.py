"""Code shared by the T5 scripts (build_dataset.py, baselines.py, train.py): the text clean-up, the dataset as arrays per split
(read from the Hugging Face Hub) and the metrics."""
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


# --- text clean-up ----------------------------------------------------------------------------------------------------
# The stored posts contain things that do not belong to the original argument and can give the outcome away: the moderator footnote
# that is appended to every thread, paragraphs the OP added later that START with an edit or update marker ("Edit:", "Update -",
# "EDIT 2:", "**Edit**", ...) and delta symbols. Only paragraphs that start with such a marker are removed; the word "edit"
# elsewhere in a paragraph is left alone.

_FOOTNOTE = re.compile(
    r"(?is)\n\s*(?:[_\-*]{3,}\s*\n\s*)?(?:&gt;|>)?\s*\*?\s*(?:hello, users of cmv! )?"
    r"this is a footnote from (?:the cmv moderators|your moderators).*$")

_EDIT_MARKER = (
    r"^[\s>*_#]*(?:edit|update)s?"                     # "Edit", "UPDATE", "**Edit"
    r"\s*(?:#?\d+|[ivx]{1,4}\b|\([^)\n]{0,40}\)|\[[^\]\n]{0,30}\])?"  # optional number, "II" or "(date)"
    r"(?:\s+(?:for|to)\s+\w+)?"                        # optional "for clarity", "to clarify"
    r"\s*(?:[*_]+|[:\-–—~/]|$)"              # closing "**", ":", "-", ... or the end of the line
)
_EDIT_PARAGRAPH = re.compile(r"(?im)" + _EDIT_MARKER + r"(?s:.*?)(?:\n\s*\n|\Z)")   # the marker line up to the end of its paragraph

_DELTA_SYMBOL = re.compile(r"[∆Δ]|&amp;#8710;|!delta", re.IGNORECASE)


def clean_text(text: str) -> str:
    """The post text without the moderator footnote, the OP's later edit/update paragraphs and delta symbols."""
    text = _FOOTNOTE.sub("", text)
    text = _EDIT_PARAGRAPH.sub("\n\n", text)  # keep a paragraph break where a paragraph was removed
    text = _DELTA_SYMBOL.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- the dataset --------------------------------------------------------------------------------------------------------
# One Hub config row per post: the raw title + body, the DeltaBot label, the author, the time of the post and the 211 LIWC/Receptiviti
# values of the author. Reading it from the Hub (and recording the commit) keeps every run on the same data.

HUB_REPO = "nlp-final-project11/persuasion-datasets"
HUB_CONFIG = "cmv_malleability_liwc"
HUB_REVISION = "main"  # branch of the Hub repo that the scripts read from
SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class Split:
    """One split: index i of every field belongs to the same post."""
    authors: List[str]
    created_utc: np.ndarray  # Unix time of the post
    texts: List[str]         # cleaned title + body of the post
    features: np.ndarray     # (posts, 211) LIWC and Receptiviti values of the author
    labels: np.ndarray       # 1 if DeltaBot confirmed a delta that the author awarded

    def __len__(self) -> int:
        return len(self.labels)

    def first(self, n: int) -> "Split":
        """The first n posts."""
        return Split(self.authors[:n], self.created_utc[:n], self.texts[:n], self.features[:n], self.labels[:n])

    def years(self) -> np.ndarray:
        """Year of the post with the fraction of the year that has passed (2015.5 = mid 2015)."""
        dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in self.created_utc]
        return np.array([d.year + d.timetuple().tm_yday / 366 for d in dates])


@dataclass
class T5Data:
    train: Split
    validation: Split
    test: Split
    feature_names: List[str]  # names of the 211 feature columns ("liwc_...", "recv_raw_...", "recv_pct_...")
    hub_commit: str           # commit of the Hub dataset that was read

    @property
    def splits(self) -> Tuple[Split, Split, Split]:
        return self.train, self.validation, self.test


def to_split(table: Dataset) -> Split:
    """The columns of one Hub split as arrays; the post text is cleaned here."""
    if table["label"] != table["label_deltabot"]:
        raise SystemExit("this dataset does not use the DeltaBot label; rebuild it with scripts/t5/build_dataset.py")
    return Split(authors=table["author"], created_utc=np.array(table["created_utc"]),
                 texts=[clean_text(text) for text in table["text"]],
                 features=np.array(table["features"], dtype=np.float64), labels=np.array(table["label"]))


def load_t5_data(revision: str = HUB_REVISION) -> T5Data:
    """Read the T5 config of the Hub dataset at the given branch (or commit); needs a Hugging Face login or HF_TOKEN."""
    commit = HfApi().dataset_info(HUB_REPO, revision=revision).sha
    tables = load_dataset(HUB_REPO, HUB_CONFIG, revision=commit)
    names_file = hf_hub_download(HUB_REPO, f"{HUB_CONFIG}/feature_names.json", repo_type="dataset", revision=commit)
    feature_names = json.loads(Path(names_file).read_text())
    return T5Data(*(to_split(tables[name]) for name in SPLIT_NAMES), feature_names=feature_names, hub_commit=commit)


def fit_scaler(train_features: np.ndarray, feature_names: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
    """A function that standardizes feature columns with the mean and standard deviation of the training posts.
    The LIWC word count has a very long tail, so its logarithm is taken first."""
    word_count = list(feature_names).index("liwc_wc") if "liwc_wc" in feature_names else None

    def prepare(features: np.ndarray) -> np.ndarray:
        features = np.array(features, dtype=float)
        if word_count is not None:
            features[:, word_count] = np.log1p(features[:, word_count])
        return features

    prepared = prepare(train_features)
    mean, std = prepared.mean(axis=0), prepared.std(axis=0) + 1e-6
    return lambda features: (prepare(features) - mean) / std


# --- metrics ------------------------------------------------------------------------------------------------------------

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
