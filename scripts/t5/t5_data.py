"""The T5 dataset (does the OP of a CMV post award a delta?) as arrays per split, read from the Hugging Face Hub.

The Hub config holds one row per post: the raw title + body, the DeltaBot label, the author, the time of the post and the
211 LIWC/Receptiviti values of the author. Reading it from the Hub (and recording the commit) keeps every run on the same data.
"""
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from t5_cleaning import clean_text

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
