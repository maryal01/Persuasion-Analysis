"""T5 (CMV malleability) baselines without a neural network.

Logistic regression on: the majority class, the posting date, the post length, the words of the post (TF-IDF), the author's
LIWC and Receptiviti numbers, and words together with the author numbers. Penalty (L1/L2) and strength C are chosen on the
validation split (AUROC); the test split is scored once per model. Test AUROC and PR-AUC get 95% intervals from a bootstrap
over authors. The data is read from the Hub (scripts/t5/t5_common.py), the same data and splits as train.py (ModernBERT).

Usage (from the repository root; needs numpy, scikit-learn and a Hugging Face login or HF_TOKEN):
    python scripts/t5/baselines.py --output-dir <out_dir>/t5_baselines
Writes baselines.json and prints a table.
"""
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from t5_common import HUB_REVISION, SPLIT_NAMES, T5Data, bootstrap_by_author, fit_scaler, load_t5_data, metrics

Matrices = Tuple[Any, Any, Any]  # the design matrices of one model for (train, validation, test)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
PENALTIES = ("l1", "l2")
AUTHOR_FEATURES = {  # name of the model -> prefixes of the feature columns it uses
    "liwc_only": ("liwc_",),
    "receptiviti_only": ("recv_",),
    "liwc_receptiviti_all": ("liwc_", "recv_"),
}


# --- design matrices ----------------------------------------------------------------------------------------------------

def standardize(matrices: Sequence[np.ndarray], column_names: List[str]) -> Matrices:
    """Standardize the (train, validation, test) matrices with the statistics of the training matrix."""
    scale = fit_scaler(matrices[0], column_names)
    return tuple(scale(matrix) for matrix in matrices)


def time_matrices(data: T5Data) -> Matrices:
    """The posting time as one standardized column (to see how much of the label is just time)."""
    return standardize([split.years().reshape(-1, 1) for split in data.splits], ["time"])


def length_matrices(data: T5Data) -> Matrices:
    """The logarithm of the post length in characters as one standardized column."""
    return standardize([np.log1p([[len(text)] for text in split.texts]) for split in data.splits], ["length"])


def word_matrices(data: T5Data) -> Matrices:
    """TF-IDF of the words of the post; the vocabulary is learned on the training posts only."""
    vectorizer = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True).fit(data.train.texts)
    return tuple(vectorizer.transform(split.texts) for split in data.splits)


def author_matrices(data: T5Data, prefixes: Tuple[str, ...]) -> Matrices:
    """The standardized author features whose names start with one of the prefixes."""
    columns = [i for i, name in enumerate(data.feature_names) if name.startswith(prefixes)]
    names = [data.feature_names[i] for i in columns]
    return standardize([split.features[:, columns] for split in data.splits], names)


def words_with_author(words: Matrices, author: Matrices) -> Matrices:
    """The word matrix with the author columns next to it, for each split."""
    return tuple(hstack([w, csr_matrix(a)]).tocsr() for w, a in zip(words, author))


# --- fitting ------------------------------------------------------------------------------------------------------------

def best_by_validation(matrices: Matrices, data: T5Data) -> Tuple[float, str, float, LogisticRegression]:
    """Fit every (penalty, C) pair on the training posts; return (validation AUROC, penalty, C, model) of the best one."""
    x_train, x_validation, _ = matrices
    fitted = []
    for penalty, c in product(PENALTIES, C_GRID):
        model = LogisticRegression(penalty=penalty, C=c, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0)
        model.fit(x_train, data.train.labels)
        validation_auroc = roc_auc_score(data.validation.labels, model.predict_proba(x_validation)[:, 1])
        fitted.append((float(validation_auroc), penalty, c, model))
    return max(fitted, key=lambda candidate: candidate[0])


def fit_logistic(matrices: Matrices, data: T5Data) -> Dict[str, Any]:
    """Choose the settings on the validation split, then report validation and test metrics and the test intervals."""
    validation_auroc, penalty, c, model = best_by_validation(matrices, data)
    validation_scores = model.predict_proba(matrices[1])[:, 1]
    test_scores = model.predict_proba(matrices[2])[:, 1]
    return {"val_auroc": validation_auroc, "penalty": penalty, "C": c,
            "val": metrics(validation_scores, data.validation.labels),
            "test": metrics(test_scores, data.test.labels),
            "test_ci95": bootstrap_by_author(test_scores, data.test.labels, data.test.authors)}


# --- report -------------------------------------------------------------------------------------------------------------

def describe(data: T5Data, revision: str) -> Dict[str, Any]:
    """The data the baselines ran on."""
    splits = dict(zip(SPLIT_NAMES, data.splits))
    return {"hub": {"revision": revision, "commit": data.hub_commit},
            "n": {name: len(split) for name, split in splits.items()},
            "n_ops": {name: len(set(split.authors)) for name, split in splits.items()},
            "delta_rate": {name: float(split.labels.mean()) for name, split in splits.items()}}


def table_row(name: str, result: Dict[str, Any]) -> str:
    """One line of the printed table."""
    auroc_low, auroc_high = result["test_ci95"]["auroc"]
    pr_low, pr_high = result["test_ci95"]["pr_auc"]
    test = result["test"]
    return (f"{name:27s} {result['val']['auroc']:8.3f} {test['auroc']:9.3f} {f'[{auroc_low:.3f},{auroc_high:.3f}]':>19s} {test['pr_auc']:7.3f} "
            f"{f'[{pr_low:.3f},{pr_high:.3f}]':>15s} {test['balanced_accuracy']:7.3f} {test['macro_f1']:7.3f}  {result['penalty']} C={result['C']}")


def print_results(results: Dict[str, Any]) -> None:
    print(f"Hub commit {results['hub']['commit'][:8]}; train/val/test {results['n']} (OPs {results['n_ops']}); "
          f"delta rate { {k: round(v, 3) for k, v in results['delta_rate'].items()} }")
    print(f"{'model':27s} {'valAUROC':>8s} {'testAUROC':>9s} {'AUROC 95% CI (OP)':>19s} {'PR-AUC':>7s} {'PR 95% CI':>15s} {'bal.acc':>7s} {'macroF1':>7s}  chosen")
    majority = results["majority_class"]["test"]
    print(f"{'majority class':27s} {'-':>8s} {majority['auroc']:9.3f} {'-':>19s} {majority['pr_auc']:7.3f} {'-':>15s} "
          f"{majority['balanced_accuracy']:7.3f} {majority['macro_f1']:7.3f}")
    for name in ("post_time", "post_length", "bow_tfidf", "liwc_only", "receptiviti_only", "liwc_receptiviti_all", "bow_plus_liwc_receptiviti"):
        print(table_row(name, results[name]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hub-revision", default=HUB_REVISION, help="branch or commit of the Hub dataset (default: %(default)s)")
    args = parser.parse_args()

    data = load_t5_data(args.hub_revision)
    results: Dict[str, Any] = describe(data, args.hub_revision)

    # the training share of delta posts as a constant score: AUROC 0.5, PR-AUC = share of delta posts
    results["majority_class"] = {"test": metrics(np.full(len(data.test), data.train.labels.mean()), data.test.labels)}
    results["post_time"] = fit_logistic(time_matrices(data), data)
    results["post_length"] = fit_logistic(length_matrices(data), data)
    words = word_matrices(data)
    results["bow_tfidf"] = fit_logistic(words, data)
    author = {name: author_matrices(data, prefixes) for name, prefixes in AUTHOR_FEATURES.items()}
    for name, matrices in author.items():
        results[name] = {"n_features": matrices[0].shape[1], **fit_logistic(matrices, data)}
    results["bow_plus_liwc_receptiviti"] = fit_logistic(words_with_author(words, author["liwc_receptiviti_all"]), data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baselines.json").write_text(json.dumps(results, indent=1))
    print_results(results)


if __name__ == "__main__":
    main()
