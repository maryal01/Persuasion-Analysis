"""T5 (CMV malleability) baselines without a neural network: majority class, posting year, post length, bag of words, author LIWC/Receptiviti features.

Hyperparameters are chosen on the validation split only (AUROC); the test split is scored once per model.
Test AUROC and PR-AUC get 95% intervals from a bootstrap over OPs (1,000 resamples). Data loading, the date filter, the
metrics and the bootstrap come from t5_common.py, shared with train.py (ModernBERT), so both see the same posts.

Usage (from the repository root; needs numpy and scikit-learn):
    python scripts/t5/baselines.py \
        --data-dir <out_dir>/t5_dataset \
        --output-dir <out_dir>/t5_baselines
`--data-dir` is the output of scripts/t5/build_dataset.py (DeltaBot label; the script stops on a dataset with another label).
All posts are used. The post text is cleaned by t5_common.clean_text() (moderator footnote, the OP's later Edit:/Update: paragraphs and delta symbols removed).
The "post_time" row (posting time as the only feature) shows how much of the label is time alone.
Writes baselines.json and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from t5_common import bootstrap_by_group, check_label, load_rows, metrics, post_datetime, standardizer

Rows = List[Dict[str, Any]]  # rows of one split, see t5_common.Row
Matrices = Tuple[Any, Any, Any]  # (train, validation, test) design matrices of one model
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
PENALTIES = ("l1", "l2")


def feature_matrix(rows: Rows) -> np.ndarray:
    """(n, n_features) matrix of the author features of the rows."""
    return np.array([r["features"] for r in rows], dtype=float)


def scaled(parts: List[np.ndarray], names: List[str]) -> Tuple[np.ndarray, ...]:
    """Standardize [train, validation, test] matrices with the train statistics."""
    transform = standardizer(parts[0], names)
    return tuple(transform(p) for p in parts)


def fit_logistic(matrices: Matrices, labels: Tuple[np.ndarray, ...], groups_test: List[str]) -> Dict[str, Any]:
    """Grid search penalty and C on validation AUROC (class-balanced weights), then score validation and test."""
    (x_train, x_val, x_test), (y_train, y_val, y_test) = matrices, labels
    best: Dict[str, Any] = {"val_auroc": -1.0}
    for penalty in PENALTIES:
        for c in C_GRID:
            model = LogisticRegression(penalty=penalty, C=c, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0)
            model.fit(x_train, y_train)
            val_auroc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
            if val_auroc > best["val_auroc"]:
                best = {"val_auroc": float(val_auroc), "penalty": penalty, "C": c, "model": model}
    model = best.pop("model")
    test_scores = model.predict_proba(x_test)[:, 1]
    return {**best, "val": metrics(model.predict_proba(x_val)[:, 1], y_val), "test": metrics(test_scores, y_test),
            "test_ci95": bootstrap_by_group(test_scores, y_test, groups_test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train, val, test = (load_rows(args.data_dir / f"{s}.jsonl") for s in ("train", "validation", "test"))
    names = json.loads((args.data_dir / "feature_names.json").read_text()) # feature names in LIWC/Receptiviti
    labels = tuple(np.array([r["label"] for r in rows]) for rows in (train, val, test))
    groups_test = [r["author"] for r in test] # OP for each test sample
    parts = (train, val, test)

    check_label(train, args.data_dir)
    results: Dict[str, Any] = {"data_dir": str(args.data_dir), "label": "deltabot", "n": {"train": len(train), "validation": len(val), "test": len(test)},
                               "n_ops": {"train": len({r["author"] for r in train}), "validation": len({r["author"] for r in val}), "test": len({r["author"] for r in test})},
                               "delta_rate": {"train": float(labels[0].mean()), "validation": float(labels[1].mean()), "test": float(labels[2].mean())}}

    # majority class: the training prior as a constant score (AUROC 0.5, PR-AUC = delta rate)
    results["majority_class"] = {"test": metrics(np.full(len(test), labels[0].mean()), labels[2])}

    # posting year as the only feature (time confound of the label)
    years = [np.array([[post_datetime(r["created_utc"]).year + post_datetime(r["created_utc"]).timetuple().tm_yday / 366] for r in rows]) for rows in parts]
    results["post_time"] = fit_logistic(scaled(years, ["t"]), labels, groups_test)

    # post length in characters (log), one feature
    lengths = [np.log1p(np.array([[len(r["text"])] for r in rows], dtype=float)) for rows in parts]
    results["post_length"] = fit_logistic(scaled(lengths, ["len"]), labels, groups_test)

    # bag of words (tf-idf) of the post text, fitted on train only
    tfidf = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True).fit([r["text"] for r in train])
    bow = tuple(tfidf.transform([r["text"] for r in rows]) for rows in parts)
    results["bow_tfidf"] = fit_logistic(bow, labels, groups_test)

    # author features: all, LIWC only, Receptiviti only
    groups = {"liwc_receptiviti_all": ("liwc_", "recv_"), "liwc_only": ("liwc_",), "receptiviti_only": ("recv_",)}
    dense: Dict[str, Matrices] = {}
    for group, prefix in groups.items():
        columns = [i for i, n in enumerate(names) if n.startswith(prefix)]
        dense[group] = scaled([feature_matrix(rows)[:, columns] for rows in parts], [names[i] for i in columns])
        results[group] = {"n_features": len(columns), **fit_logistic(dense[group], labels, groups_test)}

    # post text and all author features together
    combined = tuple(hstack([bow[i], csr_matrix(dense["liwc_receptiviti_all"][i])]).tocsr() for i in range(3))
    results["bow_plus_liwc_receptiviti"] = fit_logistic(combined, labels, groups_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baselines.json").write_text(json.dumps(results, indent=1))
    print(f"data {args.data_dir.name} (DeltaBot label); train/val/test {results['n']} (OPs {results['n_ops']}); delta rate { {k: round(v, 3) for k, v in results['delta_rate'].items()} }")
    print(f"{'model':27s} {'valAUROC':>8s} {'testAUROC':>9s} {'AUROC 95% CI (OP)':>19s} {'PR-AUC':>7s} {'PR 95% CI':>15s} {'bal.acc':>7s} {'macroF1':>7s}  chosen")
    m = results["majority_class"]["test"]
    print(f"{'majority class':27s} {'-':>8s} {m['auroc']:9.3f} {'-':>19s} {m['pr_auc']:7.3f} {'-':>15s} {m['balanced_accuracy']:7.3f} {m['macro_f1']:7.3f}")
    for name in ("post_time", "post_length", "bow_tfidf", "liwc_only", "receptiviti_only", "liwc_receptiviti_all", "bow_plus_liwc_receptiviti"):
        r = results[name]
        a, p = r["test_ci95"]["auroc"], r["test_ci95"]["pr_auc"]
        print(f"{name:27s} {r['val']['auroc']:8.3f} {r['test']['auroc']:9.3f} {f'[{a[0]:.3f},{a[1]:.3f}]':>19s} {r['test']['pr_auc']:7.3f} {f'[{p[0]:.3f},{p[1]:.3f}]':>15s} "
              f"{r['test']['balanced_accuracy']:7.3f} {r['test']['macro_f1']:7.3f}  {r['penalty']} C={r['C']}")


if __name__ == "__main__":
    main()
