"""T3 (audience profiling) baselines: predict a DDO user's self-reported stance from what they wrote.

For every label column of `ddo_users` (the four issue dimensions, religion and the political ideology
group) four models are scored on held-out users:

  majority                always the most common class among the training users
  random                  a class drawn at random with the training class frequencies (mean over many draws)
  tfidf, class-weighted   logistic regression on TF-IDF word features of the user's text, with classes
                          weighted equally; C chosen on validation macro-F1
  tfidf, unweighted       the same without class weights; C chosen on validation accuracy

The two reference rows matter for different metrics: `majority` is the bar for accuracy, `random` is the
bar for macro-F1 (a model with no signal and equal class weights scores like it, not like `majority`).

A user's text is every `ddo_history` item they wrote (debate arguments, comments, opinion arguments,
poll votes), each with its title in front, joined together. Two kinds of rows are left out:

  votes                they have no text
  held-out T2 debates  items from debates in T2's validation or test split, so a model built this way
                       can later write profiles for T2 without having read those debates
                       (`--all-debates` keeps them)

Users, labels, class names and splits come from the Hub dataset. Only users with the label and at least
one text item are scored. The vocabulary and IDF weights come from the training users, the test split is
scored once per model, and both test metrics get a 95% interval from a bootstrap over users. The
accuracy difference to `majority` gets one too, computed on the same resamples.

Usage (from the repository root; needs scikit-learn):
    python scripts/t3/baselines.py --smoke
    python scripts/t3/baselines.py --output outputs/t3_baselines.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from datasets import ClassLabel, load_dataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SPLITS = ("train", "validation", "test")
HELD_OUT = ("validation", "test")
HISTORY_COLUMNS = ["user", "title", "text", "debate_split"]
C_GRID = (0.01, 0.1, 1.0, 10.0)
# TF-IDF models: name -> (class_weight for the logistic regression, validation metric that picks C).
TFIDF_MODELS = {
    "tfidf, class-weighted": ("balanced", "macro_f1"),
    "tfidf, unweighted": (None, "accuracy"),
}
METRIC_NAMES = {"accuracy": "accuracy", "macro_f1": "macro-F1"}
SMOKE_ROWS = 20_000  # history rows read per split with --smoke
SEED = 42

Labels = Dict[str, Dict[str, np.ndarray]]  # split -> label column -> class index per user, -1 when unknown
Interval = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Majority, random and TF-IDF baselines for T3 audience profiling.")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--users-config", default="ddo_users", help="one row per user, with the labels")
    ap.add_argument("--history-config", default="ddo_history", help="one row per thing a user did")
    ap.add_argument("--all-debates", action="store_true",
                    help="also use text from debates in T2's validation and test splits")
    ap.add_argument("--no-titles", action="store_true", help="use only what the user wrote, without item titles")
    ap.add_argument("--min-df", type=int, default=5, help="drop words used by fewer training users (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test users, and random draws")
    ap.add_argument("--smoke", action="store_true", help=f"quick check on the first {SMOKE_ROWS} history rows per split")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def load_users(repo: str, config: str, revision: str) -> Tuple[Dict[str, List[str]], Labels, Dict[str, List[str]]]:
    """User names and labels per split, and the class names of every label column (the ClassLabel columns)."""
    ds = load_dataset(repo, config, revision=revision, streaming=True, token=os.environ.get("HF_TOKEN"))
    features = ds["train"].features
    if features is None:
        raise SystemExit(f"{config} has no feature metadata on the Hub, so its label columns cannot be found")
    class_names = {name: feature.names for name, feature in features.items() if isinstance(feature, ClassLabel)}
    ds = ds.select_columns(["user", *class_names])
    users, labels = {}, {}
    for split in SPLITS:
        rows = list(ds[split])
        users[split] = [row["user"] for row in rows]
        labels[split] = {name: np.array([-1 if row[name] is None else row[name] for row in rows])
                         for name in class_names}
    return users, labels, class_names


def load_texts(repo: str, config: str, revision: str, wanted: Set[str], all_debates: bool, titles: bool,
               max_rows: Optional[int]) -> Tuple[Dict[str, str], Counter]:
    """One document per wanted user: their text items joined by blank lines. Streams the history table."""
    ds = load_dataset(repo, config, revision=revision, columns=HISTORY_COLUMNS, streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    items = defaultdict(list)
    counts = Counter()
    for split in SPLITS:
        rows_read = 0
        for batch in ds[split].iter(batch_size=10_000):
            for user, title, text, debate_split in zip(*(batch[c] for c in HISTORY_COLUMNS)):
                if user not in wanted:
                    counts["rows of users without a label"] += 1
                elif not (text or "").strip():
                    counts["rows without text (votes, forfeited debates)"] += 1
                elif debate_split in HELD_OUT and not all_debates:
                    counts["text rows left out (held-out T2 debates)"] += 1
                else:
                    counts["text rows used"] += 1
                    items[user].append(f"{title or ''}\n{text}" if titles else text)
            rows_read += len(batch["user"])
            if max_rows is not None and rows_read >= max_rows:
                break
    return {user: "\n\n".join(parts) for user, parts in items.items()}, counts


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts with rows = true class, columns = predicted class."""
    return np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics(y: np.ndarray, pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    """Accuracy, and F1 averaged over all classes of the label (a class neither present nor predicted counts 0)."""
    confusion = confusion_matrix(y, pred, n_classes)
    true_positives = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.where(denominator > 0, 2 * true_positives / np.maximum(denominator, 1), 0.0)
    return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1))}


def fit_best(x: Dict[str, csr_matrix], y: Dict[str, np.ndarray], n_classes: int, class_weight: Optional[str],
             select: str) -> Tuple[float, float, np.ndarray]:
    """Fit one model per C, keep the best validation `select` metric. Returns (C, its validation value, test predictions)."""
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight=class_weight, max_iter=2000).fit(x["train"], y["train"])
        value = metrics(y["validation"], model.predict(x["validation"]), n_classes)[select]
        if best is None or value > best[1]:
            best = (c, value, model.predict(x["test"]))
    return best


def bootstrap_by_user(y: np.ndarray, preds: Dict[str, np.ndarray], n_classes: int,
                      resamples: int) -> Dict[str, Dict[str, Interval]]:
    """95% interval of each test metric per model, resampling test users.

    `accuracy_minus_majority` is the model's accuracy minus the majority model's on the same resample.
    """
    rng = np.random.default_rng(SEED)
    draws = {name: defaultdict(list) for name in preds}
    for _ in range(resamples):
        rows = rng.integers(0, len(y), len(y))
        majority_accuracy = float(np.mean(y[rows] == preds["majority"][rows]))
        for name, pred in preds.items():
            for metric, value in metrics(y[rows], pred[rows], n_classes).items():
                draws[name][metric].append(value)
            draws[name]["accuracy_minus_majority"].append(draws[name]["accuracy"][-1] - majority_accuracy)
    return {name: {metric: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
                   for metric, values in by_metric.items()} for name, by_metric in draws.items()}


def random_guess(y: np.ndarray, class_frequencies: np.ndarray, draws: int) -> Dict[str, object]:
    """Mean test metrics of predictions drawn at random with the given class frequencies, with the central
    95% of the draws as the interval."""
    rng = np.random.default_rng(SEED)
    n_classes = len(class_frequencies)
    values = defaultdict(list)
    for _ in range(draws):
        for metric, value in metrics(y, rng.choice(n_classes, len(y), p=class_frequencies), n_classes).items():
            values[metric].append(value)
    return {**{metric: float(np.mean(v)) for metric, v in values.items()},
            "ci95": {metric: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for metric, v in values.items()}}


def main() -> None:
    args = parse_args()
    users, labels, class_names = load_users(args.dataset_repo, args.users_config, args.dataset_revision)
    wanted = {user for split in SPLITS for i, user in enumerate(users[split])
              if any(labels[split][name][i] >= 0 for name in class_names)}
    print(f"[{args.users_config}] {sum(len(users[s]) for s in SPLITS)} users, {len(wanted)} with at least one label; "
          f"labels: {', '.join(class_names)}")

    texts, counts = load_texts(args.dataset_repo, args.history_config, args.dataset_revision, wanted,
                               args.all_debates, not args.no_titles, SMOKE_ROWS if args.smoke else None)
    for name, n in counts.most_common():
        print(f"[{args.history_config}] {name}: {n}")
    words = np.array([len(text.split()) for text in texts.values()])
    print(f"[TEXT] {len(texts)} labelled users have text; words per user: median {np.median(words):.0f}, "
          f"10th percentile {np.percentile(words, 10):.0f}, 90th percentile {np.percentile(words, 90):.0f}")

    # Rows of the feature matrices: the users of each split who have text.
    kept = {s: np.array([i for i, user in enumerate(users[s]) if user in texts], dtype=int) for s in SPLITS}
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1 if args.smoke else args.min_df, dtype=np.float32)
    x_all = {"train": vectorizer.fit_transform([texts[users["train"][i]] for i in kept["train"]])}
    for s in HELD_OUT:
        x_all[s] = vectorizer.transform([texts[users[s][i]] for i in kept[s]])
    print(f"[TFIDF] {x_all['train'].shape[1]} word features from {x_all['train'].shape[0]} training users")

    results = {}
    print("\n=== TEST, held-out users: accuracy and macro-F1 with 95% intervals ===")
    for name, classes in class_names.items():
        n_classes = len(classes)
        rows = {s: np.flatnonzero(labels[s][name][kept[s]] >= 0) for s in SPLITS}
        y = {s: labels[s][name][kept[s]][rows[s]] for s in SPLITS}
        if min(len(y[s]) for s in SPLITS) == 0 or len(set(y["train"])) < 2:
            print(f"  {name}: skipped, too few labelled users with text")
            continue
        x = {s: x_all[s][rows[s]] for s in SPLITS}
        train_counts = np.bincount(y["train"], minlength=n_classes)
        majority = int(train_counts.argmax())
        preds = {"majority": np.full(len(y["test"]), majority)}
        notes = {"majority": f"always '{classes[majority]}'", "random": f"mean of {args.bootstrap} draws"}
        chosen = {}
        for model, (class_weight, select) in TFIDF_MODELS.items():
            c, value, preds[model] = fit_best(x, y, n_classes, class_weight, select)
            chosen[model] = {"C": c, f"validation_{select}": value}
            notes[model] = f"C={c:g}, validation {METRIC_NAMES[select]} {value:.3f}"
        intervals = bootstrap_by_user(y["test"], preds, n_classes, args.bootstrap)
        models = {model: {**metrics(y["test"], pred, n_classes), "ci95": intervals[model], **chosen.get(model, {})}
                  for model, pred in preds.items()}
        models["majority"]["predicts"] = classes[majority]
        for model in TFIDF_MODELS:
            models[model]["confusion_rows_true_columns_predicted"] = confusion_matrix(y["test"], preds[model], n_classes).tolist()
        models = {"majority": models.pop("majority"), "random": random_guess(y["test"], train_counts / train_counts.sum(), args.bootstrap), **models}
        labelled = {s: int((labels[s][name] >= 0).sum()) for s in SPLITS}
        results[name] = {
            "classes": classes,
            "users_scored": {s: len(y[s]) for s in SPLITS},
            "labelled_users_without_text": {s: labelled[s] - len(y[s]) for s in SPLITS},
            "test_class_counts": np.bincount(y["test"], minlength=n_classes).tolist(),
            "models": models,
        }
        print(f"  {name} ({' / '.join(classes)}): {len(y['train'])} train, {len(y['validation'])} validation, "
              f"{len(y['test'])} test users")
        for model, r in models.items():
            (acc_low, acc_high), (f1_low, f1_high) = r["ci95"]["accuracy"], r["ci95"]["macro_f1"]
            print(f"    {model:22s} accuracy {r['accuracy']:.3f} ({acc_low:.3f}-{acc_high:.3f})   "
                  f"macro-F1 {r['macro_f1']:.3f} ({f1_low:.3f}-{f1_high:.3f})   {notes[model]}")
        for model in TFIDF_MODELS:
            low, high = models[model]["ci95"]["accuracy_minus_majority"]
            print(f"    accuracy of {model} minus majority: "
                  f"{models[model]['accuracy'] - models['majority']['accuracy']:+.3f} ({low:+.3f} to {high:+.3f})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "revision": args.dataset_revision,
                                   "users_config": args.users_config, "history_config": args.history_config},
                       "options": {"all_debates": args.all_debates, "titles": not args.no_titles, "min_df": args.min_df,
                                   "smoke": args.smoke, "bootstrap_resamples": args.bootstrap, "C_grid": C_GRID},
                       "history_rows": dict(counts), "word_features": int(x_all["train"].shape[1]),
                       "labels": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
