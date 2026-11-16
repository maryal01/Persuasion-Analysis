"""T2 (DDO voter switching) baselines that use the listener and no debate text.

Predicts the label of a binary T2 config (ddo_switch: did a decided voter switch sides?) with logistic
regression on four groups of features:

  context      the voter's side before the debate and the debate's year
  profile      the voter's self-description in users.json: ideology, party, demographics, the 48 "big issues"
  record       how often the voter switched in the training debates (the row's own debate is never counted)
  record_past  the same, counting only training debates from earlier years than the row's debate

Rows, labels and splits come from the Hub dataset, so the test rows are the ones ModernBERT is scored on.
Only the id and label columns are read; the transcripts are not downloaded. C is chosen on validation
AUROC, the test split is scored once per model, and test AUROC gets a 95% interval from a bootstrap
over debates.

Two extra numbers show what a text-only model cannot do and what the profile really measures:
  same-debate pairs   among voters of one debate with the same side before, one of whom switched and one
                      stayed, how often the switcher gets the higher score (a text-only model gives both
                      the same input, so it scores 50%)
  unseen voters       the profile model trained only on voters who never appear in validation or test,
                      so it cannot recognise a frequent voter by their answers

Usage (from the repository root; needs scikit-learn):
    python scripts/t1_t2/listener_baseline.py --users ddo/users.json
    python scripts/t1_t2/listener_baseline.py --users ddo/users.json --output outputs/listener_baseline.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ijson
import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SPLITS = ("train", "validation", "test")
COLUMNS = ["debate_id", "voter", "before", "year", "label"]
PROFILE_FIELDS = ("political_ideology", "religious_ideology", "party", "president",
                  "gender", "education", "ethnicity", "income")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
SEED = 42

# Rows of the result table: name -> feature groups.
MODELS = {
    "Side before x year": ["context"],
    "Profile": ["profile"],
    "Voting record, earlier years only": ["record_past"],
    "Voting record": ["record"],
    "Profile + voting record": ["profile", "record"],
    "Profile + voting record + side x year": ["profile", "record", "context"],
}

Rows = Dict[str, List[Any]]  # one split as columns: debate_id, voter, before, year, label
Matrices = Tuple[csr_matrix, csr_matrix, csr_matrix]  # train, validation, test


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Listener-only logistic regression baselines for a binary T2 config.")
    ap.add_argument("--users", default="ddo/users.json", help="users.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", default="ddo_switch", help="binary T2 config (default: %(default)s)")
    ap.add_argument("--dataset-revision", default="t1-t2", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test debates")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def load_rows(repo: str, config: str, revision: str) -> Tuple[Dict[str, Rows], List[str]]:
    """Id and label columns of every split, plus the label names. Streams, so the transcripts stay on the Hub."""
    ds = load_dataset(repo, config, revision=revision, columns=COLUMNS, streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    names = ds["train"].features["label"].names
    if len(names) != 2:
        raise SystemExit(f"{config} has {len(names)} classes; this script handles binary configs only")
    splits = {}
    for split in SPLITS:
        columns: Rows = {c: [] for c in COLUMNS}
        for row in ds[split]:
            for c in COLUMNS:
                columns[c].append(row[c])
        splits[split] = columns
    return splits, names


def load_profiles(path: str, voters: set) -> Dict[str, Dict[str, str]]:
    """Profile fields of the given voters as {voter: {feature name: value}}; voters without a record are absent."""
    profiles = {}
    with open(path, "rb") as fh:
        for name, user in ijson.kvitems(fh, ""):
            if name not in voters:
                continue
            fields = {f: str(user.get(f)) for f in PROFILE_FIELDS}
            for issue, answer in (user.get("big_issues_dict") or {}).items():
                fields[f"issue:{issue}"] = str(answer)
            profiles[name] = fields
    return profiles


def one_hot(dicts: Dict[str, List[Dict[str, float]]]) -> Matrices:
    """Vectorize per-row feature dicts; the vocabulary comes from the training split."""
    vec = DictVectorizer(sparse=True)
    return (vec.fit_transform(dicts["train"]),) + tuple(vec.transform(dicts[s]) for s in SPLITS[1:])


def context_features(splits: Dict[str, Rows]) -> Matrices:
    dicts = {s: [{f"before={b}": 1.0, f"year={y}": 1.0, f"before={b}|year={y}": 1.0}
                 for b, y in zip(rows["before"], rows["year"])] for s, rows in splits.items()}
    return one_hot(dicts)


def profile_features(splits: Dict[str, Rows], profiles: Dict[str, Dict[str, str]]) -> Matrices:
    dicts = {s: [{f"{k}={v}": 1.0 for k, v in profiles[voter].items()} if voter in profiles else {"no_profile": 1.0}
                 for voter in rows["voter"]] for s, rows in splits.items()}
    return one_hot(dicts)


def record_features(splits: Dict[str, Rows], earlier_years_only: bool) -> Matrices:
    """log(1 + switches), log(1 + stays) and a no-record flag, from the voter's votes in training debates.

    A training row never counts itself: with `earlier_years_only` only debates from earlier years count,
    otherwise the row is left out of its own voter's totals. Validation and test debates are never counted.
    """
    train = splits["train"]
    by_voter = defaultdict(list)  # voter -> [(year, label)]
    for voter, year, label in zip(train["voter"], train["year"], train["label"]):
        by_voter[voter].append((year, label))

    def counts(voter: str, year: Optional[int], own_label: Optional[int]) -> Tuple[int, int]:
        votes = by_voter.get(voter, [])
        if earlier_years_only:
            votes = [] if year is None else [(y, l) for y, l in votes if y is not None and y < year]
            own_label = None
        switches = sum(l for _, l in votes)
        stays = len(votes) - switches
        if own_label is not None:  # leave the training row itself out
            switches, stays = switches - own_label, stays - (1 - own_label)
        return switches, stays

    raw = {}
    for s, rows in splits.items():
        own = rows["label"] if s == "train" else [None] * len(rows["label"])
        pairs = [counts(v, y, l) for v, y, l in zip(rows["voter"], rows["year"], own)]
        raw[s] = np.array([[np.log1p(sw), np.log1p(st), float(sw + st == 0)] for sw, st in pairs])
    mean, std = raw["train"].mean(axis=0), raw["train"].std(axis=0)
    std[std == 0] = 1.0
    return tuple(csr_matrix((raw[s] - mean) / std) for s in SPLITS)


def fit_best(x: Matrices, y_train: np.ndarray, y_val: np.ndarray,
             train_mask: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray]:
    """Fit one model per C, keep the best validation AUROC. Returns (C, validation AUROC, test scores).

    `train_mask` restricts training to the selected rows.
    """
    x_train = x[0] if train_mask is None else x[0][train_mask]
    y_train = y_train if train_mask is None else y_train[train_mask]
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x_train, y_train)
        val_auroc = roc_auc_score(y_val, model.decision_function(x[1]))
        if best is None or val_auroc > best[1]:
            best = (c, val_auroc, model.decision_function(x[2]))
    return best


def same_debate_pairs(y: np.ndarray, scores: np.ndarray, debates: Sequence[str],
                      sides: Sequence[str]) -> Tuple[float, int]:
    """Share of (switcher, stayer) pairs from one debate and one side before whose switcher scores higher
    (ties count half), and the number of such pairs."""
    index = defaultdict(list)
    for i, key in enumerate(zip(debates, sides)):
        index[key].append(i)
    wins, pairs = 0.0, 0
    for rows in index.values():
        rows = np.array(rows)
        pos, neg = scores[rows][y[rows] == 1], scores[rows][y[rows] == 0]
        wins += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        pairs += len(pos) * len(neg)
    return (float(wins / pairs) if pairs else float("nan")), pairs


def bootstrap_by_debate(y: np.ndarray, scores: Dict[str, np.ndarray], debates: Sequence[str],
                        resamples: int) -> Dict[str, Tuple[float, float]]:
    """95% interval of test AUROC per model, resampling whole debates (their votes are not independent)."""
    index = defaultdict(list)
    for i, d in enumerate(debates):
        index[d].append(i)
    groups = [np.array(v) for v in index.values()]
    rng = np.random.default_rng(SEED)
    draws = defaultdict(list)
    for _ in range(resamples):
        rows = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(set(y[rows])) < 2:
            continue
        for name, s in scores.items():
            draws[name].append(roc_auc_score(y[rows], s[rows]))
    return {name: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for name, v in draws.items()}


def main() -> None:
    args = parse_args()
    splits, label_names = load_rows(args.dataset_repo, args.dataset_config, args.dataset_revision)
    y = {s: np.array(rows["label"]) for s, rows in splits.items()}
    for s in SPLITS:
        counts = Counter(splits[s]["label"])
        print(f"[{args.dataset_config}] {s:10s} {len(y[s]):6d} rows  "
              + ", ".join(f"{n}={counts[i]}" for i, n in enumerate(label_names)))

    voters = {v for rows in splits.values() for v in rows["voter"]}
    profiles = load_profiles(args.users, voters)
    train_voters = set(splits["train"]["voter"])
    test_voters = splits["test"]["voter"]
    coverage = {
        "test_rows_with_profile": float(np.mean([v in profiles for v in test_voters])),
        "test_rows_with_training_votes": float(np.mean([v in train_voters for v in test_voters])),
    }
    earlier = defaultdict(lambda: 10**9)  # voter -> year of their first training vote
    for v, year in zip(splits["train"]["voter"], splits["train"]["year"]):
        if year is not None:
            earlier[v] = min(earlier[v], year)
    coverage["test_rows_with_earlier_training_votes"] = float(np.mean(
        [year is not None and earlier[v] < year for v, year in zip(test_voters, splits["test"]["year"])]))
    print(f"[COVERAGE] test rows whose voter has a profile record: {coverage['test_rows_with_profile']:.1%}; "
          f"whose voter votes in a training debate: {coverage['test_rows_with_training_votes']:.1%}; "
          f"in a training debate from an earlier year: {coverage['test_rows_with_earlier_training_votes']:.1%}")

    groups = {
        "context": context_features(splits),
        "profile": profile_features(splits, profiles),
        "record": record_features(splits, earlier_years_only=False),
        "record_past": record_features(splits, earlier_years_only=True),
    }
    results, test_scores = {}, {}
    for name, parts in MODELS.items():
        x = tuple(hstack([groups[p][i] for p in parts]).tocsr() for i in range(3))
        c, val_auroc, scores = fit_best(x, y["train"], y["validation"])
        test_scores[name] = scores
        results[name] = {"features": parts, "n_features": x[0].shape[1], "C": c, "validation_auroc": float(val_auroc),
                         "test_auroc": float(roc_auc_score(y["test"], scores))}

    # The profile again, trained only on voters who never appear in validation or test.
    held_out_voters = set(splits["validation"]["voter"]) | set(test_voters)
    unseen = np.array([v not in held_out_voters for v in splits["train"]["voter"]])
    name = "Profile, trained on unseen voters only"
    c, val_auroc, scores = fit_best(groups["profile"], y["train"], y["validation"], train_mask=unseen)
    test_scores[name] = scores
    results[name] = {"features": ["profile"], "n_features": groups["profile"][0].shape[1], "C": c,
                     "validation_auroc": float(val_auroc), "test_auroc": float(roc_auc_score(y["test"], scores)),
                     "training_rows": int(unseen.sum())}

    intervals = bootstrap_by_debate(y["test"], test_scores, splits["test"]["debate_id"], args.bootstrap)
    print(f"\n=== TEST AUROC, {label_names[1]} vs {label_names[0]} ({len(y['test'])} rows) ===")
    print(f"  {'model':40s} {'features':>8s} {'C':>6s} {'val':>6s} {'test':>6s}  {'95% interval':12s} {'same-debate pairs':>17s}")
    for name, r in results.items():
        r["test_auroc_ci95"] = intervals[name]
        low, high = intervals[name]
        r["same_debate_pair_accuracy"], r["same_debate_pairs"] = same_debate_pairs(
            y["test"], test_scores[name], splits["test"]["debate_id"], splits["test"]["before"])
        print(f"  {name:40s} {r['n_features']:8d} {r['C']:6g} {r['validation_auroc']:6.3f} {r['test_auroc']:6.3f}  "
              f"{low:.3f}-{high:.3f}  {r['same_debate_pair_accuracy']:16.1%}")
    print(f"  (same-debate pairs: {r['same_debate_pairs']} pairs; a text-only model scores 50.0% on them)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "config": args.dataset_config, "revision": args.dataset_revision},
                       "label_names": label_names, "coverage": coverage, "bootstrap_resamples": args.bootstrap,
                       "models": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
