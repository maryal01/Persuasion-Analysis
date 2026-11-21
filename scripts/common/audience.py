"""Voter profiles as text, for the "text + audience" models (T2 now; T4 and the simulated audience later).

A profile is one voter's six T3 labels (economy, social issues, personal liberties, foreign policy,
religion, political ideology group), written as one line:

    Voter profile: economy: left; social issues: liberal; personal liberties: permissive; foreign policy: unknown; religion: nonreligious; political ideology group: liberal

A voter with no known label gets "Voter profile: unknown". The line is appended to `original`, i.e. it
sits in front of the debate, after the resolution and the voter's side before the debate. Column and
class names are read from the dataset.

Sources (`--audience` of scripts/train_delta.py):
    self       the labels the voter reported themselves (`ddo_users`)
    inferred   the labels the T3 model predicted from the voter's text (`ddo_profiles_inferred`, written
               by scripts/t3/infer_profiles.py); a voter without text has no profile
    filled     the reported label where there is one, the predicted label otherwise

Control: with shuffle=True every voter of a split gets the profile of another voter of that split, the
same one on all of their rows. A model that uses the profile should score lower on such input.

Usage (from the repository root):
    python scripts/common/audience.py --dataset-config ddo_switch --audience filled   # coverage; changes nothing
In code (scripts/ on the import path):
    from common.audience import add_profiles
    ds = add_profiles(ds, "self")      # a DatasetDict with `voter` and `original` columns
"""
import argparse
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datasets import ClassLabel, DatasetDict, load_dataset

PROFILES_REPO = "nlp-final-project11/persuasion-datasets"
PROFILES_REVISION = "main"
SELF_CONFIG = "ddo_users"
INFERRED_CONFIG = "ddo_profiles_inferred"
AUDIENCE_SOURCES = ("none", "self", "inferred", "filled")
UNKNOWN_PROFILE = "Voter profile: unknown"
SEED = 42

Labels = Dict[str, Dict[str, Optional[str]]]  # user -> label column -> class name, None when unknown


def profile_line(values: Dict[str, Optional[str]]) -> str:
    """One voter's labels (label column -> class name, None when unknown) as a line of text."""
    if all(value is None for value in values.values()):
        return UNKNOWN_PROFILE
    return "Voter profile: " + "; ".join(f"{column.replace('_', ' ')}: {value or 'unknown'}" for column, value in values.items())


def load_labels(config: str, repo: str = PROFILES_REPO, revision: str = PROFILES_REVISION) -> Tuple[Labels, Set[str]]:
    """The label (ClassLabel) columns of a users table as class names per user, and the users flagged
    `in_training` if the table has that column (the T3 model was trained on them). Streams the table."""
    ds = load_dataset(repo, config, revision=revision, streaming=True, token=os.environ.get("HF_TOKEN"))
    features = next(iter(ds.values())).features
    if features is None:
        raise SystemExit(f"{config} has no feature metadata on the Hub, so its label columns cannot be found")
    class_names = {name: feature.names for name, feature in features.items() if isinstance(feature, ClassLabel)}
    flag = ["in_training"] if "in_training" in features else []
    labels, in_training = {}, set()
    for split in ds.values():  # these tables are split by user for T3; a T2 voter can be in any split
        for row in split.select_columns(["user", *class_names, *flag]):
            labels[row["user"]] = {name: None if row[name] is None else classes[row[name]]
                                   for name, classes in class_names.items()}
            if flag and row["in_training"]:
                in_training.add(row["user"])
    return labels, in_training


def load_profiles(source: str, repo: str = PROFILES_REPO, revision: str = PROFILES_REVISION) -> Tuple[Dict[str, str], Set[str]]:
    """user -> profile line for a source, and the users whose line holds a label the T3 model predicted
    for a user it was trained on (always empty for `self`)."""
    if source not in AUDIENCE_SOURCES or source == "none":
        raise SystemExit(f"unknown audience source {source!r}; choose from {AUDIENCE_SOURCES[1:]}")
    if source == "self":
        labels, in_sample = load_labels(SELF_CONFIG, repo, revision)[0], set()
    elif source == "inferred":
        labels, in_sample = load_labels(INFERRED_CONFIG, repo, revision)
    else:
        reported = load_labels(SELF_CONFIG, repo, revision)[0]
        predicted, trained_on = load_labels(INFERRED_CONFIG, repo, revision)
        labels, in_sample = {}, set()
        for user, values in reported.items():
            guesses = predicted.get(user, {})
            labels[user] = {column: value if value is not None else guesses.get(column) for column, value in values.items()}
            if user in trained_on and labels[user] != values:
                in_sample.add(user)
    return {user: profile_line(values) for user, values in labels.items()}, in_sample


def swap_voters(voters: Iterable[str], seed: int = SEED) -> Dict[str, str]:
    """Control: every distinct voter mapped to another one (the next in a seeded random order), so nobody
    keeps their own profile and a voter gets the same wrong profile on all rows."""
    distinct = sorted(set(voters))
    random.Random(seed).shuffle(distinct)
    return {voter: distinct[(i + 1) % len(distinct)] for i, voter in enumerate(distinct)}


def profiles_for(voters: List[str], lines: Dict[str, str], shuffle: bool = False) -> List[str]:
    """The profile line of each row's voter; with shuffle, of the voter they are swapped with."""
    donor = swap_voters(voters) if shuffle else {}
    return [lines.get(donor.get(voter, voter), UNKNOWN_PROFILE) for voter in voters]


def add_profiles(ds: DatasetDict, source: str, shuffle: bool = False, lines: Optional[Dict[str, str]] = None,
                 revision: str = PROFILES_REVISION) -> DatasetDict:
    """Append each row's voter profile to `original`. `lines` (user -> profile line) is loaded for
    `source` from branch `revision` of the dataset repo when not given."""
    if any("voter" not in split.column_names for split in ds.values()):
        raise SystemExit("--audience needs a config with a `voter` column (ddo_switch, ddo_undecided)")
    in_sample: Set[str] = set()
    if lines is None:
        lines, in_sample = load_profiles(source, revision=revision)
    out = {}
    for name, split in ds.items():
        voters = list(split["voter"])
        own = profiles_for(voters, lines)
        used = profiles_for(voters, lines, shuffle)
        known = sum(line != UNKNOWN_PROFILE for line in own) / max(len(own), 1)
        note = f"; shuffled: {sum(a != b for a, b in zip(own, used)) / max(len(own), 1):.1%} of rows got a different line" if shuffle else ""
        if in_sample:
            note += f"; {sum(voter in in_sample for voter in voters) / max(len(voters), 1):.1%} of rows use a label predicted for a user the T3 model was trained on"
        print(f"[AUDIENCE] {name}: {known:.1%} of {len(own)} rows have a voter with a known {source} profile{note}")
        original = [f"{text}\n{line}" for text, line in zip(split["original"], used)]
        out[name] = split.remove_columns("original").add_column("original", original)
    return DatasetDict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage of the voter profiles on a T2 config; changes nothing.")
    ap.add_argument("--dataset-repo", default=PROFILES_REPO)
    ap.add_argument("--dataset-config", default="ddo_switch", help="config with a `voter` column (default: %(default)s)")
    ap.add_argument("--dataset-revision", default=PROFILES_REVISION, help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--audience", choices=AUDIENCE_SOURCES[1:], default="self", help="profile source (default: %(default)s)")
    ap.add_argument("--audience-revision", default=PROFILES_REVISION, help="branch that holds the profile tables (default: %(default)s)")
    args = ap.parse_args()
    lines, in_sample = load_profiles(args.audience, args.dataset_repo, args.audience_revision)
    print(f"[{args.audience}] {len(lines)} users, {sum(line != UNKNOWN_PROFILE for line in lines.values())} with a known profile")
    ds = load_dataset(args.dataset_repo, args.dataset_config, revision=args.dataset_revision, streaming=True,
                      columns=["voter"], token=os.environ.get("HF_TOKEN"))
    for name, split in ds.items():
        voters = [row["voter"] for row in split]
        own = profiles_for(voters, lines)
        shuffled = profiles_for(voters, lines, shuffle=True)
        complete = sum(line != UNKNOWN_PROFILE and ": unknown" not in line for line in own)
        print(f"[{args.dataset_config}] {name}: {len(voters)} rows, {len(set(voters))} voters; known profile on "
              f"{sum(line != UNKNOWN_PROFILE for line in own) / len(own):.1%} of rows; all labels known on "
              f"{complete / len(own):.1%}; voters without a row in the profile table: {len({v for v in voters if v not in lines})}; "
              f"rows that use a label predicted for a T3 training user: {sum(v in in_sample for v in voters) / len(voters):.1%}; "
              f"shuffling changes the line on {sum(a != b for a, b in zip(own, shuffled)) / len(own):.1%} of rows")
    print("most common lines in the last split read:")
    for line, n in Counter(own).most_common(3):
        print(f"  {n:6d}  {line}")


if __name__ == "__main__":
    main()
