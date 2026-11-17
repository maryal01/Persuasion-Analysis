"""Voter profiles as text, for the "text + audience" models (T2 now; T4 and the simulated audience later).

A profile is one voter's label columns of the T3 table `ddo_users` (economy, social issues, personal
liberties, foreign policy, religion, political ideology group), written as one line:

    Voter profile: economy: left; social issues: liberal; personal liberties: permissive; foreign policy: unknown; religion: nonreligious; political ideology group: liberal

A voter with no known label gets "Voter profile: unknown". The line is appended to `original`, i.e. it
sits in front of the debate, after the resolution and the voter's side before the debate. Column and
class names are read from the dataset.

Sources (`--audience` of scripts/train_delta.py):
    self   the labels the voter reported themselves (ddo_users)

Control: with shuffle=True every voter of a split gets the profile of another voter of that split, the
same one on all of their rows. A model that uses the profile should score lower on such input.

Usage (from the repository root):
    python scripts/common/audience.py --dataset-config ddo_switch      # coverage and examples; changes nothing
In code (scripts/ on the import path):
    from common.audience import add_profiles
    ds = add_profiles(ds, "self")      # a DatasetDict with `voter` and `original` columns
"""
import argparse
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional

from datasets import ClassLabel, DatasetDict, load_dataset

PROFILES_REPO = "nlp-final-project11/persuasion-datasets"
PROFILES_REVISION = "main"
PROFILES_CONFIG = "ddo_users"
AUDIENCE_SOURCES = ("none", "self")
UNKNOWN_PROFILE = "Voter profile: unknown"
SEED = 42


def profile_line(values: Dict[str, Optional[str]]) -> str:
    """One voter's labels (label column -> class name, None when unknown) as a line of text."""
    if all(value is None for value in values.values()):
        return UNKNOWN_PROFILE
    return "Voter profile: " + "; ".join(f"{column.replace('_', ' ')}: {value or 'unknown'}" for column, value in values.items())


def load_self_reported(repo: str = PROFILES_REPO, revision: str = PROFILES_REVISION,
                       config: str = PROFILES_CONFIG) -> Dict[str, str]:
    """user -> profile line, from the label (ClassLabel) columns of the users table. Streams the table."""
    ds = load_dataset(repo, config, revision=revision, streaming=True, token=os.environ.get("HF_TOKEN"))
    features = next(iter(ds.values())).features
    if features is None:
        raise SystemExit(f"{config} has no feature metadata on the Hub, so its label columns cannot be found")
    class_names = {name: feature.names for name, feature in features.items() if isinstance(feature, ClassLabel)}
    lines = {}
    for split in ds.values():  # the users table is split by user for T3; a T2 voter can be in any split
        for row in split.select_columns(["user", *class_names]):
            lines[row["user"]] = profile_line({name: None if row[name] is None else classes[row[name]]
                                               for name, classes in class_names.items()})
    return lines


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


def add_profiles(ds: DatasetDict, source: str, shuffle: bool = False,
                 lines: Optional[Dict[str, str]] = None) -> DatasetDict:
    """Append each row's voter profile to `original`. `lines` (user -> profile line) is loaded for
    `source` when not given."""
    if source not in AUDIENCE_SOURCES or source == "none":
        raise SystemExit(f"unknown audience source {source!r}; choose from {AUDIENCE_SOURCES[1:]}")
    if any("voter" not in split.column_names for split in ds.values()):
        raise SystemExit("--audience needs a config with a `voter` column (ddo_switch, ddo_undecided)")
    if lines is None:
        lines = load_self_reported()
    out = {}
    for name, split in ds.items():
        voters = list(split["voter"])
        own = profiles_for(voters, lines)
        used = profiles_for(voters, lines, shuffle)
        known = sum(line != UNKNOWN_PROFILE for line in own) / max(len(own), 1)
        note = f"; shuffled: {sum(a != b for a, b in zip(own, used)) / max(len(own), 1):.1%} of rows got a different line" if shuffle else ""
        print(f"[AUDIENCE] {name}: {known:.1%} of {len(own)} rows have a voter with a known {source} profile{note}")
        original = [f"{text}\n{line}" for text, line in zip(split["original"], used)]
        out[name] = split.remove_columns("original").add_column("original", original)
    return DatasetDict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage of the voter profiles on a T2 config; changes nothing.")
    ap.add_argument("--dataset-repo", default=PROFILES_REPO)
    ap.add_argument("--dataset-config", default="ddo_switch", help="config with a `voter` column (default: %(default)s)")
    ap.add_argument("--dataset-revision", default=PROFILES_REVISION, help="branch of the dataset repo (default: %(default)s)")
    args = ap.parse_args()
    lines = load_self_reported(args.dataset_repo, args.dataset_revision)
    print(f"[{PROFILES_CONFIG}] {len(lines)} users, {sum(line != UNKNOWN_PROFILE for line in lines.values())} with a known profile")
    ds = load_dataset(args.dataset_repo, args.dataset_config, revision=args.dataset_revision, streaming=True,
                      columns=["voter"], token=os.environ.get("HF_TOKEN"))
    for name, split in ds.items():
        voters = [row["voter"] for row in split]
        own = profiles_for(voters, lines)
        shuffled = profiles_for(voters, lines, shuffle=True)
        known_labels = Counter(line.count(": ") - 1 - line.count(": unknown") for line in own if line != UNKNOWN_PROFILE)
        print(f"[{args.dataset_config}] {name}: {len(voters)} rows, {len(set(voters))} voters; known profile on "
              f"{sum(line != UNKNOWN_PROFILE for line in own) / len(own):.1%} of rows; all labels known on "
              f"{known_labels[max(known_labels, default=0)] / len(own):.1%}; voters missing from {PROFILES_CONFIG}: "
              f"{len({v for v in voters if v not in lines})}; shuffling changes the line on "
              f"{sum(a != b for a, b in zip(own, shuffled)) / len(own):.1%} of rows")
    print("most common lines in the last split read:")
    for line, n in Counter(own).most_common(3):
        print(f"  {n:6d}  {line}")


if __name__ == "__main__":
    main()
