"""A voter's voting record: the counts, and the same counts as one line of text.

How often did a voter's votes in the training debates of a T2 config end in each class (ddo_switch:
stayed, switched; ddo_undecided: tie, pro, con)? Two scripts use this:

    scripts/t1_t2/listener_baseline.py   the counts as features of a logistic regression, without text
    scripts/train_delta.py               `--voter-record` writes them in front of the debate, as a line:

        Voting record in other debates: stayed: 11-20; switched: 2

The rules are the same for both, so that the two can be compared:
  - only votes in TRAINING debates count, never votes in validation or test debates;
  - a training row never counts its own vote, which would give its label away;
  - kind "earlier": only training debates from earlier years than the row's debate count. Kind "all"
    also counts votes the voter cast after the debate of the row.

A voter without such votes gets "Voting record in other debates: none". Counts are written as coarse
ranges (0, 1, 2, 3-5, 6-10, 11-20, 21-50, 51-100, over 100), so that the model sees few different values.
Class names are read from the dataset.

The voting record is not part of the audience description in the project proposal (that is the stance
profile, scripts/common/audience.py). A run that uses it is an extension of the proposal.

Control: with shuffle=True every voter of a split gets the record of another voter of that split, the
same swap as the profile control of scripts/common/audience.py.

In code (scripts/ on the import path):
    from common.voter_record import add_records
    ds = add_records(ds, "all")      # a DatasetDict with `voter`, `year`, `labels` and `original` columns
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from datasets import DatasetDict

from common.audience import swap_voters

RECORD_KINDS = ("none", "all", "earlier")
PREFIX = {"all": "Voting record in other debates: ", "earlier": "Voting record in debates of earlier years: "}
BUCKETS = ((0, "0"), (1, "1"), (2, "2"), (5, "3-5"), (10, "6-10"), (20, "11-20"), (50, "21-50"), (100, "51-100"))  # (largest count, text)
TOP_BUCKET = "over 100"


def record_counts(train_voters: Sequence[str], train_years: Sequence[Optional[int]], train_labels: Sequence[int],
                  voters: Sequence[str], years: Sequence[Optional[int]], n_classes: int, earlier_years_only: bool,
                  own_labels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Per row of (`voters`, `years`): how many of that voter's training votes ended in each class.

    `own_labels` is given when the rows ARE the training rows: each row's own vote is then left out of
    its voter's totals. With `earlier_years_only` only training debates from earlier years than the row's
    debate count (a row or a training vote without a year then counts nothing), which leaves the row's
    own vote out as well.
    """
    by_year: Dict[str, Dict[Optional[int], np.ndarray]] = defaultdict(dict)  # voter -> year -> votes per class
    for voter, year, label in zip(train_voters, train_years, train_labels):
        by_year[voter].setdefault(year, np.zeros(n_classes, dtype=int))[label] += 1
    counts = np.zeros((len(voters), n_classes), dtype=int)
    for i, (voter, year) in enumerate(zip(voters, years)):
        votes = by_year.get(voter)
        if not votes:
            continue
        if earlier_years_only:
            if year is not None:
                for vote_year, per_class in votes.items():
                    if vote_year is not None and vote_year < year:
                        counts[i] += per_class
        else:
            for per_class in votes.values():
                counts[i] += per_class
            if own_labels is not None:
                counts[i, own_labels[i]] -= 1
    return counts


def bucket(count: int) -> str:
    """A count as a coarse range."""
    for largest, text in BUCKETS:
        if count <= largest:
            return text
    return TOP_BUCKET


def record_line(per_class: Sequence[int], class_names: Sequence[str], kind: str) -> str:
    """One row's counts (votes per class, in the order of `class_names`) as a line of text."""
    if sum(per_class) == 0:
        return PREFIX[kind] + "none"
    return PREFIX[kind] + "; ".join(f"{name}: {bucket(int(count))}" for name, count in zip(class_names, per_class))


def add_records(ds: DatasetDict, kind: str, label_column: str = "labels", shuffle: bool = False) -> DatasetDict:
    """Append each row's voter's voting record to `original`, counted from the training split of `ds`."""
    if kind not in PREFIX:
        raise SystemExit(f"unknown voter record {kind!r}; choose from {RECORD_KINDS[1:]}")
    missing = {"voter", "year", label_column, "original"} - set(ds["train"].column_names)
    if missing:
        raise SystemExit(f"--voter-record needs the columns {sorted(missing)} (ddo_switch, ddo_undecided)")
    class_names = ds["train"].features[label_column].names
    train = [list(ds["train"][column]) for column in ("voter", "year", label_column)]
    out = {}
    for name, split in ds.items():
        voters, years = list(split["voter"]), list(split["year"])
        donor = swap_voters(voters) if shuffle else {}
        # A swapped record is another voter's, so the row's own vote is not in it and nothing is left out.
        own = list(split[label_column]) if name == "train" and not shuffle else None
        counts = record_counts(*train, [donor.get(v, v) for v in voters], years, len(class_names),
                               earlier_years_only=kind == "earlier", own_labels=own)
        lines = [record_line(row, class_names, kind) for row in counts]
        known = sum(not line.endswith(": none") for line in lines) / max(len(lines), 1)
        print(f"[VOTER RECORD] {name}: {known:.1%} of {len(lines)} rows have a voter with a record ({kind})"
              + ("; swapped between voters" if shuffle else ""))
        original = [f"{text}\n{line}" for text, line in zip(split["original"], lines)]
        out[name] = split.remove_columns("original").add_column("original", original)
    return DatasetDict(out)
