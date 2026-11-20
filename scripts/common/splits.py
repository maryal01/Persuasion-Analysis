"""Another way to split a DDO config than the one stored on the Hub: by time.

The Hub configs are split by debate at random (80/10/10; scripts/create_dataset.py). The project proposal
also asks for a temporal split as a robustness check: train on debates through 2015, test on later ones.
It is made from the stored splits, so no second dataset is needed:

    train        debates through 2015 from the stored train and test splits
    validation   debates through 2015 from the stored validation split
    test         debates after 2015 from all three stored splits

The stored splits never share a debate, so these three parts do not either. Every row with a year is
used exactly once; rows without a year are left out.

Usage (scripts/ on the import path):
    from common.splits import SPLIT_KINDS, temporal_split
    ds = temporal_split(ds)              # a DatasetDict with a `year` column
    splits = temporal_columns(splits)    # the same for splits held as columns (name -> list)
"""
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, concatenate_datasets

SPLIT_KINDS = ("random", "temporal")
LAST_TRAIN_YEAR = 2015
PARTS = ("train", "validation", "test")


def temporal_part(stored_split: str, year: Optional[int], last_train_year: int = LAST_TRAIN_YEAR) -> Optional[str]:
    """The part of the temporal split a row belongs to, from its stored split and the year of its debate;
    None for a row without a year."""
    if year is None:
        return None
    if year > last_train_year:
        return "test"
    return "validation" if stored_split == "validation" else "train"


def describe(sizes: Dict[str, int], years: Dict[str, List[int]], dropped: int, last_train_year: int) -> str:
    spans = {part: f"{min(years[part])}-{max(years[part])}" if years[part] else "no rows" for part in PARTS}
    return (f"[SPLIT] temporal, train through {last_train_year}: "
            + ", ".join(f"{part} {sizes[part]} rows ({spans[part]})" for part in PARTS)
            + f"; {dropped} rows without a year left out")


def temporal_split(ds: DatasetDict, last_train_year: int = LAST_TRAIN_YEAR) -> DatasetDict:
    """Re-split a DatasetDict with a `year` column by time, as described at the top of this file."""
    if any("year" not in split.column_names for split in ds.values()):
        raise SystemExit("--split temporal needs a config with a `year` column (ddo_switch, ddo_undecided)")
    pieces = {part: [] for part in PARTS}
    years = {part: [] for part in PARTS}
    dropped = 0
    for stored in PARTS:  # a fixed order, so the rows of a part are always in the same order
        split_years = list(ds[stored]["year"])
        target = [temporal_part(stored, year, last_train_year) for year in split_years]
        dropped += sum(part is None for part in target)
        for part in PARTS:
            rows = [i for i, t in enumerate(target) if t == part]
            if rows:
                pieces[part].append(ds[stored].select(rows))
                years[part] += [split_years[i] for i in rows]
    empty = [part for part in PARTS if not pieces[part]]
    if empty:
        raise SystemExit(f"--split temporal leaves no rows for: {', '.join(empty)}")
    out = DatasetDict({part: concatenate_datasets(pieces[part]) for part in PARTS})
    print(describe({part: len(out[part]) for part in PARTS}, years, dropped, last_train_year))
    return out


def temporal_columns(splits: Dict[str, Dict[str, List[Any]]], last_train_year: int = LAST_TRAIN_YEAR) -> Dict[str, Dict[str, List[Any]]]:
    """The same split for rows held as columns (split -> column name -> values), with the same row order."""
    columns = list(splits[PARTS[0]])
    out = {part: {column: [] for column in columns} for part in PARTS}
    dropped = 0
    for stored in PARTS:
        for i, year in enumerate(splits[stored]["year"]):
            part = temporal_part(stored, year, last_train_year)
            if part is None:
                dropped += 1
                continue
            for column in columns:
                out[part][column].append(splits[stored][column][i])
    empty = [part for part in PARTS if not out[part]["year"]]
    if empty:
        raise SystemExit(f"--split temporal leaves no rows for: {', '.join(empty)}")
    print(describe({part: len(out[part]["year"]) for part in PARTS}, {part: out[part]["year"] for part in PARTS},
                   dropped, last_train_year))
    return out
