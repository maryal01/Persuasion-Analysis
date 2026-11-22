"""RQ2 transfer: CMV persuasion (T1) and DDO switching (T2) in one common format.

The proposal compares training on one platform and testing on the other, and training on both. For that,
both tasks are written the same way: a claim someone holds, the arguments for the other side, and
whether that person was persuaded.

    original   Claim: <DDO resolution, or CMV post title>
               Position before: <Pro or Con>        (a CMV poster holds their own claim, so always Pro)
    reply      the arguments for the other side
                 DDO: every turn of the debater on the side the voter did not hold, in round order
                 CMV: the reply branch of one challenger, as in the `cmv` config
    label      not_persuaded / persuaded            (DDO: stayed / switched; CMV: no delta / delta)

Three configs, written as parquet files and, with --push, uploaded to a branch of the dataset repo:

    ddo_common    the rows, splits and other columns of `ddo_switch`, so its test votes are the ones the
                  T2 models are scored on
    cmv_common    the rows, splits and other columns of `cmv`
    joint_common  train = both training splits; validation and test = those of ddo_common, because the
                  question is whether CMV data helps on DDO

Rows, labels and splits come from the Hub configs; only the texts are rebuilt, from the raw corpora:
the turns of each side from debates.json, the post titles from pairs.jsonl. This version has no
audience profile; the inputs are text only.

Usage (from the repository root):
    python scripts/t1_t2/build_common.py --out outputs/common                 # build and check, upload nothing
    python scripts/t1_t2/build_common.py --out outputs/common --push rq2      # also upload to Hub branch rq2
"""
import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import ijson
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

from listener_baseline import load_rows

DATASET_REPO = "nlp-final-project11/persuasion-datasets"
LABEL_NAMES = ["not_persuaded", "persuaded"]
SPLITS = ("train", "validation", "test")
SIDES = ("Pro", "Con")
DDO_EXTRA = ("after", "filled_other_criteria")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the common-format configs for the CMV/DDO transfer experiments.")
    ap.add_argument("--debates", default="ddo/debates.json", help="debates.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--pairs", default="cmv/pairs.jsonl", help="pairs.jsonl of Webis-CMV-20 (default: %(default)s)")
    ap.add_argument("--dataset-repo", default=DATASET_REPO)
    ap.add_argument("--dataset-revision", default="main", help="branch that holds `ddo_switch` and `cmv` (default: %(default)s)")
    ap.add_argument("--out", default="outputs/common", help="folder for the parquet files (default: %(default)s)")
    ap.add_argument("--push", metavar="BRANCH", help="upload the three configs to this branch of the dataset repo "
                                                     "(created from the source branch if it does not exist)")
    return ap.parse_args()


def view(claim: str, position: str) -> str:
    """The first text of an input: the claim and the position the person held before reading the arguments."""
    return f"Claim: {claim}\nPosition before: {position}"


def side_arguments(debate: Dict[str, Any]) -> Dict[str, Any]:
    """The turns of each side of a debate joined by blank lines, in round order. Empty turns and the
    placeholder 'forfeit' are left out, as in scripts/create_dataset.py's debate_transcript. A turn whose
    side is neither Pro nor Con belongs to nobody and is left out too; `unassigned` counts them."""
    turns: Dict[str, List[str]] = {side: [] for side in SIDES}
    unassigned = 0
    for rnd in debate.get("rounds", []):
        for turn in rnd:
            text = (turn.get("text") or "").strip()
            if text.lower() in ("", "forfeit"):
                continue
            if turn.get("side") in turns:
                turns[turn["side"]].append(text)
            else:
                unassigned += 1
    return {**{side: "\n\n".join(texts) for side, texts in turns.items()}, "unassigned": unassigned}


def read_debates(path: str, wanted: set) -> Dict[str, Dict[str, Any]]:
    """debate id -> {'title', 'Pro', 'Con', 'unassigned'} for the wanted debates."""
    debates = {}
    with open(path, "rb") as fh:
        for debate_id, d in ijson.kvitems(fh, ""):
            if debate_id in wanted:
                debates[debate_id] = {"title": (d.get("title", "") or "").strip(), **side_arguments(d)}
    odd = [d for d in debates.values() if d["unassigned"]]
    if odd:
        print(f"[ddo_common] {sum(d['unassigned'] for d in odd)} turns in {len(odd)} debates have a side that is neither "
              f"Pro nor Con in {path}; they are left out")
    return debates


def read_titles(path: str) -> Dict[str, str]:
    """CMV post id -> post title."""
    titles = {}
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            titles[record["submission_id"]] = (record["submission"].get("title") or "").strip()
    return titles


def to_dataset(columns: Dict[str, List[Any]]) -> Dataset:
    return Dataset.from_dict(columns).cast_column("label", ClassLabel(names=LABEL_NAMES))


def build_ddo(repo: str, revision: str, debates_path: str) -> DatasetDict:
    """`ddo_switch` with the first text rewritten and the debate cut down to the other side's turns."""
    splits, names = load_rows(repo, "ddo_switch", revision, extra=DDO_EXTRA)
    if len(names) != 2:
        raise SystemExit(f"ddo_switch has the classes {names}; expected two (stayed, switched)")
    wanted = {debate for rows in splits.values() for debate in rows["debate_id"]}
    debates = read_debates(debates_path, wanted)
    if wanted - set(debates):
        raise SystemExit(f"{len(wanted - set(debates))} debates of ddo_switch are not in {debates_path}")
    out = {}
    for split in SPLITS:
        rows = splits[split]
        if set(rows["before"]) - set(SIDES):
            raise SystemExit(f"ddo_switch has voters with the sides {sorted(set(rows['before']))} before; expected {SIDES}")
        other = [SIDES[1 - SIDES.index(before)] for before in rows["before"]]
        out[split] = to_dataset({
            "original": [view(debates[d]["title"], before) for d, before in zip(rows["debate_id"], rows["before"])],
            "reply": [debates[d][side] for d, side in zip(rows["debate_id"], other)],
            "label": rows["label"],  # the second class of ddo_switch is the voter who switched
            "platform": ["ddo"] * len(other),
            **{c: rows[c] for c in ("debate_id", "voter", "before", "after", "year", "filled_other_criteria")},
        })
    print(f"[ddo_common] classes of ddo_switch {names} -> {LABEL_NAMES}")
    return DatasetDict(out)


def build_cmv(repo: str, revision: str, pairs_path: str) -> DatasetDict:
    """`cmv` with the post replaced by its title; the reply branch is kept as it is."""
    ds = load_dataset(repo, "cmv", revision=revision, token=os.environ.get("HF_TOKEN"))
    names = ds["train"].features["label"].names
    if len(names) != 2:
        raise SystemExit(f"cmv has the classes {names}; expected two (nodelta, delta)")
    titles = read_titles(pairs_path)
    out = {}
    for split in SPLITS:
        rows = {c: list(ds[split][c]) for c in ("reply", "label", "pair_id", "post_id")}
        missing = set(rows["post_id"]) - set(titles)
        if missing:
            raise SystemExit(f"{len(missing)} posts of cmv ({split}) are not in {pairs_path}")
        out[split] = to_dataset({
            "original": [view(titles[post], SIDES[0]) for post in rows["post_id"]],
            "reply": rows["reply"],
            "label": rows["label"],  # the second class of cmv is the reply that earned a delta
            "platform": ["cmv"] * len(rows["label"]),
            "pair_id": rows["pair_id"], "post_id": rows["post_id"],
        })
    print(f"[cmv_common] classes of cmv {names} -> {LABEL_NAMES}")
    return DatasetDict(out)


def joint(ddo: DatasetDict, cmv: DatasetDict) -> DatasetDict:
    """Both training splits in one, with the columns of ddo_common (empty for CMV rows); validation and
    test are DDO's."""
    n = cmv["train"].num_rows
    empty: Dict[str, List[Optional[Any]]] = {c: [None] * n for c in ddo["train"].column_names
                                             if c not in ("original", "reply", "label", "platform")}
    shared = {c: list(cmv["train"][c]) for c in ("original", "reply", "label", "platform")}
    cmv_train = Dataset.from_dict({**shared, **empty}, features=ddo["train"].features)
    train = concatenate_datasets([ddo["train"], cmv_train.select_columns(ddo["train"].column_names)])
    return DatasetDict({"train": train, "validation": ddo["validation"], "test": ddo["test"]})


def describe(name: str, ds: DatasetDict) -> Dict[str, Any]:
    """Print and return the size of a config: rows, persuaded rows, rows without arguments, words per reply."""
    stats = {}
    for split in SPLITS:
        labels, replies = list(ds[split]["label"]), list(ds[split]["reply"])
        words = sorted(len(reply.split()) for reply in replies)
        stats[split] = {"rows": len(labels), "persuaded": int(sum(labels)),
                        "empty_reply": int(sum(not reply for reply in replies)),
                        "median_reply_words": words[len(words) // 2], "platforms": dict(Counter(ds[split]["platform"]))}
        s = stats[split]
        print(f"[{name}] {split}: {s['rows']} rows, {s['persuaded']} persuaded ({s['persuaded'] / s['rows']:.1%}), "
              f"{s['empty_reply']} without arguments, median {s['median_reply_words']} words per reply, {s['platforms']}")
    return stats


def main() -> None:
    args = parse_args()
    ddo = build_ddo(args.dataset_repo, args.dataset_revision, args.debates)
    cmv = build_cmv(args.dataset_repo, args.dataset_revision, args.pairs)
    configs = {"ddo_common": ddo, "cmv_common": cmv, "joint_common": joint(ddo, cmv)}
    stats = {}
    for name, ds in configs.items():
        stats[name] = describe(name, ds)
        for split in SPLITS:
            path = os.path.join(args.out, name, f"{split}.parquet")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ds[split].to_parquet(path)
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump({"source": {"repo": args.dataset_repo, "revision": args.dataset_revision}, "label_names": LABEL_NAMES,
                   "configs": stats}, fh, indent=2)
    print(f"\nWrote {args.out}/<config>/<split>.parquet and {args.out}/stats.json")

    if args.push:
        api = HfApi()
        api.create_branch(args.dataset_repo, repo_type="dataset", branch=args.push, revision=args.dataset_revision, exist_ok=True)
        for name, ds in configs.items():
            ds.push_to_hub(args.dataset_repo, config_name=name, revision=args.push, private=True,
                           token=os.environ.get("HF_TOKEN"))
            print(f"Uploaded {name} to branch {args.push} of {args.dataset_repo}")


if __name__ == "__main__":
    main()
