import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import ijson
from datasets import ClassLabel, Dataset, DatasetDict
from huggingface_hub import login

REPO = "nlp-final-project11/persuasion-datasets"
REVISION = "t1-t2"  # Hub branch to push to; main is the team's shared copy
TEST_SIZE = 0.1
VAL_SIZE = 0.1
SEED = 42

def split_of(group_id):
    """Deterministic split for a CMV post or DDO debate, so all of its rows land in the same split
    and every config built from the same source shares the assignment.
    """
    h = int(hashlib.md5(f"{SEED}:{group_id}".encode()).hexdigest(), 16) % 1000 / 1000
    if h < TEST_SIZE:
        return "test"
    if h < TEST_SIZE + VAL_SIZE:
        return "validation"
    return "train"

def to_splits(rows, label_names):
    """Rows (dicts with a 'split' key) -> DatasetDict(train, validation, test) with a ClassLabel 'label'."""
    columns = defaultdict(lambda: defaultdict(list))
    for row in rows:
        split = row.pop("split")
        for k, v in row.items():
            columns[split][k].append(v)
    return DatasetDict({
        name: Dataset.from_dict(dict(columns[name])).cast_column("label", ClassLabel(names=label_names))
        for name in ("train", "validation", "test")
    })

def build_cmv(path):
    """T1. Webis-CMV-20 pairs.jsonl -> {original, reply, label, pair_id, post_id}, two rows per pair.

    Pairs are split by post, so a post's pairs never straddle train and test; score a pair by
    whether its delta row outscores its no-delta row.
    """
    rows = []
    with open(path) as fh:
        for pair_id, line in enumerate(fh):
            r = json.loads(line)
            s = r["submission"]
            original = (s.get("title", "") + "\n\n" + s.get("selftext", "")).strip()
            split = split_of(r["submission_id"])
            for key, label in (("delta_comment", 1), ("nodelta_comment", 0)):
                reply = "\n\n".join(c.get("body", "") for c in r[key]["comments"]).strip()
                rows.append({"original": original, "reply": reply, "label": label,
                             "pair_id": pair_id, "post_id": r["submission_id"], "split": split})
    return {"cmv": to_splits(rows, ["nodelta", "delta"])}

def build_ddo(path, limit=None):
    """debate.org debates.json -> {original, reply, label, debate_id}, two rows per decided debate.

    Both sides of a debate land in the same split.
    """
    rows = []
    with open(path) as fh:
        for i, (debate_id, d) in enumerate(ijson.kvitems(fh, "")):
            if limit is not None and i >= limit:
                break
            if d.get("forfeit_label"):
                continue  # forfeits aren't won on the merits
            status = {
                d.get("participant_1_position"): d.get("participant_1_status"),
                d.get("participant_2_position"): d.get("participant_2_status"),
            }
            if {status.get("Pro"), status.get("Con")} != {"Winning", "Losing"}:
                continue  # need exactly one clear winner (skip ties / undecided)

            resolution = (d.get("title", "") or "").strip()
            by_side = defaultdict(list)
            for rnd in d.get("rounds", []):
                for turn in rnd:
                    by_side[turn["side"]].append(turn.get("text", ""))

            for side in ("Pro", "Con"):
                reply = "\n\n".join(by_side[side]).strip()
                if not reply:
                    continue  # side never spoke
                rows.append({"original": resolution, "reply": reply,
                             "label": 1 if status[side] == "Winning" else 0,
                             "debate_id": debate_id, "split": split_of(debate_id)})
    return {"ddo": to_splits(rows, ["lost", "won"])}

SOURCES = {
    "cmv": ("cmv/pairs.jsonl", build_cmv),
    "ddo": ("ddo/debates.json", build_ddo),
}

def parse_args():
    ap = argparse.ArgumentParser(description="Build the persuasion datasets and push them to the Hub.")
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                    help="which raw sources to build")
    ap.add_argument("--revision", default=REVISION, help="Hub branch to push to, created if missing (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="build and print stats without pushing")
    return ap.parse_args()

def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run:
        login(token=token)
    for source in args.sources:
        path, builder = SOURCES[source]
        for name, dsd in builder(path).items():
            names = dsd["train"].features["label"].names
            for split, ds in dsd.items():
                counts = Counter(ds["label"])
                print(f"[{name}] {split:10s} {len(ds):7d} rows  "
                      + ", ".join(f"{n}={counts[i]}" for i, n in enumerate(names)))
            if not args.dry_run:
                dsd.push_to_hub(REPO, config_name=name, private=True, token=token, revision=args.revision)
                print(f"[{name}] pushed -> https://huggingface.co/datasets/{REPO}/tree/{args.revision} "
                      f"(config: {name}, branch: {args.revision}, private)")

if __name__ == "__main__":
    main()
