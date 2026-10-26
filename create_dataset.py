import json
import ijson
import os
from collections import defaultdict
from datasets import ClassLabel, Dataset, DatasetDict
from huggingface_hub import login

REPO = "nlp-final-project11/persuasion-datasets" 
TEST_SIZE = 0.1                     
VAL_SIZE = 0.1                     
SEED = 42                        

def build_pairs(path):
    """CMV pairs.jsonl -> {original, reply, label}, two rows per input pair.
    """
    originals, replies, labels = [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            s = r["submission"]
            original = (s.get("title", "") + "\n\n" + s.get("selftext", "")).strip()
            for key, label in (("delta_comment", 1), ("nodelta_comment", 0)):
                reply = "\n\n".join(c.get("body", "") for c in r[key]["comments"]).strip()
                originals.append(original)
                replies.append(reply)
                labels.append(label)
    ds = Dataset.from_dict({"original": originals, "reply": replies, "label": labels})
    # ClassLabel gives the Hub nice label names.
    return ds.cast_column("label", ClassLabel(names=["nodelta", "delta"]))

def build_ddo(path, limit=None):
    """debate.org debates.json -> {original, reply, label}, two rows per decided debate.
    """
    originals, replies, labels = [], [], []
    with open(path) as fh:
        for i, (_id, d) in enumerate(ijson.kvitems(fh, "")):
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
                originals.append(resolution)
                replies.append(reply)
                labels.append(1 if status[side] == "Winning" else 0)

    ds = Dataset.from_dict({"original": originals, "reply": replies, "label": labels})
    return ds.cast_column("label", ClassLabel(names=["lost", "won"]))

def main():
    token = os.environ.get("HF_TOKEN")
    login(token=token)
    sources = {
        "cmv": ("cmv/pairs.jsonl", build_pairs),
        "ddo": ("ddo/debates.json", build_ddo),
    }
    for name, (path, builder) in sources.items():
        ds = builder(path)
        pos = sum(ds["label"])
        print(f"[{name}] built {len(ds)} rows ({pos} positive / {len(ds) - pos} negative)")

        tt = ds.train_test_split(test_size=TEST_SIZE, seed=SEED, stratify_by_column="label")
        val_rel = VAL_SIZE / (1 - TEST_SIZE)  # validation as a fraction of the remaining data
        tv = tt["train"].train_test_split(test_size=val_rel, seed=SEED, stratify_by_column="label")

        dsd = DatasetDict(train=tv["train"], validation=tv["test"], test=tt["test"])
        print(f"[{name}] split -> " + ", ".join(f"{k}={len(v)}" for k, v in dsd.items()))
        dsd.push_to_hub(REPO, config_name=name, private=True, token=token)
        print(f"[{name}] pushed -> https://huggingface.co/datasets/{REPO} (config: {name}, private)")

if __name__ == "__main__":
    main()
