"""T2: how many votes have an input longer than the model reads?

scripts/train_delta.py cuts the input of a vote (resolution and the voter's side before, then the debate)
to MAX_LENGTH tokens, 4,096 in the runs so far; ModernBERT can read 8,192. This script counts, per T2 config
and split, the votes whose input is longer than each limit, so that it is known how much text the runs did
not read. When an input is too long the tokenizer cuts the debate at its end, so the last rounds are lost.

The rows (debate, split, side before) are streamed from the Hub dataset; the transcripts stay on the Hub.
The debate text is rebuilt from the raw corpus with scripts/create_dataset.py, the code that built the Hub
configs, and every debate is tokenized once.

Usage (from the repository root):
    python scripts/t1_t2/token_lengths.py --debates ddo/debates.json --output outputs/token_lengths.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import ijson
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from create_dataset import debate_transcript  # noqa: E402
from listener_baseline import SPLITS, load_rows  # noqa: E402
from train_delta import MAX_LENGTH, MODEL_NAME  # noqa: E402

BATCH = 256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Count the T2 votes whose input is longer than the model reads.")
    ap.add_argument("--debates", default="ddo/debates.json", help="debates.json of the DDO corpus (default: %(default)s)")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-config", nargs="+", default=["ddo_switch", "ddo_undecided"], help="T2 configs (default: both)")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--limits", type=int, nargs="+", default=[MAX_LENGTH, 8192],
                    help="token limits to count against (default: the training limit and ModernBERT's 8,192)")
    ap.add_argument("--model-name", default=MODEL_NAME, help="tokenizer (default: %(default)s)")
    ap.add_argument("--output", help="write the results as JSON to this path")
    return ap.parse_args()


def count_tokens(tok, texts: List[str]) -> List[int]:
    """Tokens per text, without the special tokens the model adds around an input."""
    counts = []
    for start in range(0, len(texts), BATCH):
        counts += [len(ids) for ids in tok(texts[start:start + BATCH], add_special_tokens=False)["input_ids"]]
    return counts


def debate_lengths(path: str, wanted: set, tok) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Tokens of the transcript of every wanted debate, and its title (for the resolution line)."""
    ids, transcripts, titles = [], [], {}
    with open(path, "rb") as fh:
        for debate_id, d in ijson.kvitems(fh, ""):
            if debate_id in wanted:
                ids.append(debate_id)
                transcripts.append(debate_transcript(d))
                titles[debate_id] = (d.get("title", "") or "").strip()
    return dict(zip(ids, count_tokens(tok, transcripts))), titles


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name, token=os.environ.get("HF_TOKEN"))
    # An input is two texts, so the model adds this many special tokens around them.
    specials = len(tok("a", "b")["input_ids"]) - len(tok("a", add_special_tokens=False)["input_ids"]) \
        - len(tok("b", add_special_tokens=False)["input_ids"])
    configs = {config: load_rows(args.dataset_repo, config, args.dataset_revision)[0] for config in args.dataset_config}
    wanted = {debate for splits in configs.values() for rows in splits.values() for debate in rows["debate_id"]}
    lengths, titles = debate_lengths(args.debates, wanted, tok)
    missing = wanted - set(lengths)
    if missing:
        raise SystemExit(f"{len(missing)} debates of the dataset are not in {args.debates}")
    print(f"{len(lengths)} debates tokenized with {args.model_name}; {specials} special tokens per input")

    # The first text of an input: the same for all voters of a debate with the same side before.
    firsts = sorted({(debate, before) for splits in configs.values() for rows in splits.values()
                     for debate, before in zip(rows["debate_id"], rows["before"])})
    first_tokens = dict(zip(firsts, count_tokens(
        tok, [f"Resolution: {titles[debate]}\nVoter's side before the debate: {before}" for debate, before in firsts])))

    results = {}
    print(f"\n{'config':14s} {'split':10s} {'votes':>7s} {'median':>7s} {'90%':>7s} {'max':>7s}"
          + "".join(f" {'over ' + format(limit, ','):>11s}" for limit in args.limits))
    for config, splits in configs.items():
        results[config] = {}
        for split in SPLITS:
            rows = splits[split]
            total = np.array([first_tokens[(debate, before)] + lengths[debate] + specials
                              for debate, before in zip(rows["debate_id"], rows["before"])])
            over = {limit: float(np.mean(total > limit)) for limit in args.limits}
            results[config][split] = {"votes": int(len(total)), "median_tokens": float(np.median(total)),
                                      "p90_tokens": float(np.percentile(total, 90)), "max_tokens": int(total.max()),
                                      "share_over": {str(limit): share for limit, share in over.items()}}
            print(f"{config:14s} {split:10s} {len(total):7d} {np.median(total):7.0f} {np.percentile(total, 90):7.0f} {total.max():7d}"
                  + "".join(f" {over[limit]:11.1%}" for limit in args.limits))
    print("(tokens of the whole input of a vote: resolution and side before, the debate, and the special tokens;")
    print(" the profile line of a text + audience run comes on top)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"dataset": {"repo": args.dataset_repo, "revision": args.dataset_revision}, "tokenizer": args.model_name,
                       "special_tokens_per_input": specials, "limits": args.limits, "configs": results}, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
