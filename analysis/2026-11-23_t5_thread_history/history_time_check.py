"""Is the OP history in the Hub configs cmv_thread_history / cmv_thread_own_history written before the post?

The history items in `original` carry no timestamps, so each item is looked up in Webis-CMV-20 threads.jsonl by its text
(the first characters, lower-cased, whitespace collapsed, delta symbols removed) and gets the creation and edit time of the
matching post or comment. An item counts as "after" when every match was created after the post, "edited after" when it was
created before but every match was edited after the post (the stored text is the edited one), "unmatched" when no text matches.

Usage (one core, about 10 minutes, 3-4 GB RAM; needs a Hugging Face login):
    python analysis/2026-11-23_t5_thread_history/history_time_check.py <cmv_dir>/threads.jsonl.bz2 --output <out_dir>/history_time.json
"""
import argparse
import bz2
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset

HUB_REPO = "nlp-final-project11/persuasion-datasets"
CONFIGS = ("cmv_thread_history", "cmv_thread_own_history")
KEY_LENGTH = 80  # characters of normalized text used to find an item
ITEM_TAG = re.compile(r"(?:^|\n\n)\[(comment|post|reply in own thread)\] ")
SPEAKER = re.compile(r"^(OP|User): ")
DELTA_SYMBOL = re.compile(r"[∆Δ]|&(?:amp;)?#8710;|!delta", re.IGNORECASE)
MISSING = (None, "", "[deleted]", "[removed]")

Times = Tuple[int, Optional[int]]  # (created_utc, edited_utc or None)


def text_key(text: str) -> str:
    """The lookup key of a text: normalized first characters."""
    return " ".join(DELTA_SYMBOL.sub(" ", text).lower().split())[:KEY_LENGTH]


def history_items(original: str) -> List[Tuple[str, str]]:
    """(kind, text) of each history item before the last [SEP]; the speaker prefix (OP:/User:) is removed."""
    history = original.rsplit(" [SEP] ", 1)[0] if " [SEP] " in original else ""
    tags = list(ITEM_TAG.finditer(history))
    items = []
    for tag, following in zip(tags, tags[1:] + [None]):
        text = history[tag.end():following.start() if following else len(history)]
        items.append((tag.group(1), SPEAKER.sub("", text)))
    return items


def comments(tree: Optional[list]) -> Iterator[dict]:
    for comment in tree or []:
        yield comment
        yield from comments(comment.get("children"))


def item_times(item: dict) -> Times:
    edited = item.get("edited")
    edited_utc = None if edited in (False, None, True) else int(float(edited))
    return int(float(item["created_utc"])), edited_utc


def index_threads(path: Path, authors: set) -> Dict[str, List[Times]]:
    """Key -> times of every post by the authors and every comment by them or in a thread they started."""
    index: Dict[str, List[Times]] = defaultdict(list)
    with bz2.open(path, "rt") as handle:
        for line in handle:
            thread = json.loads(line)
            op = thread.get("author")
            if op in authors:
                body = thread.get("selftext")
                text = (thread.get("title") or "") + "\n\n" + ("" if body in MISSING else body)
                index[text_key(text)].append(item_times(thread))
            for comment in comments(thread.get("comments")):
                if comment.get("body") not in MISSING and (op in authors or comment.get("author") in authors):
                    index[text_key(comment["body"])].append(item_times(comment))
    return index


def classify(times: List[Times], post_utc: int) -> str:
    if not times:
        return "unmatched"
    if all(created > post_utc for created, _ in times):
        return "after"
    if all(created > post_utc or (edited is not None and edited > post_utc) for created, edited in times):
        return "edited_after"
    return "before"


def check(rows, index: Dict[str, List[Times]]) -> dict:
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    rows_with_after = Counter()
    ordered = Counter()
    for row in rows:
        statuses, first_times = [], []
        for kind, text in history_items(row["original"]):
            times = index.get(text_key(text), [])
            status = classify(times, row["created_utc"])
            by_kind[kind][status] += 1
            statuses.append(status)
            if times:
                first_times.append(min(created for created, _ in times))
        rows_with_after["rows"] += 1
        rows_with_after["with_after"] += any(s == "after" for s in statuses)
        rows_with_after["with_after_or_edited_after"] += any(s in ("after", "edited_after") for s in statuses)
        if len(first_times) > 1:
            ordered["rows"] += 1
            ordered["chronological"] += first_times == sorted(first_times)
            ordered["last_item_is_newest"] += first_times[-1] == max(first_times)
            newest = set(sorted(first_times)[-5:])  # the five newest items: are they the last five?
            ordered["last_5_are_5_newest"] += set(first_times[-5:]) == newest
    return {"items": {kind: dict(counts) for kind, counts in by_kind.items()}, "rows": dict(rows_with_after),
            "order_of_matched_items": dict(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("threads", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {config: load_dataset(HUB_REPO, config) for config in CONFIGS}
    authors = {author for splits in data.values() for split in splits.values() for author in split["author"]}
    index = index_threads(args.threads, authors)
    print(f"{len(authors)} OPs, {len(index)} distinct text keys")
    results = {config: {split: check(rows, index) for split, rows in splits.items()} for config, splits in data.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
