"""Read threads of Webis-CMV-20 `posts_malleability` to check what the `delta` label means.

`collect` streams the file once and writes a JSONL with, for posts created on or after --min-date:
  false_op_symbol  posts with delta=False where the OP wrote a delta symbol (∆, Δ, !delta) in a comment
  true_sample      a random sample of posts with delta=True
  false_clean      a random sample of posts with delta=False and no such OP comment
Each record has the flair, the subtree-delta flag, and for every OP comment with the symbol: its text, its parent comment and the replies to it
(DeltaBot answers to a delta comment are among them).
`show` prints random records of one category in a readable form.

Usage (from the repository root, stdlib only):
    python scripts/t5/inspect_labels.py collect --posts <cmv_dir>/posts_malleability.jsonl.bz2 \
        --output <out_dir>/label_samples/candidates.jsonl
    python scripts/t5/inspect_labels.py show --input .../candidates.jsonl --category false_op_symbol --n 10 --seed 1
"""

import argparse
import bz2
import datetime
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SYMBOL = re.compile(r"[∆Δ]|&amp;#8710;|!delta", re.IGNORECASE)
CATEGORIES = ("false_op_symbol", "true_sample", "false_clean")
Post = Dict[str, Any]     # one record of posts_malleability.jsonl
Comment = Dict[str, Any]  # a comment of a post: author, body, children, ...
Record = Dict[str, Any]   # {"category": "false_op_symbol", "id": "t3_3d3hj8", "flair": None, "delta": False, "op_symbol_comments": [...], ...}


def walk(comments: Optional[List[Comment]], parent: Optional[Comment] = None) -> Iterator[Tuple[Comment, Optional[Comment]]]:
    """Every comment of a thread (depth first) with its parent comment (None for top-level comments)."""
    for comment in comments or []:
        yield comment, parent
        yield from walk(comment.get("children"), comment)


def shorten(text: Optional[str], limit: int) -> str:
    """First `limit` characters of a text on a single line."""
    return re.sub(r"\s+", " ", text or "")[:limit]


def op_delta_comments(post: Post) -> List[Tuple[Comment, Optional[Comment]]]:
    """The comments written by the OP that contain a delta symbol, each with its parent comment."""
    return [(comment, parent) for comment, parent in walk(post.get("comments"))
            if comment.get("author") == post.get("author") and SYMBOL.search(comment.get("body") or "")]


def describe_comment(post: Post, comment: Comment, parent: Optional[Comment]) -> Dict[str, Any]:
    """An OP comment with a delta symbol, the comment it answers and the replies to it (DeltaBot's answer is among them)."""
    parent_author, parent_text = (parent.get("author"), parent.get("body")) if parent else ("(the post)", post.get("selftext"))
    replies = [{"author": reply.get("author"), "body": shorten(reply.get("body"), 300)} for reply in comment.get("children") or []]
    return {"body": shorten(comment.get("body"), 400), "parent_author": parent_author, "parent_body": shorten(parent_text, 300), "replies": replies}


def build_record(post: Post, category: str) -> Record:
    """Compact description of one post and of the OP comments that carry a delta symbol."""
    created = datetime.datetime.fromtimestamp(int(post["created_utc"]), datetime.timezone.utc).strftime("%Y-%m-%d")
    return {"category": category, "id": post["id"], "title": post.get("title"), "created": created, "op": post.get("author"),
            "flair": post.get("link_flair_text"), "delta": bool(post.get("delta")),
            "subtree_delta": any(subtree.get("delta") for subtree in post.get("subtrees") or []),
            "op_symbol_comments": [describe_comment(post, comment, parent) for comment, parent in op_delta_comments(post)]}


def category_of(post: Post) -> str:
    """false_op_symbol: delta=False although the OP wrote a delta symbol; true_sample: delta=True; false_clean: delta=False, no symbol."""
    if post.get("delta"):
        return "true_sample"
    return "false_op_symbol" if op_delta_comments(post) else "false_clean"


class Reservoir:
    """A random sample of fixed size from a stream of unknown length (each item has the same chance to be kept)."""

    def __init__(self, size: int, rng: random.Random):
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def add(self, make_item) -> None:
        """Offer an item; `make_item` is only called if the item is kept."""
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(make_item())
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.size:
            self.items[slot] = make_item()


def read_posts(path: Path, since: float) -> Iterator[Post]:
    """The posts of posts_malleability.jsonl.bz2 created at or after the Unix time `since`."""
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            post = json.loads(line)
            if int(post["created_utc"]) >= since:
                yield post


def collect(posts_path: Path, output: Path, min_date: str, sample_size: int, seed: int) -> None:
    """One pass over the posts file; write all false_op_symbol posts and random samples of the two other categories."""
    since = datetime.datetime.strptime(min_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    rng = random.Random(seed)
    suspicious: List[Record] = []
    samples = {"true_sample": Reservoir(sample_size, rng), "false_clean": Reservoir(sample_size, rng)}
    n_posts = n_delta = 0
    for post in read_posts(posts_path, since):
        n_posts += 1
        n_delta += bool(post.get("delta"))
        category = category_of(post)
        if category == "false_op_symbol":
            suspicious.append(build_record(post, category))
        else:
            samples[category].add(lambda post=post, category=category: build_record(post, category))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in suspicious + samples["true_sample"].items + samples["false_clean"].items:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"posts from {min_date}: {n_posts} (delta=True {n_delta}, delta=False {n_posts - n_delta}); "
          f"delta=False with an OP delta symbol: {len(suspicious)}; wrote {output}")


def show(input_path: Path, category: str, n: int, seed: int) -> None:
    """Print n random records of a category."""
    records = [record for record in map(json.loads, open(input_path, encoding="utf-8")) if record["category"] == category]
    for record in random.Random(seed).sample(records, min(n, len(records))):
        print(f"\n=== {record['id']} {record['created']} | delta={record['delta']} subtree_delta={record['subtree_delta']} "
              f"flair={record['flair']!r}\n    {record['title']}")
        for comment in record["op_symbol_comments"][:3]:
            print(f"  OP wrote: {comment['body']}\n    in reply to {comment['parent_author']}: {comment['parent_body']}")
            for reply in comment["replies"][:3]:
                print(f"    -> {reply['author']}: {reply['body']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--posts", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--min-date", default="2015-05-01")
    collect_parser.add_argument("--sample-size", type=int, default=60)
    collect_parser.add_argument("--seed", type=int, default=0)
    show_parser = commands.add_parser("show")
    show_parser.add_argument("--input", type=Path, required=True)
    show_parser.add_argument("--category", choices=CATEGORIES, required=True)
    show_parser.add_argument("--n", type=int, default=10)
    show_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.posts, args.output, args.min_date, args.sample_size, args.seed)
    else:
        show(args.input, args.category, args.n, args.seed)


if __name__ == "__main__":
    main()
