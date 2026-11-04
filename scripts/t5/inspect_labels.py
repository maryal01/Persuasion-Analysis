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
Comment = Dict[str, Any]  # a comment dict of the dataset: author, body, children, ...
Record = Dict[str, Any]  # e.g. {"category": "false_op_symbol", "id": "3d3hj8", "flair": None, "delta": False, "op_symbol_comments": [...]}


def walk(comments: Optional[List[Comment]], parent: Optional[Comment] = None) -> Iterator[Tuple[Comment, Optional[Comment]]]:
    """Every comment of a thread (depth first) with its parent comment (None for top-level comments)."""
    for comment in comments or []:
        yield comment, parent
        yield from walk(comment.get("children"), comment)


def shorten(text: Optional[str], limit: int) -> str:
    """First `limit` characters of a text on a single line."""
    return re.sub(r"\s+", " ", text or "")[:limit]


def build_record(post: Dict[str, Any], category: str) -> Record:
    """Compact description of one post and the OP comments that carry a delta symbol."""
    op_comments = []
    for comment, parent in walk(post.get("comments")):
        if comment.get("author") == post.get("author") and SYMBOL.search(comment.get("body") or ""):
            op_comments.append({"body": shorten(comment.get("body"), 400), "parent_author": parent.get("author") if parent else "(the post)",
                                "parent_body": shorten(parent.get("body"), 300) if parent else shorten(post.get("selftext"), 300),
                                "replies": [{"author": c.get("author"), "body": shorten(c.get("body"), 300)} for c in comment.get("children") or []]})
    return {"category": category, "id": post["id"], "title": post.get("title"), "created": datetime.datetime.fromtimestamp(int(post["created_utc"]), datetime.timezone.utc).strftime("%Y-%m-%d"),
            "op": post.get("author"), "flair": post.get("link_flair_text"), "delta": bool(post.get("delta")),
            "subtree_delta": any(s.get("delta") for s in post.get("subtrees") or []), "op_symbol_comments": op_comments}


def collect(posts_path: Path, output: Path, min_date: str, sample_size: int, seed: int) -> None:
    """One pass over the posts file; write all false_op_symbol posts and reservoir samples of the two other categories."""
    start = datetime.datetime.strptime(min_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    rng = random.Random(seed)
    records: List[Record] = []
    reservoirs: Dict[str, List[Record]] = {"true_sample": [], "false_clean": []}
    seen = {"true_sample": 0, "false_clean": 0}
    counts = {"posts": 0, "delta_true": 0, "delta_false": 0, "false_op_symbol": 0}
    with bz2.open(posts_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            post = json.loads(line)
            if int(post["created_utc"]) < start:
                continue
            counts["posts"] += 1
            counts["delta_true" if post.get("delta") else "delta_false"] += 1
            has_symbol = any(c.get("author") == post.get("author") and SYMBOL.search(c.get("body") or "") for c, _ in walk(post.get("comments")))
            if not post.get("delta") and has_symbol:
                records.append(build_record(post, "false_op_symbol")); counts["false_op_symbol"] += 1
                continue
            category = "true_sample" if post.get("delta") else "false_clean"
            seen[category] += 1
            if len(reservoirs[category]) < sample_size:
                reservoirs[category].append(build_record(post, category))
            elif (j := rng.randrange(seen[category])) < sample_size:
                reservoirs[category][j] = build_record(post, category)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records + reservoirs["true_sample"] + reservoirs["false_clean"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"posts from {min_date}: {counts['posts']} (delta=True {counts['delta_true']}, delta=False {counts['delta_false']}); delta=False with an OP delta symbol: {counts['false_op_symbol']}; wrote {output}")


def show(input_path: Path, category: str, n: int, seed: int) -> None:
    """Print n random records of a category."""
    records = [r for r in map(json.loads, open(input_path, encoding="utf-8")) if r["category"] == category]
    for r in random.Random(seed).sample(records, min(n, len(records))):
        print(f"\n=== {r['id']} {r['created']} | delta={r['delta']} subtree_delta={r['subtree_delta']} flair={r['flair']!r}\n    {r['title']}")
        for c in r["op_symbol_comments"][:3]:
            print(f"  OP wrote: {c['body']}\n    in reply to {c['parent_author']}: {c['parent_body']}")
            for reply in c["replies"][:3]:
                print(f"    -> {reply['author']}: {reply['body']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--posts", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--min-date", default="2015-05-01")
    c.add_argument("--sample-size", type=int, default=60)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("show")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--category", choices=["false_op_symbol", "true_sample", "false_clean"], required=True)
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.posts, args.output, args.min_date, args.sample_size, args.seed)
    else:
        show(args.input, args.category, args.n, args.seed)


if __name__ == "__main__":
    main()
