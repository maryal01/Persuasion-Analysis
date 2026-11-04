"""T5 (CMV malleability): build the post-level dataset with author LIWC/Receptiviti features and OP-grouped splits.

One row per r/changemyview post whose OP has a record in author_liwc.jsonl (the paper's malleability
sample, Al-Khatib et al. 2020). Splits are grouped by OP (all posts of one author land in the same split).
Label = 1 if DeltaBot replied "Confirmed: ... delta awarded" to a comment written by the OP. The post's `delta` field (from the post flair, missing
before 2015-03 and replaced by other flairs in some threads) is NOT used as the label; it is stored as `label_flair` for comparison only.

Usage (from the repository root, stdlib only):
    python scripts/t5/build_dataset.py \
        --posts <cmv_dir>/posts_malleability.jsonl.bz2 \
        --liwc <cmv_dir>/author_liwc.jsonl.bz2 \
        --output-dir <out_dir>/t5_dataset
<cmv_dir> holds the .jsonl.bz2 files of Webis-CMV-20 (https://zenodo.org/records/3778298), <out_dir> is any folder outside the repository
Writes train/validation/test .jsonl, feature_names.json and stats.json to --output-dir.
"""
import argparse
import bz2
import datetime
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SEED = 42  # same value and rule as split_of in create_dataset.py (T1/T2), keyed by OP here
TEST_SIZE = 0.1
VAL_SIZE = 0.1
Row = Dict[str, Any]  # e.g. {"post_id": "3d3hj8", "author": "name", "label": 0, "text": "...", "features": [0.1, ...], "split": "train"}
DELTA_MARKER = re.compile(r"[∆Δ]|&amp;#8710;|!delta|\bdelta\b", re.IGNORECASE)


def walk(comments: Optional[List[Dict[str, Any]]], parent: Optional[Dict[str, Any]] = None) -> Iterator[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Every comment of a thread (depth first) with its parent comment (None for top-level comments)."""
    for comment in comments or []:
        yield comment, parent
        yield from walk(comment.get("children"), comment)


def deltabot_confirmed(post: Dict[str, Any]) -> bool:
    """True if DeltaBot confirmed a delta in a reply to a comment written by the OP ("Confirmed: 1 delta awarded to ...")."""
    return any(c.get("author") == "DeltaBot" and (c.get("body") or "").startswith("Confirmed")
               and parent is not None and parent.get("author") == post.get("author") for c, parent in walk(post.get("comments")))


def split_of(group_id: str) -> str:
    """Deterministic split of a group (here: the OP's username); same rule as create_dataset.split_of."""
    h = int(hashlib.md5(f"{SEED}:{group_id}".encode()).hexdigest(), 16) % 1000 / 1000
    if h < TEST_SIZE:
        return "test"
    if h < TEST_SIZE + VAL_SIZE:
        return "validation"
    return "train"


def liwc_vector(traits: Dict[str, Any]) -> Dict[str, float]:
    """Flatten one author_liwc record into {feature_name: value}: LIWC scalars and categories, Receptiviti raw scores and percentiles."""
    liwc = traits["liwc_scores"]
    features = {f"liwc_{k}": float(v) for k, v in liwc.items() if k != "categories"}
    features.update({f"liwc_cat_{k}": float(v) for k, v in liwc["categories"].items()})
    receptiviti = traits["receptiviti_scores"]
    features.update({f"recv_raw_{k}": float(v) for k, v in receptiviti["raw_scores"].items()})
    features.update({f"recv_pct_{k}": float(v) for k, v in receptiviti["percentiles"].items()})
    return features


def read_liwc(path: Path) -> Dict[str, Dict[str, float]]:
    """author -> flattened feature dict, for every record in author_liwc.jsonl.bz2."""
    by_author: Dict[str, Dict[str, float]] = {}
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_author[record["author"]] = liwc_vector(record["traits"])
    return by_author


def build_rows(posts_path: Path, liwc_by_author: Dict[str, Dict[str, float]]) -> Tuple[List[Row], List[str]]:
    """Rows for all posts whose OP has LIWC features, and the sorted feature names shared by all authors."""
    names = sorted(set.intersection(*(set(f) for f in liwc_by_author.values())))
    rows: List[Row] = []
    with bz2.open(posts_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            post = json.loads(line)
            author: Optional[str] = post.get("author")
            if author not in liwc_by_author:
                continue
            features = liwc_by_author[author]
            rows.append({
                "post_id": post["id"].removeprefix("t3_"), "author": author, "created_utc": int(post["created_utc"]),
                "label": 0,  # = label_deltabot, set below
                "label_flair": int(bool(post["delta"])), "label_deltabot": int(deltabot_confirmed(post)),
                "text": ((post.get("title") or "") + "\n\n" + (post.get("selftext") or "")).strip(),
                "features": [features[n] for n in names], "split": split_of(author),
            })
            rows[-1]["label"] = rows[-1]["label_deltabot"]
    return rows, names


def summarize(rows: List[Row], names: List[str], n_authors_liwc: int, n_missing_feature_authors: int) -> Dict[str, Any]:
    """Counts per split, label balance, OP overlap between splits and delta markers in the post text."""
    stats: Dict[str, Any] = {"n_posts": len(rows), "n_features": len(names), "n_liwc_authors": n_authors_liwc,
                             "liwc_authors_with_missing_features": n_missing_feature_authors}
    authors_of: Dict[str, set] = {}
    for split in ("train", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        authors_of[split] = {r["author"] for r in part}
        stats[split] = {"posts": len(part), "ops": len(authors_of[split]), "delta": sum(r["label"] for r in part),
                        "delta_rate": round(sum(r["label"] for r in part) / max(len(part), 1), 3)}
    stats["ops_shared_between_splits"] = {"train&validation": len(authors_of["train"] & authors_of["validation"]),
                                          "train&test": len(authors_of["train"] & authors_of["test"]),
                                          "validation&test": len(authors_of["validation"] & authors_of["test"])}
    stats["label_agreement_flair_vs_deltabot"] = {f"flair={f},deltabot={d}": sum(1 for r in rows if (r["label_flair"], r["label_deltabot"]) == (f, d)) for f in (0, 1) for d in (0, 1)}
    by_year: Dict[int, List[int]] = {}
    for r in rows:
        y = datetime.datetime.fromtimestamp(r["created_utc"], datetime.timezone.utc).year
        by_year.setdefault(y, [0, 0, 0])
        by_year[y][0] += 1; by_year[y][1] += r["label_flair"]; by_year[y][2] += r["label_deltabot"]
    stats["rate_by_year_flair_deltabot"] = {y: [v[0], round(v[1] / v[0], 3), round(v[2] / v[0], 3)] for y, v in sorted(by_year.items())}
    marked = Counter((r["label"], bool(DELTA_MARKER.search(r["text"]))) for r in rows)
    stats["post_text_with_delta_marker"] = {"label0": marked[(0, True)] / max(marked[(0, True)] + marked[(0, False)], 1),
                                            "label1": marked[(1, True)] / max(marked[(1, True)] + marked[(1, False)], 1)}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--liwc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    liwc_by_author = read_liwc(args.liwc)
    rows, names = build_rows(args.posts, liwc_by_author)
    n_missing = sum(1 for f in liwc_by_author.values() if len(f) != len(names))
    stats = summarize(rows, names, len(liwc_by_author), n_missing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with open(args.output_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                if row["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "feature_names.json").write_text(json.dumps(names, indent=1))
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
