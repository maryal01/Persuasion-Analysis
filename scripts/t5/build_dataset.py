"""T5 (CMV malleability): build the post-level dataset with the author's LIWC/Receptiviti features and publish it on the Hub.

One row per r/changemyview post whose OP has a record in author_liwc.jsonl (the sample of Al-Khatib et al. 2020).
Label = 1 if DeltaBot replied "Confirmed: ... delta awarded" to a comment written by the OP. The `delta` field of the post
comes from the post flair (missing before 2015-03, replaced by other flairs in some threads) and is NOT the label; it is
stored as `label_flair` for comparison only. Posts are split by OP 80/10/10, so one author is never in two splits.
The text is stored as in the corpus; scripts/t5/cleaning.py cleans it when the models read it.

Usage (from the repository root; needs the `datasets` package and a Hugging Face login with write access for --push):
    python scripts/t5/build_dataset.py \
        --posts <cmv_dir>/posts_malleability.jsonl.bz2 --liwc <cmv_dir>/author_liwc.jsonl.bz2 \
        [--push] [--output-dir <dir>]
<cmv_dir> holds the .jsonl.bz2 files of Webis-CMV-20 (https://zenodo.org/records/3778298).
--push uploads the config `cmv_malleability_liwc` to the Hub branch `t5` (created if missing; the scripts read the copy on `main`); --output-dir also writes
train/validation/test.jsonl, feature_names.json and stats.json.
"""
import argparse
import bz2
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from t5_data import HUB_CONFIG, HUB_REPO, SPLIT_NAMES

STAGING_REVISION = "t5"  # Hub branch that --push uploads to; copy the config to main afterwards (as for T2 and T3)
SEED = 42  # same value and rule as split_of in create_dataset.py (T1/T2), applied to the OP here
TEST_SIZE = 0.1
VAL_SIZE = 0.1

Post = Dict[str, Any]     # one record of posts_malleability.jsonl
Comment = Dict[str, Any]  # a comment of a post, with its replies in "children"
Row = Dict[str, Any]      # {"post_id", "author", "created_utc", "label", "label_flair", "label_deltabot", "text", "features", "split"}
Features = Dict[str, float]  # {"liwc_wc": 8437.0, "recv_raw_openness": 0.93, ...}


def read_jsonl_bz2(path: Path) -> Iterator[Dict[str, Any]]:
    """The records of a .jsonl.bz2 file, one at a time."""
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


# --- label --------------------------------------------------------------------------------------------------------------

def walk(comments: Optional[List[Comment]], parent: Optional[Comment] = None) -> Iterator[Tuple[Comment, Optional[Comment]]]:
    """Every comment of a thread, depth first, with its parent comment (None for top-level comments)."""
    for comment in comments or []:
        yield comment, parent
        yield from walk(comment.get("children"), comment)


def is_delta_confirmation(comment: Comment, parent: Optional[Comment], op: Optional[str]) -> bool:
    """True for a DeltaBot reply "Confirmed: 1 delta awarded ..." to a comment written by the OP."""
    return (comment.get("author") == "DeltaBot" and (comment.get("body") or "").startswith("Confirmed")
            and parent is not None and parent.get("author") == op)


def deltabot_confirmed(post: Post) -> bool:
    """True if DeltaBot confirmed at least one delta that the OP of the post awarded."""
    op = post.get("author")
    return any(is_delta_confirmation(comment, parent, op) for comment, parent in walk(post.get("comments")))


# --- author features ----------------------------------------------------------------------------------------------------

def flatten_traits(traits: Dict[str, Any]) -> Features:
    """One author_liwc record as {feature name: value}: the LIWC scores, the LIWC categories and the Receptiviti raw scores and percentiles."""
    liwc, receptiviti = traits["liwc_scores"], traits["receptiviti_scores"]
    features = {f"liwc_{name}": float(value) for name, value in liwc.items() if name != "categories"}
    features.update({f"liwc_cat_{name}": float(value) for name, value in liwc["categories"].items()})
    features.update({f"recv_raw_{name}": float(value) for name, value in receptiviti["raw_scores"].items()})
    features.update({f"recv_pct_{name}": float(value) for name, value in receptiviti["percentiles"].items()})
    return features


def read_author_features(path: Path) -> Dict[str, Features]:
    """author -> features, for every author of author_liwc.jsonl.bz2."""
    return {record["author"]: flatten_traits(record["traits"]) for record in read_jsonl_bz2(path)}


def common_feature_names(features_by_author: Dict[str, Features]) -> List[str]:
    """Sorted names of the features that every author has."""
    return sorted(set.intersection(*(set(features) for features in features_by_author.values())))


# --- rows ---------------------------------------------------------------------------------------------------------------

def split_of(author: str) -> str:
    """"train", "validation" or "test" for an author, decided by a hash so that all posts of an author go to the same split."""
    fraction = int(hashlib.md5(f"{SEED}:{author}".encode()).hexdigest(), 16) % 1000 / 1000
    if fraction < TEST_SIZE:
        return "test"
    if fraction < TEST_SIZE + VAL_SIZE:
        return "validation"
    return "train"


def make_row(post: Post, features: Features, feature_names: List[str]) -> Row:
    """The row of one post."""
    deltabot = int(deltabot_confirmed(post))
    return {
        "post_id": post["id"].removeprefix("t3_"),
        "author": post["author"],
        "created_utc": int(post["created_utc"]),
        "label": deltabot,
        "label_flair": int(bool(post["delta"])),
        "label_deltabot": deltabot,
        "text": ((post.get("title") or "") + "\n\n" + (post.get("selftext") or "")).strip(),
        "features": [features[name] for name in feature_names],
        "split": split_of(post["author"]),
    }


def build_rows(posts_path: Path, features_by_author: Dict[str, Features], feature_names: List[str]) -> List[Row]:
    """The rows of all posts whose OP has author features."""
    return [make_row(post, features_by_author[post["author"]], feature_names)
            for post in read_jsonl_bz2(posts_path) if post.get("author") in features_by_author]


# --- statistics ---------------------------------------------------------------------------------------------------------

def split_statistics(rows: List[Row]) -> Dict[str, Any]:
    """Posts, authors and share of delta posts per split, and the authors that appear in more than one split."""
    stats: Dict[str, Any] = {}
    authors: Dict[str, set] = {}
    for name in SPLIT_NAMES:
        part = [row for row in rows if row["split"] == name]
        authors[name] = {row["author"] for row in part}
        delta = sum(row["label"] for row in part)
        stats[name] = {"posts": len(part), "ops": len(authors[name]), "delta": delta, "delta_rate": round(delta / max(len(part), 1), 3)}
    stats["ops_shared_between_splits"] = {f"{a}&{b}": len(authors[a] & authors[b])
                                          for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return stats


def label_statistics(rows: List[Row]) -> Dict[str, Any]:
    """How often the flair label and the DeltaBot label agree, and the share of delta posts per year under each."""
    agreement = {f"flair={flair},deltabot={bot}": sum(1 for row in rows if (row["label_flair"], row["label_deltabot"]) == (flair, bot))
                 for flair in (0, 1) for bot in (0, 1)}
    years = sorted({datetime.datetime.fromtimestamp(row["created_utc"], datetime.timezone.utc).year for row in rows})
    by_year = {}
    for year in years:
        posts = [row for row in rows if datetime.datetime.fromtimestamp(row["created_utc"], datetime.timezone.utc).year == year]
        by_year[year] = [len(posts), round(sum(row["label_flair"] for row in posts) / len(posts), 3),
                         round(sum(row["label_deltabot"] for row in posts) / len(posts), 3)]
    return {"label_agreement_flair_vs_deltabot": agreement, "posts_flair_rate_deltabot_rate_by_year": by_year}


def statistics(rows: List[Row], feature_names: List[str]) -> Dict[str, Any]:
    return {"n_posts": len(rows), "n_features": len(feature_names), **split_statistics(rows), **label_statistics(rows)}


# --- output -------------------------------------------------------------------------------------------------------------

def to_dataset_dict(rows: List[Row]) -> DatasetDict:
    """The rows as a Hugging Face DatasetDict with one table per split."""
    tables = {}
    for name in SPLIT_NAMES:
        part = [{key: value for key, value in row.items() if key != "split"} for row in rows if row["split"] == name]
        tables[name] = Dataset.from_dict({key: [row[key] for row in part] for key in part[0]})
    return DatasetDict(tables)


def push_to_hub(dataset: DatasetDict, feature_names: List[str], revision: str) -> None:
    """Upload the config and the feature names to a branch of the Hub dataset (the branch is created if it does not exist)."""
    dataset.push_to_hub(HUB_REPO, config_name=HUB_CONFIG, private=True, revision=revision)
    HfApi().upload_file(path_or_fileobj=json.dumps(feature_names, indent=1).encode(), path_in_repo=f"{HUB_CONFIG}/feature_names.json",
                        repo_id=HUB_REPO, repo_type="dataset", revision=revision, commit_message=f"Add the feature names of {HUB_CONFIG}")
    print(f"pushed -> https://huggingface.co/datasets/{HUB_REPO}/tree/{revision} (config {HUB_CONFIG})")


def write_local_copy(rows: List[Row], feature_names: List[str], stats: Dict[str, Any], output_dir: Path) -> None:
    """train/validation/test.jsonl, feature_names.json and stats.json in a local folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        with open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row["split"] == name)
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=1))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--posts", type=Path, required=True)
    parser.add_argument("--liwc", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help=f"upload the config {HUB_CONFIG} to the Hub")
    parser.add_argument("--hub-revision", default=STAGING_REVISION, help="Hub branch to upload to (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None, help="also write a local copy here")
    args = parser.parse_args()

    features_by_author = read_author_features(args.liwc)
    feature_names = common_feature_names(features_by_author)
    rows = build_rows(args.posts, features_by_author, feature_names)
    stats = statistics(rows, feature_names)
    print(json.dumps(stats, indent=1))

    if args.output_dir:
        write_local_copy(rows, feature_names, stats, args.output_dir)
    if args.push:
        push_to_hub(to_dataset_dict(rows), feature_names, args.hub_revision)


if __name__ == "__main__":
    main()
