import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import ijson
from datasets import ClassLabel, Dataset, DatasetDict
from huggingface_hub import login

REPO = "nlp-final-project11/persuasion-datasets"
REVISION = "t1-t2"  # Hub branch to push to; main is the team's shared copy
TEST_SIZE = 0.1
VAL_SIZE = 0.1
SEED = 42

BEFORE = "Agreed with before the debate"
AFTER = "Agreed with after the debate"
OTHER_CRITERIA = (
    "Who had better conduct",
    "Had better spelling and grammar",
    "Made more convincing arguments",
    "Used the most reliable sources",
)

DELTA_SYMBOL = re.compile(r"[∆Δ]|&(?:amp;)?#8710;|!delta", re.IGNORECASE)
DELTA_MENTION = re.compile(r"[∆Δ]|&(?:amp;)?#8710;|\bdelta", re.IGNORECASE)
MISSING = (None, "", "[deleted]", "[removed]")
BOTS = ("DeltaBot", "AutoModerator")
# a line that starts with "Edit:", "EDIT2 -", "**Update**", "Further edit:", ... up to the end of its paragraph
EDIT_NOTE = re.compile(r"(?ims)^(?:&gt;|[ \t>*_#(\[\-])*(?:\w+ )?(?:edit(?:ed|s)?|update[ds]?|eta)\s*#?\d*\b\s*[:\-–—*)\]].*?(?=\n\s*\n|\Z)")
MOD_FOOTNOTE = re.compile(r"(?is)[_\s]*(?:&gt;|>)?\s*\*?\s*(?:Hello, [^\n]{0,40}?[.!]\s*)?This is a footnote from (?:the |your )(?:CMV )?moderators.*\Z")

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
    whether its delta row outscores its no-delta row. Pairs whose reply was written by the OP
    (21 no-delta branches) are dropped, since OPs don't award themselves deltas.
    """
    rows = []
    with open(path) as fh:
        for pair_id, line in enumerate(fh):
            r = json.loads(line)
            s = r["submission"]
            if s.get("author") in (r["delta_comment"]["author"], r["nodelta_comment"]["author"]):
                continue
            original = (s.get("title", "") + "\n\n" + s.get("selftext", "")).strip()
            split = split_of(r["submission_id"])
            for key, label in (("delta_comment", 1), ("nodelta_comment", 0)):
                reply = "\n\n".join(c.get("body", "") for c in r[key]["comments"]).strip()
                rows.append({"original": original, "reply": reply, "label": label,
                             "pair_id": pair_id, "post_id": r["submission_id"], "split": split})
    return {"cmv": to_splits(rows, ["nodelta", "delta"])}

def debate_transcript(d):
    """Rounds as 'Round i' blocks of 'Pro: ...' / 'Con: ...' turns, dropping empty turns and the
    placeholder text 'forfeit' that DDO stores for a skipped turn.
    """
    rounds = []
    for i, rnd in enumerate(d.get("rounds", []), 1):
        turns = [f"{t['side']}: {t['text'].strip()}" for t in rnd
                 if (t.get("text") or "").strip().lower() not in ("", "forfeit")]
        if turns:
            rounds.append(f"Round {i}\n" + "\n\n".join(turns))
    return "\n\n".join(rounds)

def stance(votes_map, positions, field):
    """Side ('Pro', 'Con' or 'Tie') a voter marked for `field`, or None if not exactly one is marked."""
    sides = [("Tie" if name == "Tied" else positions.get(name))
             for name, marks in votes_map.items() if marks.get(field) is True]
    return sides[0] if len(sides) == 1 else None

def build_ddo_votes(path):
    """T2. debates.json -> one row per audience vote, in two configs:

    ddo_switch    voters with a side before and after the debate; label stayed / switched
    ddo_undecided voters tied before the debate; label tie / pro / con after it

    Side-to-tie votes are dropped, matching the proposal's ~66k decided votes. So are votes by the
    debaters themselves, repeat votes, and votes without before/after stances. `filled_other_criteria`
    flags voters who marked a debater on another criterion, since a pre-debate tie can mean "skipped".
    """
    switch, undecided = [], []
    with open(path, "rb") as fh:
        for debate_id, d in ijson.kvitems(fh, ""):
            positions = {d.get("participant_1_name"): d.get("participant_1_position"),
                         d.get("participant_2_name"): d.get("participant_2_position")}
            title = (d.get("title", "") or "").strip()
            year = int(d["start_date"].split("/")[-1]) if d.get("start_date") else None
            transcript = None
            seen = set()
            for v in d.get("votes", []):
                voter = v.get("user_name")
                if voter in positions or voter in seen:
                    continue
                seen.add(voter)
                votes_map = v.get("votes_map") or {}
                before = stance(votes_map, positions, BEFORE)
                after = stance(votes_map, positions, AFTER)
                if before is None or after is None or (before != "Tie" and after == "Tie"):
                    continue
                if transcript is None:
                    transcript = debate_transcript(d)
                if not transcript:
                    break
                row = {
                    "original": f"Resolution: {title}\nVoter's side before the debate: {before}",
                    "reply": transcript,
                    "debate_id": debate_id, "voter": voter, "before": before, "after": after, "year": year,
                    "filled_other_criteria": any(marks.get(c) is True for name, marks in votes_map.items()
                                                 if name != "Tied" for c in OTHER_CRITERIA),
                    "split": split_of(debate_id),
                }
                if before == "Tie":
                    undecided.append({**row, "label": {"Tie": 0, "Pro": 1, "Con": 2}[after]})
                else:
                    switch.append({**row, "label": int(after != before)})
    return {"ddo_switch": to_splits(switch, ["stayed", "switched"]),
            "ddo_undecided": to_splits(undecided, ["tie", "pro", "con"])}

def cmv_comments(comments, parent=None):
    """Every comment of a CMV thread with its parent comment (None at the top level)."""
    for c in comments or []:
        yield c, parent
        yield from cmv_comments(c.get("children"), c)

def cmv_clean(text):
    """CMV text without "Edit:" notes (written after the discussion started) and without the
    paragraphs that mention a delta, either of which can give the outcome away.
    """
    paragraphs = (p.strip() for p in re.split(r"\n\s*\n", EDIT_NOTE.sub("", text)))
    return "\n\n".join(p for p in paragraphs if p and not DELTA_MENTION.search(p))

def cmv_thread_rows(path):
    """The cmv_thread rows of threads.jsonl (see build_cmv_thread), one post at a time."""
    with open(path) as fh:
        for line in fh:
            s = json.loads(line)
            op = s.get("author")
            if op in MISSING or s.get("selftext") == "[removed]" or (s.get("link_flair_text") or "").lower().startswith("removed"):
                continue
            comments = list(cmv_comments(s.get("comments")))
            answered = [(c, parent) for c, parent in comments
                        if c.get("author") == "DeltaBot" and parent is not None and parent.get("author") == op]
            delta = any((c.get("body") or "").startswith("Confirmed") for c, _ in answered)
            # the discussion stops at the OP's first award attempt: a comment of theirs with a delta
            # symbol or an answer from DeltaBot (confirmed or rejected)
            attempts = [parent for _, parent in answered] + [c for c, _ in comments
                                                             if c.get("author") == op and DELTA_SYMBOL.search(c.get("body") or "")]
            cutoff = min((float(c["created_utc"]) for c in attempts), default=float("inf"))
            users, turns = {}, []
            for c, _ in sorted(comments, key=lambda pair: float(pair[0]["created_utc"])):
                author = c.get("author")
                if author in BOTS or c.get("body") in MISSING or float(c["created_utc"]) >= cutoff:
                    continue
                text = cmv_clean(c["body"])
                if text:
                    name = ("OP" if author == op else "Deleted user" if author in MISSING
                            else f"User {users.setdefault(author, len(users) + 1)}")
                    turns.append(f"{name}: {text}")
            if not turns:
                continue  # nothing was said before the outcome
            body = "" if s.get("selftext") in MISSING else MOD_FOOTNOTE.sub("", s["selftext"])
            yield {"original": ((s.get("title") or "") + "\n\n" + cmv_clean(body)).strip(), "reply": "\n\n".join(turns),
                   "label": int(delta), "post_id": s["id"], "author": op,
                   "created_utc": int(float(s["created_utc"])), "n_comments": len(turns), "split": split_of(s["id"])}

def build_cmv_thread(path):
    """CMV thread outcome. Webis-CMV-20 threads.jsonl -> cmv_thread, one row per post that was not
    removed, has a known author and at least one comment:
    {original, reply, label, post_id, author, created_utc, n_comments}.

    original  title + selftext, without the moderator footnote
    reply     the discussion in time order, one "OP: ..." / "User 3: ..." turn per comment
    label     delta if DeltaBot confirmed a delta the OP awarded (the `delta` field is empty before 2015)

    The discussion stops right before the OP's first award attempt: their first comment that has a
    delta symbol or that DeltaBot answered (confirmed or rejected). A delta thread therefore holds
    the comment that earned the delta, but not the award or anything after it. A no-delta thread
    usually has no attempt and keeps every comment, so the two classes differ in length and in how
    they end: compare a model against n_comments alone. Bot comments, "Edit:" notes and paragraphs
    that mention a delta are removed. Split by post like cmv, so a post is in the same split in
    both configs.
    Rows are streamed to the datasets cache, since the threads do not fit in memory.
    """
    ds = Dataset.from_generator(cmv_thread_rows, gen_kwargs={"path": path})
    ds = ds.cast_column("label", ClassLabel(names=["nodelta", "delta"]))
    splits = list(ds["split"])
    return {"cmv_thread": DatasetDict({
        name: ds.select([i for i, s in enumerate(splits) if s == name]).remove_columns("split")
        for name in ("train", "validation", "test")
    })}

SOURCES = {
    "cmv": ("cmv/pairs.jsonl", build_cmv),
    "ddo": ("ddo/debates.json", build_ddo_votes),
    "cmv_thread": ("cmv/threads.jsonl", build_cmv_thread),
}

def parse_args():
    ap = argparse.ArgumentParser(description="Build the persuasion datasets and push them to the Hub.")
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                    help="which raw sources to build (cmv -> cmv; ddo -> ddo_switch, ddo_undecided; "
                         "cmv_thread -> cmv_thread)")
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
