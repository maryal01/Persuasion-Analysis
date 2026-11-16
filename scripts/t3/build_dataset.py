from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
import create_dataset as cd  # noqa: E402  (split_of, stance, BEFORE, AFTER: the team's T1/T2 rules)

T2_REPO = "nlp-final-project11/persuasion-datasets"
T2_CONFIGS = ("ddo_switch", "ddo_undecided")
# t1-t2 head on 2026-11-16 (ddo_undecided rebuilt without old-form tied votes, 2026-11-12)
T2_REVISION = "7b4add94855e8ba9181d0516ef47f1dc66330d8a"
SPLITS = ("train", "validation", "test")
KINDS = ("debate_argument", "vote", "comment", "opinion_argument", "poll_vote")
DEBATE_KINDS = ("debate_argument", "vote", "comment")
ANSWERS = ("Pro", "Con", "Und", "N/O", "N/S")
NOT_SAYING = "Not Saying"
HISTORY_COLUMNS = ("user", "kind", "debate_id", "text", "side", "side_before", "title", "category",
                   "debate_date", "debate_split")
# users.json fields that must never be published (contract rule 8)
PRIVATE_FIELDS =("email", "birthday", "url", "friends", "ethnicity", "gender", "income", "education",
                  "relationship", "interested", "looking", "elo_ranking", "percentile", "joined",
                  "last_online", "last_updated")
DEFAULT_MAPPING = Path(__file__).resolve().parent / "dimensions.json"
# Votes the corpus dates "10 years ago" were cast before about September 2008, under a voting form that
# most likely did not ask for a stance: 98.9% of them are stored as tied before and after, against 2.3%
# of the votes dated "9 years ago". T2 leaves the tied-before ones out of ddo_undecided (PR #24). Here
# the vote row stays, with empty sides.
OLD_FORM_VOTE_TIME = "10 years ago"


# --------------------------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------------------------

def issue_column(issue: str) -> str:
    return "issue_" + re.sub(r"[^a-z0-9]+", "_", issue.lower()).strip("_")


@dataclass
class Mapping:
    dimensions: dict[str, dict]           # dim -> {"classes": [...], "issues": {issue: polarity}}
    excluded: dict[str, str]
    political_groups: dict[str, str]
    political_classes: list[str]
    religion_classes: list[str]
    religion_rules: list[dict]
    sha256: str = ""
    issues: list[str] = field(default_factory=list)  # all 48, sorted

    @property
    def issue_dims(self) -> list[str]:
        return list(self.dimensions)

    def religion_group(self, value: str | None) -> str | None:
        if value is None or value == NOT_SAYING:
            return None
        for rule in self.religion_rules:
            if (rule["match"] == "prefix" and value.startswith(rule["value"])) or \
               (rule["match"] == "exact" and value in rule["values"]):
                return rule["group"]
        raise KeyError(f"religious_ideology {value!r} is not covered by dimensions.json")

    def political_group(self, value: str | None) -> str | None:
        if value is None or value == NOT_SAYING:
            return None
        if value not in self.political_groups:
            raise KeyError(f"political_ideology {value!r} is not covered by dimensions.json")
        return self.political_groups[value]


def load_mapping(path: Path = DEFAULT_MAPPING) -> Mapping:
    raw = path.read_bytes()
    m = json.loads(raw)
    dims = {name: {"classes": d["classes"], "issues": {i: v["pro_polarity"] for i, v in d["issues"].items()}}
            for name, d in m["dimensions"].items()}
    mapped = [i for d in dims.values() for i in d["issues"]]
    if len(mapped) != len(set(mapped)):
        raise ValueError("an issue is mapped to more than one dimension")
    if set(mapped) & set(m["excluded_issues"]):
        raise ValueError("an issue is both mapped and excluded")
    for name, d in dims.items():
        if len(d["classes"]) != 3 or d["classes"][1] != "mixed":
            raise ValueError(f"{name}: expected [negative pole, 'mixed', positive pole]")
        if any(p not in (1, -1) for p in d["issues"].values()):
            raise ValueError(f"{name}: pro_polarity must be +1 or -1")
    pol = m["political_ideology_group"]
    rel = m["religion"]
    if set(pol["groups"].values()) - set(pol["classes"]) or set(r["group"] for r in rel["rules"]) - set(rel["classes"]):
        raise ValueError("a group is not one of its declared classes")
    return Mapping(dims, m["excluded_issues"], pol["groups"], pol["classes"], rel["classes"], rel["rules"],
                   hashlib.sha256(raw).hexdigest(), sorted(mapped + list(m["excluded_issues"])))


def check_mapping_covers(mapping: Mapping, users: dict) -> None:
    """Stop if the raw data has an issue or ideology value the mapping does not cover (or vice versa)."""
    seen_issues = {i for u in users.values() for i in (u.get("big_issues_dict") or {})}
    if seen_issues != set(mapping.issues):
        raise ValueError(f"issues in users.json and dimensions.json differ: only in data {sorted(seen_issues - set(mapping.issues))}, "
                         f"only in mapping {sorted(set(mapping.issues) - seen_issues)}")
    answers = {a for u in users.values() for a in (u.get("big_issues_dict") or {}).values()}
    if answers - set(ANSWERS):
        raise ValueError(f"unexpected issue answers {sorted(answers - set(ANSWERS))}")
    for u in users.values():
        mapping.political_group(u.get("political_ideology"))
        mapping.religion_group(u.get("religious_ideology"))


# --------------------------------------------------------------------------------------------
# ddo_users
# --------------------------------------------------------------------------------------------

def score_dimension(answers: dict[str, str], polarity: dict[str, int], min_answered: int = 2) -> tuple[float | None, int, int | None]:
    """(score, n answered, class index) for one dimension. Pro = polarity, Con = -polarity, Und = 0;
    N/O, N/S and missing do not count. Classes: 0 if score <= -1/3, 2 if >= +1/3, else 1 (mixed),
    compared in integers; None below min_answered."""
    values = [{"Pro": p, "Con": -p, "Und": 0}[answers[i]] for i, p in polarity.items() if answers.get(i) in ("Pro", "Con", "Und")]
    n, total = len(values), sum(values)
    if n < min_answered:
        return None, n, None
    cls = 0 if 3 * total <= -n else 2 if 3 * total >= n else 1
    return total / n, n, cls


def user_row(name: str, record: dict | None, mapping: Mapping) -> dict:
    row: dict = {"user": name}
    answers = (record or {}).get("big_issues_dict") or {}
    scores = {}
    for dim, d in mapping.dimensions.items():
        score, n, cls = score_dimension(answers, d["issues"])
        row[dim] = cls
        scores[dim] = (score, n)
    pol_raw = (record or {}).get("political_ideology")
    rel_raw = (record or {}).get("religious_ideology")
    rel = mapping.religion_group(rel_raw)
    pol = mapping.political_group(pol_raw)
    row["religion"] = None if rel is None else mapping.religion_classes.index(rel)
    row["political_ideology_group"] = None if pol is None else mapping.political_classes.index(pol)
    for dim, (score, n) in scores.items():
        row[f"{dim}_score"] = score
        row[f"{dim}_n"] = n
    row["political_ideology"] = None if pol_raw in (None, NOT_SAYING) else pol_raw
    row["religious_ideology"] = None if rel_raw in (None, NOT_SAYING) else rel_raw
    for issue in mapping.issues:
        row[issue_column(issue)] = answers.get(issue) if record is not None else None
    row["has_profile"] = record is not None
    return row


def users_features(mapping: Mapping):
    from datasets import ClassLabel, Features, Value
    f = {"user": Value("string")}
    for dim, d in mapping.dimensions.items():
        f[dim] = ClassLabel(names=d["classes"])
    f["religion"] = ClassLabel(names=mapping.religion_classes)
    f["political_ideology_group"] = ClassLabel(names=mapping.political_classes)
    for dim in mapping.dimensions:
        f[f"{dim}_score"] = Value("float64")
        f[f"{dim}_n"] = Value("int64")
    f["political_ideology"] = Value("string")
    f["religious_ideology"] = Value("string")
    for issue in mapping.issues:
        f[issue_column(issue)] = Value("string")
    f["has_profile"] = Value("bool")
    return Features(f)


def build_user_rows(users: dict, t2_voters: Iterable[str], mapping: Mapping) -> list[dict]:
    """users.json order, then T2 voters without a record in sorted order."""
    rows = [user_row(name, rec, mapping) for name, rec in users.items()]
    rows += [user_row(name, None, mapping) for name in sorted(set(t2_voters) - set(users))]
    return rows


def has_issue_label(row: dict, mapping: Mapping) -> bool:
    return any(row[d] is not None for d in mapping.issue_dims)


def sample_users(rows: list[dict], mapping: Mapping, limit: int) -> set[str]:
    """Deterministic ~limit users: 60% with an issue-dimension label, 20% T2 voters without a
    profile, 20% other profiles; within each stratum the lowest md5("t3-sample:" + user)."""
    strata = {"labelled": [], "t2_only": [], "other": []}
    for r in rows:
        key = "labelled" if has_issue_label(r, mapping) else "t2_only" if not r["has_profile"] else "other"
        strata[key].append(r["user"])
    quota = {"labelled": round(limit * 0.6), "t2_only": round(limit * 0.2)}
    quota["other"] = limit - quota["labelled"] - quota["t2_only"]
    rank = lambda u: hashlib.md5(f"t3-sample:{u}".encode()).hexdigest()
    return {u for k, names in strata.items() for u in sorted(names, key=rank)[:quota[k]]}


# --------------------------------------------------------------------------------------------
# ddo_history
# --------------------------------------------------------------------------------------------

def history_row(user, kind, text, title, debate_id=None, side=None, side_before=None, category=None, debate_date=None) -> dict:
    return {"user": user, "kind": kind, "debate_id": debate_id, "text": text, "side": side, "side_before": side_before,
            "title": title, "category": category, "debate_date": debate_date,
            "debate_split": cd.split_of(debate_id) if debate_id is not None else None}


def iter_debates(debates_path: Path) -> Iterator[tuple[str, dict]]:
    import ijson
    with open(debates_path, "rb") as fh:
        yield from ijson.kvitems(fh, "")


def debate_history(debate_id: str, d: dict, keep: Callable[[str], bool], counts: Counter) -> Iterator[dict]:
    """debate_argument, vote and comment rows of one debate (T2's voter rules and stance()).

    A tied-before vote cast under the old voting form keeps its row but has no sides (OLD_FORM_VOTE_TIME).
    """
    base = {"debate_id": debate_id, "title": (d.get("title") or "").strip(), "category": d.get("category") or None,
            "debate_date": d.get("start_date") or None}
    positions = {d.get("participant_1_name"): d.get("participant_1_position"),
                 d.get("participant_2_name"): d.get("participant_2_position")}
    for rnd in d.get("rounds", []):
        counts["turns_with_unknown_side"] += sum(t.get("side") not in ("Pro", "Con") for t in rnd)
    for name, side in positions.items():
        if keep(name):
            turns = [t["text"].strip() for rnd in d.get("rounds", []) for t in rnd
                     if t.get("side") == side and (t.get("text") or "").strip().lower() not in ("", "forfeit")]
            counts["debate_arguments_without_text"] += not turns
            yield history_row(name, "debate_argument", "\n\n".join(turns), side=side, **base)
    seen = set()
    for v in d.get("votes", []):
        voter = v.get("user_name")
        if voter in positions:
            counts["votes_skipped_debater"] += 1
            continue
        if voter in seen:
            counts["votes_skipped_repeat"] += 1
            continue
        seen.add(voter)
        if keep(voter):
            votes_map = v.get("votes_map") or {}
            side = cd.stance(votes_map, positions, cd.AFTER)
            side_before = cd.stance(votes_map, positions, cd.BEFORE)
            if side_before == "Tie" and (v.get("time") or "").strip() == OLD_FORM_VOTE_TIME:
                counts["votes_old_form_sides_emptied"] += 1
                side = side_before = None
            yield history_row(voter, "vote", "", side=side, side_before=side_before, **base)
    for c in d.get("comments", []):
        if keep(c.get("user_name")):
            yield history_row(c["user_name"], "comment", c.get("comment_text") or "", **base)


def user_history(name: str, record: dict) -> Iterator[dict]:
    """opinion_argument and poll_vote rows of one users.json record (questions/polls asked are left out)."""
    for a in record.get("opinion_arguments") or []:
        yield history_row(name, "opinion_argument", a.get("opinion text") or "", a.get("opinion title") or "")
    for v in record.get("poll_votes") or []:
        text, expl = v.get("vote text") or "", v.get("vote explanation") or ""
        yield history_row(name, "poll_vote", text + ("\n\n" + expl if expl.strip() else ""), v.get("vote title") or "")


def iter_history(debates: Iterable[tuple[str, dict]], users: dict, keep: Callable[[str], bool], counts: Counter) -> Iterator[dict]:
    for debate_id, d in debates:
        counts["debates"] += 1
        yield from debate_history(debate_id, d, keep, counts)
    for name, record in users.items():
        if keep(name):
            yield from user_history(name, record)


# --------------------------------------------------------------------------------------------
# T2
# --------------------------------------------------------------------------------------------

def load_t2(revision: str = T2_REVISION) -> list[dict]:
    """T2 vote rows: voter, debate_id, before, after and the split they are in."""
    from datasets import load_dataset
    rows = []
    for config in T2_CONFIGS:
        ds = load_dataset(T2_REPO, config, revision=revision, columns=["voter", "debate_id", "before", "after"])
        for split in SPLITS:
            for r in ds[split]:
                rows.append({**r, "config": config, "split": split})
    return rows


# --------------------------------------------------------------------------------------------
# Statistics (computed while streaming)
# --------------------------------------------------------------------------------------------

class HistoryStats:
    def __init__(self, user_split: dict[str, str]):
        self.user_split = user_split
        self.rows = Counter()
        self.empty_text = Counter()
        self.words = Counter()
        self.text_users: set[str] = set()

    def add(self, row: dict) -> None:
        split = self.user_split[row["user"]]
        self.rows[(split, row["kind"])] += 1
        text = row["text"]
        if text.strip():
            self.text_users.add(row["user"])
            self.words[row["user"]] += len(text.split())
        elif row["kind"] != "vote":
            self.empty_text[row["kind"]] += 1


def percentiles(values: list[int]) -> dict:
    import numpy as np
    if not values:
        return {}
    a = np.asarray(values)
    return {"users": int(a.size), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99))}


def compute_stats(user_rows: list[dict], user_split: dict[str, str], hs: HistoryStats, t2_rows: list[dict],
                  mapping: Mapping, counts: Counter) -> dict:
    labels = mapping.issue_dims + ["religion", "political_ideology_group"]
    classes = {**{d: mapping.dimensions[d]["classes"] for d in mapping.issue_dims},
               "religion": mapping.religion_classes, "political_ideology_group": mapping.political_classes}
    by_user = {r["user"]: r for r in user_rows}
    st: dict = {"users_by_split": dict(Counter(user_split.values())),
                "users_with_profile": sum(r["has_profile"] for r in user_rows),
                "users_without_profile_t2_voters": sum(not r["has_profile"] for r in user_rows)}
    st["class_counts"] = {lab: {split: {c: 0 for c in classes[lab] + ["null"]} for split in SPLITS} for lab in labels}
    for r in user_rows:
        for lab in labels:
            st["class_counts"][lab][user_split[r["user"]]][classes[lab][r[lab]] if r[lab] is not None else "null"] += 1
    st["labelled_users"] = {lab: sum(r[lab] is not None for r in user_rows) for lab in labels}
    any_issue = {r["user"] for r in user_rows if has_issue_label(r, mapping)}
    any_label = {r["user"] for r in user_rows if any(r[l] is not None for l in labels)}
    st["users_with_issue_label_and_text"] = len(any_issue & hs.text_users)
    st["users_with_any_label_and_text"] = len(any_label & hs.text_users)
    st["users_with_text"] = len(hs.text_users)
    st["history_rows_by_kind"] = {k: {s: hs.rows[(s, k)] for s in SPLITS} for k in KINDS}
    st["history_rows_total"] = sum(hs.rows.values())
    st["empty_text_rows_by_kind_excluding_votes"] = dict(hs.empty_text)
    st["text_per_user_words"] = percentiles([hs.words[u] for u in hs.text_users])
    st["build_counts"] = dict(counts)
    if t2_rows:
        voters = Counter(r["voter"] for r in t2_rows)
        cov = {"t2_rows": len(t2_rows), "t2_distinct_voters": len(voters)}
        for lab in labels:
            have = {v for v in voters if v in by_user and by_user[v][lab] is not None}
            cov[f"label_{lab}"] = {"voters": len(have) / len(voters), "rows": sum(voters[v] for v in have) / len(t2_rows)}
        groups = {"any_issue_label": {v for v in voters if v in any_issue}, "any_label": {v for v in voters if v in any_label},
                  "any_text": {v for v in voters if v in hs.text_users}}
        groups["neither_label_nor_text"] = {v for v in voters if v not in any_label and v not in hs.text_users}
        groups["no_profile_record"] = {v for v in voters if v in by_user and not by_user[v]["has_profile"]}
        for name, have in groups.items():
            cov[name] = {"voters": len(have) / len(voters), "rows": sum(voters[v] for v in have) / len(t2_rows), "voter_count": len(have)}
        st["t2_coverage"] = cov
    return st




# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def null(value):
    """None for None and NaN (pandas' null for float and object columns), else the value."""
    return None if value is None or (isinstance(value, float) and value != value) else value


def validate_tables(users_dsd, history_dsd, mapping: Mapping, t2_rows: list[dict] | None = None,
                    raw_users: dict | None = None, debate_ids: set[str] | None = None,
                    expected_users: set[str] | None = None) -> dict:
    """All checks on the two DatasetDicts (as saved/reloaded). Optional raw inputs enable source checks.
    Raises ValidationError listing every failure."""
    errors, passed = [], []

    def check(ok, name, detail=""):
        (passed.append(name) if ok else errors.append(f"{name}: {detail}" if detail else name))

    feats = users_features(mapping)
    labels = mapping.issue_dims + ["religion", "political_ideology_group"]
    user_split = {}
    for split in SPLITS:
        ds = users_dsd[split]
        check(ds.features == feats, f"ddo_users/{split}: columns, dtypes and class names", f"{list(ds.features)[:6]}...")
        df = ds.to_pandas()
        check(df["user"].notna().all() and df["user"].is_unique, f"ddo_users/{split}: user non-null and unique")
        wrong = [u for u in df["user"] if cd.split_of(u) != split]
        check(not wrong, f"ddo_users/{split}: every user is in split_of(user)", f"{len(wrong)} e.g. {wrong[:3]}")
        for u in df["user"]:
            if u in user_split:
                errors.append(f"user {u!r} in {user_split[u]} and {split}")
            user_split[u] = split
        forbidden = set(df.columns) & (set(PRIVATE_FIELDS) | {"party", "president", "description", "big_issues_dict"})
        check(not forbidden, f"ddo_users/{split}: no private or extra profile columns", f"{sorted(forbidden)}")
        issue_cols = [issue_column(i) for i in mapping.issues]
        prof, nop = df[df["has_profile"]], df[~df["has_profile"]]
        check(prof[issue_cols].isin(ANSWERS).all().all(), f"ddo_users/{split}: issue answers legal for profiles")
        check(nop[issue_cols].isna().all().all() and nop[labels + ["political_ideology", "religious_ideology"]].isna().all().all(),
              f"ddo_users/{split}: users without a profile have only nulls")
        check(not df[["political_ideology", "religious_ideology"]].isin([NOT_SAYING]).any().any(),
              f"ddo_users/{split}: 'Not Saying' stored as null")
        # recompute every label from the stored raw columns (pandas may turn nulls into NaN)
        bad = 0
        for r in df.to_dict("records"):
            r = {k: null(v) for k, v in r.items()}
            answers = {i: r[issue_column(i)] for i in mapping.issues if r[issue_column(i)] is not None}
            for dim, d in mapping.dimensions.items():
                score, n, cls = score_dimension(answers, d["issues"])
                stored = None if r[dim] is None else int(r[dim])
                s_score = r[f"{dim}_score"]
                bad += stored != cls or r[f"{dim}_n"] != n or (score is None) != (s_score is None) or \
                    (score is not None and abs(score - s_score) > 1e-12)
            rel = mapping.religion_group(r["religious_ideology"])
            pol = mapping.political_group(r["political_ideology"])
            for lab, val, names in (("religion", rel, mapping.religion_classes), ("political_ideology_group", pol, mapping.political_classes)):
                stored = None if r[lab] is None else int(r[lab])
                bad += stored != (None if val is None else names.index(val))
        check(bad == 0, f"ddo_users/{split}: labels, scores and counts recompute from the raw columns", f"{bad} mismatches")
    check(len(user_split) == sum(len(users_dsd[s]) for s in SPLITS), "ddo_users: no user in two splits")
    if expected_users is not None:
        check(set(user_split) == expected_users, "ddo_users: exactly the expected users",
              f"missing {len(expected_users - set(user_split))}, extra {len(set(user_split) - expected_users)}")
    if raw_users is not None:
        profile_flags = {}
        for split in SPLITS:
            for u, hp in zip(users_dsd[split]["user"], users_dsd[split]["has_profile"]):
                profile_flags[u] = hp
        check(all(profile_flags[u] == (u in raw_users) for u in profile_flags), "ddo_users: has_profile == has a users.json record")

    # history
    vote_keys, arg_keys = Counter(), Counter()
    votes_by_key = {}
    hist_users = set()
    for split in SPLITS:
        ds = history_dsd[split]
        check(tuple(ds.column_names) == HISTORY_COLUMNS, f"ddo_history/{split}: columns", f"{ds.column_names}")
        df = ds.to_pandas()
        check(df["kind"].isin(KINDS).all(), f"ddo_history/{split}: kind legal")
        check(df["user"].notna().all() and df["text"].notna().all() and df["title"].notna().all(),
              f"ddo_history/{split}: user, text, title never null")
        unknown = set(df["user"]) - set(user_split)
        check(not unknown, f"ddo_history/{split}: every user is in ddo_users", f"{len(unknown)}")
        wrong = int((df["user"].map(user_split) != split).sum())
        check(wrong == 0, f"ddo_history/{split}: rows are in their user's split", f"{wrong}")
        hist_users |= set(df["user"])
        deb = df["kind"].isin(DEBATE_KINDS)
        check(df.loc[deb, ["debate_id", "debate_split", "debate_date", "category"]].notna().all().all(),
              f"ddo_history/{split}: debate rows have debate_id, debate_split, debate_date, category")
        check(df.loc[~deb, ["debate_id", "debate_split", "debate_date", "category", "side", "side_before"]].isna().all().all(),
              f"ddo_history/{split}: opinion/poll rows have no debate fields or sides")
        check(bool((df.loc[deb, "debate_split"] == df.loc[deb, "debate_id"].map(cd.split_of)).all()),
              f"ddo_history/{split}: debate_split == split_of(debate_id)")
        v = df[df["kind"] == "vote"]
        check((v["text"] == "").all(), f"ddo_history/{split}: vote text is empty")
        check(v["side"].isin(["Pro", "Con", "Tie"]).where(v["side"].notna(), True).all() and
              v["side_before"].isin(["Pro", "Con", "Tie"]).where(v["side_before"].notna(), True).all(),
              f"ddo_history/{split}: vote sides legal")
        a = df[df["kind"] == "debate_argument"]
        check(a["side"].isin(["Pro", "Con"]).all() and a["side_before"].isna().all(), f"ddo_history/{split}: argument sides legal")
        c = df[df["kind"] == "comment"]
        check(c[["side", "side_before"]].isna().all().all(), f"ddo_history/{split}: comments have no side")
        if debate_ids is not None:
            missing = set(df.loc[deb, "debate_id"]) - debate_ids
            check(not missing, f"ddo_history/{split}: every debate_id is a debates.json key", f"{len(missing)}")
        vote_keys.update(zip(v["user"], v["debate_id"]))
        arg_keys.update(zip(a["user"], a["debate_id"]))
        votes_by_key.update({(u, d): (s, b) for u, d, s, b in zip(v["user"], v["debate_id"], v["side"], v["side_before"])})
    check(all(n == 1 for n in vote_keys.values()), "ddo_history: one vote per user and debate")
    check(all(n == 1 for n in arg_keys.values()), "ddo_history: one debate_argument per user and debate")

    report = {"checks_passed": len(passed)}
    if t2_rows is not None:
        in_scope = [r for r in t2_rows if r["voter"] in user_split]
        missing_voters = {r["voter"] for r in t2_rows} - set(user_split)
        if expected_users is None or {r["voter"] for r in t2_rows} <= expected_users:
            check(not missing_voters, "every T2 voter is in ddo_users", f"{len(missing_voters)} missing")
        miss = [r for r in in_scope if (r["voter"], r["debate_id"]) not in votes_by_key]
        check(not miss, "every T2 row (voter, debate_id) has its vote in ddo_history", f"{len(miss)} missing, e.g. {miss[:2]}")
        diff = [r for r in in_scope if (r["voter"], r["debate_id"]) in votes_by_key and
                votes_by_key[(r["voter"], r["debate_id"])] != (r["after"], r["before"])]
        check(not diff, "vote side / side_before equal T2 after / before", f"{len(diff)} differ, e.g. {diff[:2]}")
        bad_split = [r for r in in_scope if cd.split_of(r["debate_id"]) != r["split"]]
        check(not bad_split, "T2 rows are in split_of(debate_id), so debate_split is T2's split", f"{len(bad_split)}")
        report["t2_rows_checked"] = len(in_scope)
    report["checks_passed"] = len(passed)
    if errors:
        raise ValidationError(f"{len(errors)} check(s) failed:\n  - " + "\n  - ".join(errors))
    return report


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------

def history_schema():
    import pyarrow as pa
    return pa.schema([(c, pa.string()) for c in HISTORY_COLUMNS])


class HistoryWriter:
    def __init__(self, folder: Path, user_split: dict[str, str], batch: int = 50_000):
        import pyarrow.parquet as pq
        folder.mkdir(parents=True, exist_ok=True)
        self.writers = {s: pq.ParquetWriter(str(folder / f"{s}.parquet"), history_schema()) for s in SPLITS}
        self.buffers = {s: [] for s in SPLITS}
        self.user_split, self.batch = user_split, batch

    def add(self, row: dict) -> None:
        s = self.user_split[row["user"]]
        self.buffers[s].append(row)
        if len(self.buffers[s]) >= self.batch:
            self._flush(s)

    def _flush(self, s: str) -> None:
        import pyarrow as pa
        if self.buffers[s]:
            self.writers[s].write_table(pa.Table.from_pylist(self.buffers[s], schema=history_schema()))
            self.buffers[s] = []

    def close(self) -> None:
        for s in SPLITS:
            self._flush(s)
            self.writers[s].close()


def write_users(rows: list[dict], user_split: dict[str, str], mapping: Mapping, folder: Path) -> None:
    from datasets import Dataset
    folder.mkdir(parents=True, exist_ok=True)
    feats = users_features(mapping)
    for s in SPLITS:  # column lists, so an empty split still has every column
        part = [r for r in rows if user_split[r["user"]] == s]
        Dataset.from_dict({c: [r[c] for r in part] for c in feats}, features=feats).to_parquet(str(folder / f"{s}.parquet"))


def load_build(folder: str | Path):
    """(ddo_users, ddo_history) DatasetDicts from a build folder. Reads each split with pyarrow, so an
    empty split (possible in a small --limit sample) loads too; class names come from the Parquet
    schema metadata that `datasets` wrote."""
    import pyarrow.parquet as pq
    from datasets import Dataset, DatasetDict
    folder = Path(folder)
    load = lambda t: DatasetDict({s: Dataset(pq.read_table(str(folder / t / f"{s}.parquet"))) for s in SPLITS})
    return load("ddo_users"), load("ddo_history")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T3 audience profiling: build the DDO ddo_users and ddo_history tables from the raw corpus.")
    ap.add_argument("--users", required=True, help="raw DDO users.json")
    ap.add_argument("--debates", required=True, help="raw DDO debates.json (streamed)")
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="dimensions.json (default: %(default)s)")
    ap.add_argument("--out-dir", default="outputs/t3", help="git-ignored output folder (default: %(default)s)")
    ap.add_argument("--t2-revision", default=T2_REVISION, help="persuasion-datasets commit holding ddo_switch/ddo_undecided")
    ap.add_argument("--limit", type=int, default=None, help="deterministic sample of about N users (same schema)")
    ap.add_argument("--dry-run", action="store_true", help="build and print counts, write nothing")
    return ap.parse_args(argv)


def run(users_path: Path, debates_path: Path, mapping: Mapping, t2_rows: list[dict], out_dir: Path | None,
        limit: int | None = None, extra_manifest: dict | None = None) -> dict:
    """Build, compute statistics, and (unless out_dir is None) write and validate. Returns the stats."""
    users = json.loads(users_path.read_text())
    check_mapping_covers(mapping, users)
    t2_voters = {r["voter"] for r in t2_rows}
    rows = build_user_rows(users, t2_voters, mapping)
    if limit:
        chosen = sample_users(rows, mapping, limit)
        rows = [r for r in rows if r["user"] in chosen]
    user_split = {r["user"]: cd.split_of(r["user"]) for r in rows}
    if limit:  # coverage and T2 checks only for the sampled voters
        t2_rows = [r for r in t2_rows if r["voter"] in user_split]
    counts = Counter()
    hs = HistoryStats(user_split)
    staging = None
    if out_dir is not None:
        name = f"sample{limit}" if limit else "full"
        staging = out_dir / f".staging_{name}"
        if staging.exists():
            shutil.rmtree(staging)
        writer = HistoryWriter(staging / "ddo_history", user_split)
    debate_ids: set[str] = set()

    def debates():
        for debate_id, d in iter_debates(debates_path):
            debate_ids.add(debate_id)
            yield debate_id, d

    for row in iter_history(debates(), users, user_split.__contains__, counts):
        hs.add(row)
        if staging is not None:
            writer.add(row)
    stats = compute_stats(rows, user_split, hs, t2_rows, mapping, counts)
    print_stats(stats, mapping)
    if staging is None:
        return stats

    writer.close()
    write_users(rows, user_split, mapping, staging / "ddo_users")
    users_dsd, history_dsd = load_build(staging)
    expected = set(user_split)
    report = validate_tables(users_dsd, history_dsd, mapping, t2_rows=t2_rows, raw_users=users,
                             debate_ids=debate_ids, expected_users=expected)
    print(f"[VALID] {report['checks_passed']} checks passed on the reloaded Parquet files"
          + (f"; {report.get('t2_rows_checked', 0)} T2 rows matched to their votes" if t2_rows else ""))
    stats["validation"] = report
    manifest = {
        "task": "T3 DDO audience profiling (contract 2026-11-09)",
        "inputs": {"users.json": sha256_file(users_path), "debates.json": sha256_file(debates_path),
                   "dimensions.json": mapping.sha256, "t2": f"{T2_REPO} {list(T2_CONFIGS)} @ {extra_manifest.get('t2_revision') if extra_manifest else T2_REVISION}"},
        "limit": limit, "split_rule": "scripts/create_dataset.py:split_of(user)",
        "files": {str(p.relative_to(staging)): sha256_file(p) for p in sorted(staging.rglob("*.parquet"))},
        **(extra_manifest or {}),
    }
    (staging / "stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    final = out_dir / (f"sample{limit}" if limit else "full")
    if final.exists():
        old = json.loads((final / "manifest.json").read_text()).get("files") if (final / "manifest.json").exists() else None
        if old != manifest["files"]:
            backup = out_dir / "previous" / time.strftime("%Y%m%d-%H%M%S") / final.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(final), str(backup))
            print(f"[KEPT] earlier build moved to {backup}")
        else:
            shutil.rmtree(final)
    shutil.move(str(staging), str(final))
    print(f"[SAVED] {final}/ddo_users/*.parquet, {final}/ddo_history/*.parquet, stats.json, manifest.json")
    return stats


def print_stats(st: dict, mapping: Mapping) -> None:
    print(f"[USERS] by split {st['users_by_split']}; with profile {st['users_with_profile']}, "
          f"T2 voters without a profile {st['users_without_profile_t2_voters']}")
    for lab, by_split in st["class_counts"].items():
        print(f"[LABEL {lab}] " + " | ".join(f"{s}: " + ", ".join(f"{c}={n}" for c, n in d.items()) for s, d in by_split.items()))
    print(f"[TEXT] users with text {st['users_with_text']}; with an issue label and text {st['users_with_issue_label_and_text']}; "
          f"with any label and text {st['users_with_any_label_and_text']}")
    print("[HISTORY] " + ", ".join(f"{k}={sum(v.values())}" for k, v in st["history_rows_by_kind"].items()) + f"; total {st['history_rows_total']}")
    print(f"[WORDS/USER] {st['text_per_user_words']}")
    print(f"[BUILD] {st['build_counts']}")
    if "t2_coverage" in st:
        c = st["t2_coverage"]
        print(f"[T2] {c['t2_distinct_voters']} voters, {c['t2_rows']} rows; "
              + "; ".join(f"{k}: {v['voters']:.1%} of voters / {v['rows']:.1%} of rows" for k, v in c.items() if isinstance(v, dict)))


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = load_mapping(Path(args.mapping))
    t2_rows = load_t2(args.t2_revision)
    run(Path(args.users).expanduser(), Path(args.debates).expanduser(), mapping, t2_rows,
        None if args.dry_run else Path(args.out_dir), args.limit, {"t2_revision": args.t2_revision})
    if args.dry_run:
        print("[DRY RUN] nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
