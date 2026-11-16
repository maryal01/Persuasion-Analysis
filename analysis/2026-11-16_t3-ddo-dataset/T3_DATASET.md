# T3 audience profiling dataset (DDO)

These are the user profiles and activity histories for T3. They're built from the DDO corpus
(`users.json`, `debates.json`) by `scripts/t3/build_dataset.py`, using the mapping in
`scripts/t3/dimensions.json`.

## Load

```python
from datasets import load_dataset

REPO = "nlp-final-project11/persuasion-datasets"
users   = load_dataset(REPO, "ddo_users",   revision="t3")
history = load_dataset(REPO, "ddo_history", revision="t3")
```

## `ddo_users`: one row per user (every `users.json` user plus every T2 voter)

| Column | Content |
|---|---|
| `user` | DDO username (= T2 `voter`) |
| `economy` | `left / mixed / right` |
| `social_issues` | `liberal / mixed / conservative` |
| `personal_liberties` | `restrictive / mixed / permissive` |
| `foreign_policy` | `dovish / mixed / hawkish` |
| `religion` | `christian / other religion / nonreligious / other` |
| `political_ideology_group` | `liberal / conservative / libertarian / socialist / other` |
| `<dim>_score`, `<dim>_n` | issue score in [−1, 1] and number of issues answered (four issue dimensions) |
| `political_ideology`, `religious_ideology` | raw values |
| `issue_<name>` (48) | raw answers: `Pro / Con / Und / N/O / N/S` |
| `has_profile` | user has a `users.json` record |

Labels are ClassLabels, and unknown labels are null.

**How an issue dimension is labelled:**
- Pro counts as the issue's polarity (±1), Con as the opposite, and Und as 0. N/O and N/S count as
  not answered.
- The score is the mean over the answered issues.
- The class is the first one if the score is ≤ −1/3, the last one if it is ≥ +1/3, and `mixed`
  otherwise.
- The label is null when fewer than 2 issues are answered.

## `ddo_history`: one row per activity

The columns are `user`, `kind`, `debate_id`, `text`, `side`, `side_before`, `title`, `category`,
`debate_date` and `debate_split`.

| `kind` | `text` | `side` / `side_before` |
|---|---|---|
| `debate_argument` | the user's turns in one debate | side argued |
| `vote` | `""` | side after / side before the debate |
| `comment` | comment text | — |
| `opinion_argument` | opinion text | — |
| `poll_vote` | chosen option and explanation | — |

`debate_id`, `category` and `debate_date` are set for debate rows and null otherwise.

**Votes without sides.** `side` and `side_before` are null for two groups of vote rows:
- **Old-form votes (37,426 rows):** votes dated "10 years ago" in `debates.json` that are tied before
  the debate. They were cast before about September 2008, under a voting form that most likely did
  not ask for a stance. T2 leaves the same votes out of `ddo_undecided` (PR #24).
- **Unclear votes (3,480 rows):** the voter marked no side, or more than one.

## Splits

- **Both tables split by user** with `split_of(user)` from `scripts/create_dataset.py`.
- **`debate_split`** is the T2 split of the debate.
- **For T2 profiles:** keep rows with `debate_split` equal to `"train"` or null, and drop the target
  debate.

## Counts

| | train | validation | test | total |
|---|---:|---:|---:|---:|
| `ddo_users` | 37,712 | 4,726 | 4,707 | 47,145 |
| `ddo_history` | 963,521 | 108,240 | 137,021 | 1,208,782 |

All counts are train / validation / test.

| Label | Labelled users | Classes |
|---|---|---|
| economy | 10,427 / 1,354 / 1,289 | left 5,418 / 701 / 648 · mixed 2,802 / 351 / 366 · right 2,207 / 302 / 275 |
| social_issues | 12,506 / 1,594 / 1,569 | liberal 5,334 / 670 / 679 · mixed 3,395 / 433 / 407 · conservative 3,777 / 491 / 483 |
| personal_liberties | 11,209 / 1,411 / 1,401 | restrictive 1,609 / 221 / 214 · mixed 3,826 / 453 / 483 · permissive 5,774 / 737 / 704 |
| foreign_policy | 8,917 / 1,151 / 1,093 | dovish 5,199 / 661 / 626 · mixed 1,564 / 198 / 177 · hawkish 2,154 / 292 / 290 |
| religion | 12,318 / 1,527 / 1,542 | christian 5,373 / 651 / 709 · other religion 1,643 / 196 / 209 · nonreligious 4,527 / 583 / 544 · other 775 / 97 / 80 |
| political_ideology_group | 10,013 / 1,275 / 1,254 | liberal 2,445 / 297 / 286 · conservative 2,068 / 260 / 275 · libertarian 1,181 / 170 / 155 · socialist 859 / 103 / 120 · other 3,460 / 445 / 418 |

**Blank profiles:** 26,927 of the 45,348 `users.json` users (59.4%) answered N/S to all 48 issues.

**History rows by kind:**
- debate_argument 148,527
- vote 175,365
- comment 561,485
- opinion_argument 61,183
- poll_vote 262,222

**Users with text:** 43,606. Of these, 19,123 have at least one label and 15,343 have an
issue-dimension label.

**Text per user** (all of a user's items together; 43,606 users), measured on the published build:

| | Median | 90th percentile | 99th percentile |
|---|---:|---:|---:|
| words | 492 | 5,541.5 | 59,025.5 |
| ModernBERT tokens | 630 | 7,387.5 | 80,566.3 |

16.2% of users have more than 4,096 tokens, and 9.1% have more than 8,192. Tokens are summed per
item, without special tokens.

**T2 coverage** (distinct voters in `ddo_switch` + `ddo_undecided`: 5,730 voters, 132,819 rows):

| T2 voters with… | Voters | Share of T2 rows |
|---|---:|---:|
| an issue label | 3,067 (53.5%) | 72.8% |
| any label | 3,454 (60.3%) | 80.6% |
| any text | 4,567 (79.7%) | 97.2% |
| neither a label nor text | 1,019 (17.8%) | 2.2% |

Share of T2 voters / rows with each label:
- economy 48.5% / 70.0%
- social_issues 51.5% / 69.7%
- personal_liberties 49.7% / 70.2%
- foreign_policy 42.3% / 63.1%
- religion 48.8% / 70.9%
- political_ideology_group 47.0% / 68.8%

## Build

```bash
python scripts/t3/build_dataset.py --users users.json --debates debates.json --dry-run     # counts only
python scripts/t3/build_dataset.py --users users.json --debates debates.json --limit 500  # sample
python scripts/t3/build_dataset.py --users users.json --debates debates.json              # full build
```

The build is deterministic. It validates the output and writes Parquet files to `outputs/t3/`.
