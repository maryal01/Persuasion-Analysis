# T5: why the label is a DeltaBot confirmation, and which text is cleaned

**Status:** done (measurements of 2-16 November 2026; the models that use the result are in `RESULTS.md` and `docs/T5_report.md`).

## Goal
One question for T5 (does the OP of a CMV post award a delta?): which label and which post text should the models use?
Checked: what the `delta` field of `posts_malleability` really measures and whether a better label exists, which text of a post leaks the outcome, and how far the LIWC/Receptiviti author features can be trusted.

## What it does
Reads the Webis-CMV-20 files from Zenodo ([record 3778298](https://zenodo.org/records/3778298)), all records, no sampling. `<cmv_dir>` is the folder with the `.jsonl.bz2` files.

| script | question | input |
|---|---|---|
| `label_by_year.py` | `delta` field vs subtree delta vs OP comments with a delta symbol, per year | posts_malleability, author_liwc |
| `deltabot_by_month.py` | per month: share of posts with the delta flag vs with a DeltaBot confirmation | posts_malleability |
| `t1_label_evidence.py` | do the delta comments of `pairs.jsonl` (T1) carry an OP symbol / DeltaBot confirmation? | pairs, threads |
| `label_noise.py` | label-0 posts of the T5 dataset where the OP wrote a delta symbol | posts_malleability, Hub dataset |
| `scripts/t5/inspect_labels.py` | read the suspicious threads yourself | posts_malleability |

The comparison of text clean-up rules (section 3) was a one-off script that is not kept; the rule is in `scripts/t5/t5_common.py` (`clean_text`). The data path of the scripts was turned into an argument after the measurements; they were only compiled afterwards, not run again.

## Run
```bash
python label_by_year.py <cmv_dir>     # the other scripts the same way; they print their tables (about 5 minutes each on one core)
python scripts/t5/inspect_labels.py collect --posts <cmv_dir>/posts_malleability.jsonl.bz2 --output <out_dir>/candidates.jsonl
python scripts/t5/inspect_labels.py show --input <out_dir>/candidates.jsonl --category false_op_symbol --n 10
```
`label_noise.py` reads the T5 dataset from the Hub and needs a Hugging Face login.

## Results

### 1. The `delta` field is a flair artifact
It comes from the post flair (`[Deltas Awarded]`, `[∆(s) from OP]`, `[View Changed]`). The 8,121 posts whose OP has LIWC features are the paper's sample (3,186 delta and 4,935 no-delta posts under this field).

| share of the 8,121 posts | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---|---|---|---|---|
| with `delta` (flair) | 0.0% | 0.1% | 48% | 58% | 65% |
| with a DeltaBot confirmation | 53% | 55% | 62% | 65% | 76% |

- Per month the flair share is 0% up to 2015-02, 17% in 2015-03, 25% in 2015-04, 57% in 2015-05, 62% in 2015-06. A logistic regression on the posting time alone reaches test AUROC 0.815 with this label.
- In 2013-2014 the OP wrote a delta symbol in 836 of 2,858 posts (2013) and 1,304 of 3,793 posts (2014) of `posts_malleability` (all 16,150 posts), all but one with `delta = False`.
- From 2015-05 on (8,433 posts, all OPs): 4,042 have `delta = True`, 4,391 `False`; 721 of the `False` posts contain an OP comment with a delta symbol. DeltaBot replied "Confirmed: 1 delta awarded" in 639 of them, only rejected in 22, and did not reply in 60. The confirmed misses carry other flairs that replace the delta flair: `FTFdeltaOP` 167, `Election` 108, none 102, `[OP ∆/Election]` 89, `[FreshTopicFriday]` 79. Of a random 60 posts with `delta = True`, 52 have a DeltaBot confirmation under an OP comment.
- A DeltaBot confirmation exists in every month from 2013-02 (27% of the posts in 2013-02, 38% in 2014-09, 47% in 2015-05, 56% in 2017-06).
- On the 8,121 posts: flair and DeltaBot both positive 2,987; flair only 199; DeltaBot only 2,192; both negative 2,743.
- **T1 uses the same evidence.** Of the 10,303 delta comments of `pairs.jsonl` found in `threads.jsonl`, 99.7% have an OP delta symbol and 95.7% a DeltaBot confirmation in their subtree (of the 9,028 no-delta comments found: 2.7% and 2.6%). The pairs cover 2013-2017 (845, 1,579, 2,202, 2,371, 3,306 per year), so they do not depend on the flair.

**Decision:** the label is "DeltaBot confirmed a delta that the OP awarded". Remaining noise: 331 of the 2,942 label-0 posts (11%, 4% of all posts) have an OP delta symbol without a confirmation (132 rejected, usually for a too short explanation; 199 got no bot reply).

### 2. Text that leaks the outcome
The stored posts are the crawled versions, written after the discussion. Share of the 8,121 posts that contain it:

| in the post text | delta | no delta |
|---|---|---|
| "edit" / "update" | 35.3% | 23.2% |
| the moderator footnote (49 different texts, 57% of all posts) | 17.5% | 9.4% |
| the word "delta" | 7.3% | 2.2% |
| a delta symbol (∆, Δ, !delta; 55 posts) | 1.0% | 0.1% |

Bag-of-words logistic regression (test AUROC) with different clean-up rules:

| rule | characters removed | AUROC |
|---|---|---|
| none | 0% | 0.603 |
| symbols only | 0.0% | 0.603 |
| footnote only | 16.7% | 0.580 |
| footnote + symbols + the paragraph of each edit marker | 20.4% | 0.563 |
| footnote + symbols + everything after the first edit marker | 23.2% | 0.560 |

The rule used removes the footnote, paragraphs that START with an edit/update marker ("Update the tax code ..." and "Editing is a skill ..." stay) and the symbols: 20.8% of the characters, 72.7% of the posts change, none becomes empty; 71 unusual edit lines in 65 posts and 5 footnote variants remain. With it, bag of words reaches 0.564 and the post length alone 0.500 (0.560 without). Part of the text signal was the footnote and the edits ("Edit: my view has not been changed 180 degrees ..." gives the outcome away).

### 3. LIWC / Receptiviti
`liwc_wc` (words the vector was computed from) is 1,245 to 10,242 for 90% of the authors, median 6,393; 99.9% have more than 1,000 words, the paper says the first 1,000. The vector is one per author, so it can contain text written after the post it is used for. This cannot be checked with the released files.

## Notes
- Not verified: that every OP delta symbol is an attempt to award a delta; the cause of the missing flair before 2015.
- The statistics of what the OPs wrote on CMV before a post (history for a possible history model) are not part of this analysis.
