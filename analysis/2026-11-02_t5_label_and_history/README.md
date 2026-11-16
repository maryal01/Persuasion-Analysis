# T5: what the delta label means, and what the OPs wrote before

**Status:** done (measurements of 2-16 November 2026; the models that use the result are in `RESULTS.md` and `docs/T5_report.md`).

## Goal
Before training anything for T5 (does the OP of a CMV post award a delta?) find out
1. what the `delta` field of `posts_malleability` really measures and whether a better label exists,
2. which text of a post leaks the outcome and has to be removed,
3. how trustworthy the LIWC/Receptiviti author features are,
4. how much the OPs wrote on CMV before a post (a possible input for a history model).

## What it does
Everything reads the Webis-CMV-20 files from Zenodo ([record 3778298](https://zenodo.org/records/3778298)), all records, no sampling.
The small scripts in this folder reproduce the label and coverage numbers; `<cmv_dir>` is the folder with the `.jsonl.bz2` files.

| script | question | input |
|---|---|---|
| `label_by_year.py` | `delta` field vs subtree delta vs OP comments with a delta symbol, per year | posts_malleability, author_liwc |
| `deltabot_by_month.py` | per month: share of posts with the delta flag vs with a DeltaBot confirmation | posts_malleability |
| `t1_label_evidence.py` | do the delta comments of `pairs.jsonl` carry an OP symbol / DeltaBot confirmation? | pairs, threads |
| `label_noise.py` | label 0 posts of the T5 dataset where the OP wrote a delta symbol | posts_malleability, Hub dataset |
| `threads_coverage.py` | is `threads.jsonl` complete? overlap with the other files | threads, posts_malleability, pairs |
| `threads_posts_per_op.py` | posts per OP and earlier posts of the same OP in `threads.jsonl` | threads |
| `earlier_posts_per_post.py` | delta rate by the number of earlier posts of the OP (T5 posts) | threads, Hub dataset |
| `scripts/t5/inspect_labels.py` | read the suspicious threads yourself | posts_malleability |

The checks of the text clean-up and of the history length (token counts, ModernBERT tokenizer) were one-off scripts that are not kept;
their method is described below. The path of the data was turned into an argument after the measurements, the scripts were only
compiled afterwards, not run again.

## Run
```bash
python label_by_year.py <cmv_dir>     # the other scripts the same way; they print their tables
python scripts/t5/inspect_labels.py collect --posts <cmv_dir>/posts_malleability.jsonl.bz2 --output <out_dir>/candidates.jsonl
python scripts/t5/inspect_labels.py show --input <out_dir>/candidates.jsonl --category false_op_symbol --n 10
```
Scripts that read the T5 dataset need a Hugging Face login. They take about 5 minutes each on one core (`threads.jsonl` is 660 MB compressed).

## Results

### 1. The data
- `posts_malleability`: 16,150 posts, all in r/changemyview, 11,806 OPs (1,949 with two or more posts, at most 40). The base-36 post id is in the same order as `created_utc` for all 16,150 posts (Spearman 1.0).
- The Zenodo record describes no train/test split. The paper (Al-Khatib et al. 2020) used 8,121 posts with a holdout of 1,330 that cannot be identified in the files.
- The 8,121 posts whose OP has LIWC features (4,641 OPs) are exactly the paper's sample: 3,186 delta and 4,935 no-delta posts under the flair label. 5,629 OPs have the subreddit and entity files.

### 2. The `delta` field is a flair artifact
The field comes from the post flair (`[Deltas Awarded]`, `[∆(s) from OP]`, `[View Changed]`).

| | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---|---|---|---|---|
| share of posts with `delta` (flair), 8,121 posts | 0.0% | 0.1% | 48% | 58% | 65% |
| share with a DeltaBot confirmation, 8,121 posts | 53% | 55% | 62% | 65% | 76% |

- Per month the flair share is 0% up to 2015-02, 17% in 2015-03, 25% in 2015-04, 57% in 2015-05, 62% in 2015-06. A logistic regression on the posting time alone reaches test AUROC 0.815 with this label.
- In 2013-2014 the OP wrote a delta symbol in 836 of 2,858 posts (2013) and 1,304 of 3,793 posts (2014) of `posts_malleability` (all 16,150 posts), all but one with `delta = False`.
- In 2015-05 and later (8,433 posts, all OPs): 4,042 have `delta = True`, 4,391 `False`; 721 of the `False` posts contain an OP comment with a delta symbol. DeltaBot replied "Confirmed: 1 delta awarded" in 639 of them, only rejected in 22, and did not reply in 60.
  The confirmed misses carry other flairs that replace the delta flair: `FTFdeltaOP` 167, `Election` 108, none 102, `[OP ∆/Election]` 89, `[FreshTopicFriday]` 79.
- Of a random 60 posts with `delta = True`, 52 have a DeltaBot confirmation under an OP comment.
- The share of posts with a DeltaBot confirmation (all 16,150 posts) exists in every month from 2013-02: 27% (2013-02), 38% (2014-09), 47% (2015-05), 56% (2017-06).
- On the 8,121 posts: flair and DeltaBot both positive 2,987; flair only 199; DeltaBot only 2,192; both negative 2,743.
- **T1 uses the same evidence.** Of the 10,303 delta comments of `pairs.jsonl` found in `threads.jsonl`, 99.7% have an OP delta symbol and 95.7% a DeltaBot confirmation in their subtree; of the 9,028 no-delta comments found, 2.7% and 2.6%. The pairs cover 2013-2017 (845, 1,579, 2,202, 2,371, 3,306 per year), so they do not depend on the flair.

**Decision:** the label is "DeltaBot confirmed a delta that the OP awarded". Remaining noise: 331 of the 2,942 label-0 posts (11%, 4% of all posts) have an OP delta symbol without a confirmation (132 rejected, usually for a too short explanation; 199 got no bot reply).

### 3. Text that leaks the outcome
The stored posts are the crawled versions, written after the discussion. Share of posts that contain it (positive vs negative posts of the 8,121):

| in the post text | delta | no delta |
|---|---|---|
| "edit" / "update" | 35.3% | 23.2% |
| the moderator footnote (49 different texts, 57% of all posts) | 17.5% | 9.4% |
| the word "delta" | 7.3% | 2.2% |
| a delta symbol (∆, Δ, !delta; 55 posts) | 1.0% | 0.1% |

Bag-of-words logistic regression (test AUROC, same validation-based settings) with different clean-up rules:

| rule | characters removed | AUROC |
|---|---|---|
| none | 0% | 0.603 |
| symbols only | 0.0% | 0.603 |
| footnote only | 16.7% | 0.580 |
| footnote + symbols + the paragraph of each edit marker | 20.4% | 0.563 |
| footnote + symbols + everything after the first edit marker | 23.2% | 0.560 |

The rule used (`scripts/t5/t5_common.py`, `clean_text`): footnote, paragraphs that start with an edit/update marker, symbols. Only a paragraph that STARTS with the marker is removed ("Update the tax code ..." and "Editing is a skill ..." stay). It removes 20.8% of the characters, changes 72.7% of the posts, empties none; 71 unusual edit lines in 65 posts and 5 footnote variants remain.
Bag of words with the rule: 0.564. The post length alone: 0.560 without, 0.500 with the clean-up. Part of the text signal was the footnote and the edits ("Edit: my view has not been changed 180 degrees ..." gives the outcome away).

### 4. LIWC / Receptiviti
`liwc_wc` (words the vector was computed from) is 1,245 to 10,242 for 90% of the authors, median 6,393, maximum 14,316; 99.9% have more than 1,000 words. The paper says the first 1,000 words. The vector is one per author, so it can contain text written after the post it is used for. This cannot be checked with the released files.

### 5. `threads.jsonl` and what the OPs wrote earlier
- `threads.jsonl` has 65,169 posts (2013: 16,409; 2014: 17,641; 2015: 10,701; 2016: 10,646; 2017: 9,772). It contains all 16,150 posts of `posts_malleability` and all 5,990 submissions of `pairs`. Comments found vs Reddit's `num_comments`: median 1.00, at least 90% in every thread with comments; the file is described as "selected threads", the criterion is unknown. 18,026 posts (27.7%) have a deleted or unknown author; 28,721 known OPs remain.
- Posts per OP in the whole file: 76.8% of the OPs have one post (46.8% of the posts), 12.8% two, 4.3% three; 85 OPs with more than 20 posts wrote 6.7% of the posts (median 1, p90 3, maximum 112).
- Earlier posts of the same OP, per post (whole file): 0 for 60.9%, 1 for 14.1%, 2 for 6.3%, 3-4 for 6.3%, 5-9 for 5.8%, 10 or more for 6.5%; the OP's first post in the file is 71% of the posts in 2013 and 56-60% afterwards.

For the 8,121 T5 posts (earlier posts counted in `threads.jsonl`, strictly before the post):

| earlier posts of the OP | posts | share | share with a delta |
|---|---|---|---|
| 0 | 3,421 | 42.1% | 78.9% |
| 1 | 1,292 | 15.9% | 61.8% |
| 2 | 765 | 9.4% | 56.2% |
| 3-4 | 795 | 9.8% | 53.0% |
| 5-9 | 839 | 10.3% | 47.4% |
| 10 or more | 1,009 | 12.4% | 42.9% |

- By the OP's k-th post in the T5 data: 1st 74.9%, 2nd 52.5%, 3rd 48.7%, 4th 47.9%, 5th or later 45.2%. Of two consecutive posts of an OP at least 14 days apart (2,571 pairs) the next one has a delta in 55.4% if the previous had one (n = 1,209) and in 42.6% if not (n = 1,362). 68 OPs with ten or more posts wrote about 12.6% of the posts.
- **How much text the OPs wrote** (CMV only; ModernBERT tokens; own posts and comments in `threads.jsonl`). Over all 11,806 OPs and all time: median 21 comments (p90 176), median 2,717 tokens (p90 24,264); 58.6% have more than 2,048 and 23.9% more than 8,192 tokens. Before the first post that the OP has in `posts_malleability`: 50.0% of the OPs have no text at all, 47.8% at least one comment, 19.0% at least one own post, median 0 tokens (p75 1,015, p90 5,122).
  For the 8,121 T5 posts: 18.9% have no earlier text, 80.5% at least one earlier comment, median 3,680 tokens (p75 18,436), 58.5% more than 2,048 and 37.3% more than 8,192 tokens. The delta rate is 85.6% when the OP has no earlier CMV text and 58.7% when there is some.
- The files contain no raw text from other subreddits: only LIWC/Receptiviti scores, comment/post counts per subreddit (median 1,558 comments per author, p90 11,261) and entity lists (median 23,809 tokens if written out, 71.7% above 8,192).

## Notes
- The delta rate of the OPs' first posts (79%) is a strong, time-safe signal (counts of earlier posts), not yet used in the models of `RESULTS.md`.
- The corpus starts in 2013, so the posts of the first months have no earlier text by construction. This does not explain the 85.6%, because 2013 has a delta rate of 53%.
- Not verified: that every OP delta symbol is an attempt to award a delta; that `threads.jsonl` contains all CMV posts of 2013-2017; the cause of the missing flair before 2015.
- Not covered: history written on other subreddits (it is not in the files).
