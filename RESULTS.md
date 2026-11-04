# Results

Test-set results for each trained model. Each row is one training run; add new rows as runs finish.

## T1: CMV persuasion (`cmv`)

Test set: 1,019 delta/no-delta pairs (2,038 replies) from posts that never appear in training
(dataset branch `t1-t2`). The main metric is **pairwise accuracy**: how often the model scores the
delta reply above its no-delta partner. Chance is 50%.

| Model | Pairwise acc. | AUROC | Accuracy | F1 (delta) | Macro-F1 |
|---|---|---|---|---|---|
| Coin flip | 50.0% | 0.500 | – | – | – |
| More comments wins | 53.7% | – | – | – | – |
| More links wins | 54.3% | – | – | – | – |
| Longer reply wins | 55.3% | – | – | – | – |
| **ModernBERT-base, text only** | **66.1%** | **0.629** | 55.9% | 0.645 | 0.532 |

- Model: [`nlp-final-project11/nlp-project-modernbert-cmv`](https://huggingface.co/nlp-final-project11/nlp-project-modernbert-cmv)
  (private). Trained on Gilbreth (one A100, ~1h45m, Slurm job 12026112), 3 epochs, max 4,096 tokens.
- With 1,019 pairs, pairwise accuracy has a rough 95% interval of ±3 points (about 63–69%).
- Single-reply accuracy is low because the model labels 74% of replies "delta" (precision 0.54,
  recall 0.80). Its ranking is better than its yes/no cutoff, which is why the pairwise score is higher.
- No delta markers ("Δ", "!delta") leak into the replies: they appear in under 0.5% of replies on either side.

**Caveats**
- One run, one random seed. Final numbers should average a few seeds.
- This run picked its checkpoint by validation F1 (step 1000, validation AUROC 0.630). Later
  checkpoints had higher validation AUROC (0.644). The code now selects by AUROC.
- Text only. Audience features (RQ2) haven't been added yet.

**Prior work.** Al-Khatib et al. (ACL 2020), who built Webis-CMV-20, report per-comment accuracy
rather than pairwise: word count 51.9%, Tan et al.'s best text features 57.8%, their model with
debater traits 61.1%. Their metric differs from ours, so don't compare the numbers directly.
Tan et al. (2016)'s 65.1% is on a different corpus.

## T2: DDO voter switching (`ddo_switch`, `ddo_undecided`)

Not trained yet.

## T5: Does the author of a CMV post award a delta?

**Task.** For a post on r/changemyview, predict whether its author (the OP) gives at least one delta (∆) to a commenter, i.e. admits that their view changed.

**Data.** 8,121 posts from the `posts_malleability` file of Webis-CMV-20, 2013-2017. Only authors with LIWC/Receptiviti features are kept (4,641 authors); this is the same sample as in Al-Khatib et al. (2020).
Split by author, so one author is never in two sets: train 6,626 posts, validation 725, test 770. Share of posts where a delta was awarded: 63% / 67% / 67%.

**Label.** 1 if DeltaBot confirmed a delta that the author gave ("Confirmed: 1 delta awarded" under a comment of the author). T1 relies on the same kind of evidence.
We do not use the `delta` field of the dataset: it comes from the post flair, which does not exist before March 2015 and is replaced by other flairs (Fresh Topic Friday, Election, ...) in some threads.
On our 8,121 posts it misses 2,192 of the 5,179 posts that DeltaBot confirmed.
The label is still a bit noisy: in 331 posts (4%) the author wrote ∆ but DeltaBot did not confirm it (132 were rejected, usually because the explanation was too short), and these are labeled 0.
The share of delta posts grows over the years (53% in 2013, 76% in 2017), so the posting date is included as a baseline.

**Post text.** Before the models see it, we remove three things that are not part of the original argument: the moderator footnote that is added to threads, paragraphs the author added later that start with
`Edit:` or `Update:`, and delta symbols (`scripts/t5/t5_common.py`, `clean_text`). This removes 20.8% of the characters. A few unusual edit lines (71 in 65 posts) remain.

**Models.** Logistic regression (a simple linear model) on a CPU, `scripts/t5/baselines.py`. The settings are chosen on the validation set. There is no ModernBERT result yet.

**Results on the test set (770 posts).**
AUROC: 0.5 means guessing, 1.0 means perfect. The interval is a 95% bootstrap interval over authors. PR-AUC of a model that knows nothing is 0.668 (the share of delta posts in the test set).
Balanced accuracy and macro-F1 are given instead of the F1 of the delta class, because "always delta" already gets a high F1 (delta posts are the majority).

| Model | AUROC | 95% interval | PR-AUC | Balanced acc. | Macro-F1 |
|---|---|---|---|---|---|
| Always "delta" | 0.500 | – | 0.668 | 0.500 | 0.400 |
| Posting date only | 0.589 | 0.543–0.634 | 0.731 | 0.553 | 0.540 |
| Post length only | 0.500 | – | 0.668 | 0.500 | 0.400 |
| Words of the post (bag of words) | 0.564 | 0.520–0.610 | 0.720 | 0.541 | 0.537 |
| LIWC of the author (93 numbers) | 0.705 | 0.660–0.748 | 0.829 | 0.660 | 0.605 |
| Receptiviti of the author (118 numbers) | **0.726** | 0.681–0.769 | 0.841 | 0.677 | 0.647 |
| LIWC + Receptiviti | 0.718 | 0.672–0.766 | 0.835 | 0.667 | 0.611 |
| Words + LIWC + Receptiviti | 0.718 | 0.672–0.766 | 0.835 | 0.667 | 0.611 |

**What this tells us.**
- The author's traits (LIWC, Receptiviti) predict a delta clearly better (AUROC about 0.71-0.73) than the post text (0.56) or the date (0.59).
- Post length has no signal. For the words, the best model on the validation set ignores all of them, so "Words + LIWC + Receptiviti" is the same as "LIWC + Receptiviti".
- With 770 test posts the interval is about ±0.045, so LIWC, Receptiviti and both together cannot be ranked against each other.
- Al-Khatib et al. (2020) report 0.52 for words and 0.62 for personality traits with the flair label and a different test set, so the numbers cannot be compared directly.

**Cautions.**
- LIWC and Receptiviti are computed once per author from a lot of text (median 6,393 words, the paper says the first 1,000). This text can include posts written after the post we predict. We cannot check this with the released files, so the author results may be too optimistic.
- The label noise above (4% of the posts) lowers all scores a little.
- If the cleaning is not used, the words and the post length look better (words 0.603, length 0.560) because the footnote and the edits give away part of the outcome.
