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

### `ddo_switch`: did a decided voter switch sides?

Test set: 5,845 votes from 2,504 debates that never appear in training (dataset branch `t1-t2`).
Each row is one audience member who held a side before the debate; 368 of them (6.3%) ended on the
other side. The main metric is **AUROC**: how often the model scores a voter who switched above one
who stayed. Chance is 0.500.

| Model | AUROC | F1 (switched) | Precision | Recall | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|
| Always "stayed" | 0.500 | 0.000 | – | 0.000 | 0.484 | 93.7% |
| Side before the debate | 0.586 | – | – | – | – | – |
| Year of the debate | 0.609 | – | – | – | – | – |
| Side before × year | 0.640 | – | – | – | – | – |
| **ModernBERT-base, text only** | **0.719** | 0.217 | 0.138 | 0.511 | 0.540 | 76.8% |

- Model: [`nlp-final-project11/nlp-project-modernbert-ddo_switch`](https://huggingface.co/nlp-final-project11/nlp-project-modernbert-ddo_switch)
  (private). Seed 42, 3 epochs (4,545 steps), max 4,096 tokens, bf16 mixed precision (`--bf16`; the
  T1 run used fp32). Trained on Gilbreth (one A100) in two chained standby jobs, about 7h10m in total:
  Slurm job 12297805 hit the 4-hour limit at step 2,833, and job 12297806 resumed from checkpoint 2500
  and finished in 3h09m.
- Checkpoint: step 4000, the highest validation AUROC (0.705). Validation AUROC was 0.698 at step 2500
  and stayed between 0.695 and 0.705 after that.
- With 368 switchers, AUROC has a rough 95% interval of ±0.03 (about 0.69–0.75). Voters of the same
  debate share a transcript, so the true interval is wider.
- The three middle rows use no text. Each scores a vote with the training-set switch rate of its
  group. Voters who started on Pro switched more often than those on Con (8.3% against 5.0%), and
  votes in 2007–2010 debates switched about 8–9% of the time, against 4–5% in 2011–2015.
- Accuracy is below "always stayed" because the loss weights the rare class up. The model flags 1,366
  of the 5,845 votes as switches. 188 of those are real (precision 13.8%, against a base rate of
  6.3%), out of 368 switchers (recall 51.1%).

**Caveats**
- One run, one random seed.
- Part of the score doesn't come from the arguments: side and year alone reach 0.640. The model is
  given the side, and it may also pick up the era from the transcript (not tested).
- Every voter with the same side in a debate gets the same input: the 5,845 test rows have 3,160
  distinct inputs. The model can tell debates apart, not voters. Audience features (RQ2, T3) are
  meant to add that.
- Inputs are cut at 4,096 tokens, while the proposal planned 8k. About a third of the debates are
  longer than the cut (34% by an earlier count, not re-measured for this run).
- The model card on the Hub is generated automatically. It reports the last validation step, not the
  test set, and its results table has shifted columns from step 3000 on (after the resume). Use the
  numbers here.

### `ddo_undecided`: which side does a tied voter end on?

Test set: 7,091 votes from 3,649 debates that never appear in training (dataset branch `t1-t2`,
commit `7b4add9`). Each row is one audience member recorded as tied before the debate; 6,508 of them
(91.8%) are still tied after it, 256 end on Pro and 327 on Con. Tied votes cast before about
September 2008 are left out (see "Old-form votes" below). The main metric is **macro-F1** over the
three classes.

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Random guess | 0.212 | 33.3% |
| Always "tie" | 0.319 | 91.8% |
| Year of the debate | 0.377 | 77.9% |
| **ModernBERT-base, text only** | **0.360** | 64.4% |

- **The model does not beat a rule that uses only the year of the debate.** It is 0.04 above always
  answering "tie" and 0.02 below the year rule. The year rule uses no text and the same class
  weighting as the model: it predicts the class that is most over-represented in that year in the
  training set. The year matters because 22% of tied voters left the tie in 2009 debates, against 3%
  in 2015 ones.
- Model: [`nlp-final-project11/nlp-project-modernbert-ddo_undecided`](https://huggingface.co/nlp-final-project11/nlp-project-modernbert-ddo_undecided)
  (private). Seed 42, 3 epochs (5,454 steps), max 4,096 tokens, bf16 mixed precision (`--bf16`),
  class weights tie 0.36, pro 8.86, con 6.95. Trained on Gilbreth (one A100) in two chained standby
  jobs, about 7h00m in total: Slurm job 12342062 hit the 4-hour limit, and job 12342063 resumed from
  checkpoint 3000 and finished in 3h00m.
- Checkpoint: step 5454, the last one, which had the highest validation macro-F1 (0.345).
- The script reports only accuracy and macro-F1 for three-class configs, so there are no per-class
  numbers. The two reported numbers put the F1 for `tie` between 0.76 and 0.82, which leaves `pro`
  and `con` at about 0.13 to 0.16 each on average.
- Accuracy is far below "always tie" because the loss weights the rare classes up: at least 1,944
  voters who stayed tied (30%) are predicted to pick a side.

**Old-form votes.** Earlier builds of this config included tied votes cast before about September
2008. The corpus dates 49,910 votes as cast "10 years ago", and 98.9% of them have one fixed pattern:
tied before and after, one debater marked on "Made more convincing arguments", everything else tied.
Voters active in both periods use that pattern in 97% of their early votes and 5% of their later
ones. This fits an older voting form without the before/after questions, but the corpus readme
doesn't say so. Those votes were 49,455 of the 121,865 rows, and 26 of them left the tie.
`ddo_switch` has 446 votes from that period and keeps them, because those voters marked a side.

A first run on the earlier build (dataset commit `e236e8d`, model commit `8b027e5`, Slurm jobs
12309143 to 12309146) scored macro-F1 0.375 and accuracy 82.6% on 12,073 test votes, against 0.325
and 95.1% for always "tie". Its test set is different, so the two runs can't be compared directly.

**Caveats**
- One run, one random seed. The comparisons rest on the 583 test voters who left the tie.
- Every tied voter in a debate gets the same input, so the model gives them all the same answer: the
  7,091 test rows have 3,649 distinct inputs, and in 347 debates the tied voters end differently.
- A later "tied before" can still be a skipped answer, and that can't be told apart from a real tie.
- Inputs are cut at 4,096 tokens, while the proposal planned 8k.
