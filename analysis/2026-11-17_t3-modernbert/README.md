# T3 ModernBERT: stance from a user's text

**Status:** trained on Gilbreth with three seeds (42 on 2026-11-17; 43 and 44 finished on 2026-11-20).
The model is above random guessing on every label, but not above the class-weighted TF-IDF baseline.
The three seeds agree: macro-F1 averaged over the six labels is 0.395 ± 0.004, against 0.416 for TF-IDF.

## Goal

Predict a DDO user's self-reported stance from what they wrote, with a fine-tuned ModernBERT, and
compare it with the majority, random and TF-IDF rows of `analysis/2026-11-16_t3-baselines/`. The trained
model is also what later writes profiles for voters who left theirs blank (for the T2 text + audience
model).

## What it does

`scripts/t3/train.py` fine-tunes one ModernBERT-base for all six label columns of `ddo_users` (economy,
social issues, personal liberties, foreign policy, religion, political ideology group):

- **Same users and text rule as the baselines.** Users with at least one label and at least one text
  item; items from debates in T2's validation or test split are left out; votes are not used. The
  loading code is shared with `scripts/t3/baselines.py`.
- **One document per user.** The user's items go in a random order (seeded per user) and the document
  is cut to 4,096 tokens, so the model reads a sample of a long history, not its first items.
- **One classifier with a block of logits per label column** (3 + 3 + 3 + 3 + 4 + 5 = 21). The loss is
  the cross-entropy of every label the user has, so a user with only some labels still trains the
  model. Classes are weighted by inverse frequency within each label column (`--no-class-weights`
  turns that off).
- **Same training settings as the T1/T2 script:** batch 4 with 8 accumulation steps, 3 epochs, learning
  rate 2e-5, checkpoints that a later job resumes from. The best checkpoint is the one with the highest
  validation macro-F1 averaged over the six label columns; validation runs twice per epoch.
- **Outputs** in `outputs/t3_profiles/`: `predictions_validation.parquet` and `predictions_test.parquet`
  (per user and label column: true class, predicted class, class probabilities), `test_metrics.json`,
  and the printed test table with 95% intervals from a bootstrap over users. The prediction files
  hold user names; they are written after the model is pushed, so they are not uploaded with it.

## Run

```bash
python scripts/t3/train.py --smoke                                 # about 1 minute, also works without a GPU (--cpu)
sbatch -A <account> jobs/train.sub t3_profiles --smoke --bf16      # the same check on Gilbreth
sbatch -A <account> jobs/train.sub t3_profiles --bf16 --push       # the real run
sbatch -A <account> jobs/train.sub t3_profiles --seed 43 --bf16 --push   # another seed, in its own folder and repo
```

`jobs/train.sub` starts `scripts/t3/train.py` when the name is `t3_profiles` and passes the other
arguments through. `--push` uploads the model to the private repo
`nlp-final-project11/nlp-project-modernbert-t3_profiles`. A run with another seed than 42 is named
`t3_profiles-seed<N>`: its own `outputs/` folder, markers and model repo. Seeds 43 and 44 were in the
end run without `--push` (see Notes).

## Results

The first run: seed 42, 3 epochs (1,383 steps), `--bf16`, Slurm job 12412873 (1h32m on one A100). Model:
`nlp-final-project11/nlp-project-modernbert-t3_profiles` (private). The numbers are for the test users,
with 95% intervals from a bootstrap over users. The random, majority and TF-IDF columns are copied from
`analysis/2026-11-16_t3-baselines/README.md` and are for the same test users.

Macro-F1. This script weights classes, so its fair comparison is the class-weighted TF-IDF model:

| Label | Test users | Random | TF-IDF, class-weighted | ModernBERT |
|---|---|---|---|---|
| economy | 1,164 | 0.333 | 0.418 (0.391–0.447) | 0.411 (0.381–0.439) |
| social_issues | 1,427 | 0.333 | 0.462 (0.439–0.488) | 0.432 (0.406–0.456) |
| personal_liberties | 1,271 | 0.333 | 0.450 (0.420–0.480) | 0.407 (0.381–0.432) |
| foreign_policy | 991 | 0.333 | 0.424 (0.392–0.455) | 0.402 (0.371–0.430) |
| religion | 1,374 | 0.250 | 0.395 (0.370–0.421) | 0.393 (0.369–0.418) |
| political_ideology_group | 1,122 | 0.200 | 0.345 (0.316–0.376) | 0.302 (0.274–0.329) |
| mean of the six | | | 0.416 | 0.391 |

Accuracy:

| Label | Majority | TF-IDF, class-weighted | ModernBERT | ModernBERT minus majority |
|---|---|---|---|---|
| economy | 0.497 | 0.448 | 0.420 (0.392–0.448) | −0.076 (−0.120 to −0.036) |
| social_issues | 0.434 | 0.483 | 0.459 (0.434–0.486) | +0.025 (−0.006 to +0.060) |
| personal_liberties | 0.505 | 0.512 | 0.452 (0.424–0.478) | −0.054 (−0.087 to −0.022) |
| foreign_policy | 0.571 | 0.484 | 0.423 (0.390–0.453) | −0.148 (−0.194 to −0.103) |
| religion | 0.468 | 0.539 | 0.492 (0.467–0.517) | +0.024 (−0.011 to +0.059) |
| political_ideology_group | 0.325 | 0.359 | 0.310 (0.283–0.337) | −0.015 (−0.056 to +0.025) |

- **The model learned something, but not more than TF-IDF.** Macro-F1 is 0.07 to 0.14 above random
  guessing on every label. Against the class-weighted TF-IDF model it is within 0.01 on economy and
  religion and 0.02 to 0.04 lower on the other four. The intervals of the two models overlap on every
  label and the two were not compared on the same resamples, so the run supports "no better than
  TF-IDF", not "worse".
- **Accuracy is below the majority class on three labels** (economy, personal liberties, foreign
  policy) and not clearly different from it on the other three. Weighting classes trades accuracy for
  the small classes. The class-weighted TF-IDF model pays less for that: its accuracy is 0.02 to 0.06
  above ModernBERT's on every label.
- **One setting.** No other learning rate, number of epochs or token limit was tried.

### Three seeds

Seeds 43 and 44 have the settings of the first run and the same 1,841 test users. Test macro-F1 per
seed, with the mean ± the standard deviation between the three seeds. The TF-IDF column is the one
above.

| Label | TF-IDF, class-weighted | Seed 42 | Seed 43 | Seed 44 | ModernBERT, three seeds |
|---|---|---|---|---|---|
| economy | 0.418 | 0.411 | 0.417 | 0.413 | 0.413 ± 0.003 |
| social_issues | 0.462 | 0.432 | 0.436 | 0.448 | 0.439 ± 0.009 |
| personal_liberties | 0.450 | 0.407 | 0.429 | 0.412 | 0.416 ± 0.012 |
| foreign_policy | 0.424 | 0.402 | 0.416 | 0.416 | 0.411 ± 0.008 |
| religion | 0.395 | 0.393 | 0.388 | 0.403 | 0.395 ± 0.007 |
| political_ideology_group | 0.345 | 0.302 | 0.296 | 0.291 | 0.297 ± 0.005 |
| mean of the six | 0.416 | 0.391 | 0.397 | 0.397 | 0.395 ± 0.004 |

Accuracy:

| Label | Majority | TF-IDF, class-weighted | Seed 42 | Seed 43 | Seed 44 | ModernBERT, three seeds | Minus majority |
|---|---|---|---|---|---|---|---|
| economy | 0.497 | 0.448 | 0.420 | 0.435 | 0.444 | 0.433 ± 0.012 | −0.064 |
| social_issues | 0.434 | 0.483 | 0.459 | 0.481 | 0.477 | 0.473 ± 0.012 | +0.039 |
| personal_liberties | 0.505 | 0.512 | 0.452 | 0.488 | 0.493 | 0.478 ± 0.023 | −0.028 |
| foreign_policy | 0.571 | 0.484 | 0.423 | 0.452 | 0.449 | 0.441 ± 0.016 | −0.130 |
| religion | 0.468 | 0.539 | 0.492 | 0.488 | 0.507 | 0.496 ± 0.010 | +0.028 |
| political_ideology_group | 0.325 | 0.359 | 0.310 | 0.301 | 0.298 | 0.303 ± 0.006 | −0.022 |

- **The seeds agree.** Macro-F1 differs by at most 0.022 between seeds on a label, and the mean of
  the six labels is 0.391, 0.397 and 0.397.
- **Still not above TF-IDF.** Averaged over the seeds, macro-F1 equals TF-IDF's on religion, is 0.005
  lower on economy and 0.013 to 0.048 lower on the other four labels. Every seed is below TF-IDF on
  five labels; on religion one seed of three is above it.
- **More seeds do not settle the comparison with TF-IDF.** The seeds differ by less (standard
  deviation at most 0.012) than the interval of one run is wide (about ±0.03, from the test users). A
  paired test of the two models on the same test users would settle it. It was not run: the
  prediction files of seeds 43 and 44 are on Gilbreth.
- **Accuracy is higher with seeds 43 and 44 on four labels** (by 0.015 to 0.041 on economy, social
  issues, personal liberties and foreign policy), and the reading stays. In every seed accuracy is
  below the majority class on economy and foreign policy, with intervals that exclude 0, and below
  the class-weighted TF-IDF model on every label. With an interval that excludes 0 it is above the
  majority class on social issues in two seeds and on religion in one.

## Notes

- **What the seed controls** (`--seed`, added 2026-11-19 for runs with other seeds): the start weights
  of the classifier on top of the encoder, the order of the training users, and the order of each
  user's items. For a user with more text than the model reads, another seed therefore reads another
  part of their history. In the seed-42 run above the classifier's start weights were not seeded, so
  that run cannot be repeated exactly. Checked on made-up users: the same seed twice gives identical
  predictions, another seed gives different ones, no `--seed` equals `--seed 42`, and `jobs/train.sub`
  names the run as the script does in 7 argument forms.
- **Size of the run.** 14,725 training, 1,831 validation and 1,841 test users (the same 18,397 users
  as the baselines). 461 steps per epoch, 1,383 in total. Documents have a median of 1,410 tokens and
  27% are cut at 4,096. A batch of four is padded to 3,586 tokens on average, so a step should cost
  about as much as a `ddo_switch` step (4.9 s). The estimate before the run was 2 to 2.5 hours with
  validation; the run took 1h32m, so it fits one 4-hour standby job.
- **A quarter of the users have more text than the model reads.** The TF-IDF model reads all of a
  user's text and the baselines found that more text helps, so this is one possible reason for the gap.
  It is not tested. Reading several windows per user and averaging would test it.
- **Where the models of seeds 43 and 44 are.** They first ran with `--push`; training finished, and
  the upload at the end failed because the org's private storage on the Hub was full (a run with
  `--push` uploads the weights at every checkpoint). Started again without `--push`, each run
  continued from its last checkpoint and wrote its results. The two models are in
  `outputs/t3_profiles-seed43/` and `outputs/t3_profiles-seed44/` on Gilbreth scratch, which is purged
  after 60 days, and not on the Hub. The model that writes the inferred profiles is the one of seed
  42, which is on the Hub.
- **Not copied from the job logs:** which checkpoint each run selected, its validation score, and the
  job ids and run times of seeds 43 and 44.
- **Checked without a GPU:** the loss equals a hand-written loop and ignores unknown labels; the
  metrics equal the baseline's on the same predictions; a run on made-up users picks the best
  checkpoint, saves the model and resumes from a checkpoint when started again.
- **Not in `RESULTS.md` yet.**
