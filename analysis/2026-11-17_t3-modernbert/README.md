# T3 ModernBERT: stance from a user's text

**Status:** trained on Gilbreth on 2026-11-17 (one run, one seed). The model is above random guessing on
every label, but not above the class-weighted TF-IDF baseline.

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
```

`jobs/train.sub` starts `scripts/t3/train.py` when the name is `t3_profiles` and passes the other
arguments through. `--push` uploads the model to the private repo
`nlp-final-project11/nlp-project-modernbert-t3_profiles`.

## Results

One run: seed 42, 3 epochs (1,383 steps), `--bf16`, Slurm job 12412873 (1h32m on one A100). Model:
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
- **One seed, one setting.** No other learning rate, number of epochs or token limit was tried.

## Notes

- **Size of the run.** 14,725 training, 1,831 validation and 1,841 test users (the same 18,397 users
  as the baselines). 461 steps per epoch, 1,383 in total. Documents have a median of 1,410 tokens and
  27% are cut at 4,096. A batch of four is padded to 3,586 tokens on average, so a step should cost
  about as much as a `ddo_switch` step (4.9 s). The estimate before the run was 2 to 2.5 hours with
  validation; the run took 1h32m, so it fits one 4-hour standby job.
- **A quarter of the users have more text than the model reads.** The TF-IDF model reads all of a
  user's text and the baselines found that more text helps, so this is one possible reason for the gap.
  It is not tested. Reading several windows per user and averaging would test it.
- **Not copied from the job log:** which checkpoint was selected and its validation score.
- **Checked without a GPU:** the loss equals a hand-written loop and ignores unknown labels; the
  metrics equal the baseline's on the same predictions; a run on made-up users picks the best
  checkpoint, saves the model and resumes from a checkpoint when started again.
- **Not in `RESULTS.md` yet.**
