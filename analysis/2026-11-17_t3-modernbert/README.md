# T3 ModernBERT: stance from a user's text

**Status:** training script written and checked on a laptop (2026-11-17). Not trained yet: the run needs
a GPU job on Gilbreth.

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
  and the printed test table with 95% intervals from a bootstrap over users.

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

Not run yet. The rows to beat are in `analysis/2026-11-16_t3-baselines/README.md`: accuracy against the
majority class and the unweighted TF-IDF model, macro-F1 against random guessing and the class-weighted
TF-IDF model. This script weights classes, so its fair comparison is the class-weighted TF-IDF row.

## Notes

- **Size of the run.** 14,725 training, 1,831 validation and 1,841 test users (the same 18,397 users
  as the baselines). 461 steps per epoch, 1,383 in total. Documents have a median of 1,410 tokens and
  27% are cut at 4,096. A batch of four is padded to 3,586 tokens on average, so a step should cost
  about as much as a `ddo_switch` step (4.9 s): an estimated 2 to 2.5 hours with validation, which fits
  one 4-hour standby job. The estimate is not measured on a GPU.
- **A quarter of the users have more text than the model reads.** The baselines found that more text
  helps, so reading several windows per user and averaging is the obvious next step if the single
  window is not enough.
- **Checked without a GPU:** the loss equals a hand-written loop and ignores unknown labels; the
  metrics equal the baseline's on the same predictions; a run on made-up users picks the best
  checkpoint, saves the model and resumes from a checkpoint when started again.
- **Not in `RESULTS.md` yet.** The T3 rows go in after the run.
