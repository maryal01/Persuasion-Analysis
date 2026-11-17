# T3 inferred profiles: a profile for every user with text

**Status:** script written and checked on a laptop (2026-11-17). Not run yet: it needs the trained T3
model (`analysis/2026-11-17_t3-modernbert/`) and a GPU job.

## Goal

Give every DDO user who wrote something a profile, including the users who left theirs blank. The T2
text + audience model then reads these profiles for voters without a self-reported one, and the
self-reported and inferred profiles can be compared.

## What it does

`scripts/t3/infer_profiles.py` runs the trained T3 model over every user with at least one usable text
item and writes the table `ddo_profiles_inferred`, one row per user:

| Column | Content |
|---|---|
| `user` | DDO user name |
| the six label columns | the predicted class, with the class names of `ddo_users` |
| `<label>_probs` | the model's probability of each class |
| `tokens` | tokens of the user's text the model read |
| `in_training` | whether the model was trained on this user (a training user with a label) |

- **Same input as in training.** The text rule, the item order (seeded per user) and the cut at 4,096
  tokens are the training script's, so a user's document is the same as in training. Items from debates
  in T2's validation or test split are left out, so a profile never uses a debate that T2 is scored on.
- **Splits** are those of `ddo_users`. Users without text are not in the table.
- **`in_training`** matters for one comparison: for those users the predicted labels are not a fair
  test of the model. Profiles of users without any label, the ones this table is for, are never in
  training.
- It prints how often the predicted class equals the self-reported one, separately for held-out and
  for training users, and writes `outputs/t3_profiles_infer/{train,validation,test}.parquet` and
  `summary.json`. `--push` uploads the table to branch `t3` of the dataset repo (the staging branch).

## Run

```bash
python scripts/t3/infer_profiles.py --smoke --cpu --model <path of a trained model>   # local check
sbatch -A <account> jobs/train.sub t3_profiles_infer --bf16 --push                    # after the T3 run
```

By default the model is read from `nlp-final-project11/nlp-project-modernbert-t3_profiles`, where the
training run pushes it.

## Results

Not run yet. What the run will cover, measured without a model:

| | Users | Median tokens | Cut at 4,096 | Under 100 tokens |
|---|---|---|---|---|
| With text | 40,739 of 47,145 | | | |
| ... with at least one label | 18,397 | 1,404 | 26.5% | 7.6% |
| ... without any label | 22,342 | 336 | 7.2% | 22.6% |

- **The users this is for have much less text.** Half of the unlabelled users have under 336 tokens and
  almost a quarter have under 100. The baselines scored below the majority class for users with under
  100 words, so many of these profiles will be weak guesses.

## Notes

- **Keep the table private.** It holds stances the users did not report themselves, including the
  model's guess of their religion and politics. It belongs in the private dataset repo only.
- **Size of the run.** 54.6 million tokens to read, against 84.9 million in training (with a backward
  pass). An estimated 20 to 40 minutes on one GPU; not measured.
- **Checked without a GPU:** on made-up users, the profiles of validation and test users equal the
  predictions the training script saved for the same model (largest difference 2e-7), users without a
  label get a profile and users without text do not. On the real tables, a smoke run with an untrained
  stand-in model writes files with the right columns and class names.
- **The predicted class is always written,** however unsure the model is. The probabilities are in the
  table, so a confidence threshold can be tried later without running the model again.
