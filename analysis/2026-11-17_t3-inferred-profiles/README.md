# T3 inferred profiles: a profile for every user with text

**Status:** run on Gilbreth on 2026-11-19. The table `ddo_profiles_inferred` is on branch `t3` of the
private dataset repo. The profiles are weak: the predicted class equals the self-reported one for under
half of the held-out users, and as T2 audience input they lower the score.

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

Slurm job 12438104, with the model `nlp-final-project11/nlp-project-modernbert-t3_profiles` (one seed).
It wrote 40,739 profiles: 32,585 / 4,080 / 4,074 users in the train / validation / test split of
`ddo_users`, 14,725 of them users the model was trained on. The table is on branch `t3` of the dataset
repo (commit `cb1915a`).

How often the predicted class equals the self-reported one. Held-out users are the validation and test
users with that label:

| Label | Held-out users | Accuracy | Training users | Accuracy |
|---|---|---|---|---|
| economy | 2,391 | 0.431 | 9,472 | 0.500 |
| social_issues | 2,871 | 0.478 | 11,368 | 0.510 |
| personal_liberties | 2,548 | 0.463 | 10,183 | 0.489 |
| foreign_policy | 2,034 | 0.425 | 8,117 | 0.468 |
| religion | 2,737 | 0.491 | 11,018 | 0.542 |
| political_ideology_group | 2,270 | 0.313 | 9,017 | 0.352 |

- **Under half of the held-out profiles match the self-report** (0.31 to 0.49 per label). That agrees
  with the test table of the training run (`analysis/2026-11-17_t3-modernbert/`).
- **The model barely fits the users it was trained on.** Their accuracy is only 0.03 to 0.07 higher. So
  the model is not overfit; it has not learned its training users. Whether longer training or another
  learning rate changes that is not tested.
- **Small classes are predicted much more often than users report them.** Share of each class among
  the users who reported that label (all of `ddo_users`), and among the predictions for all 40,739
  users in the table:

  | Label | Class | Reported | Predicted |
  |---|---|---|---|
  | personal_liberties | restrictive | 14.6% | 44.4% |
  | political_ideology_group | socialist | 8.6% | 24.3% |
  | foreign_policy | mixed | 17.4% | 33.9% |
  | religion | other religion | 13.3% | 25.4% |
  | economy | mixed | 26.9% | 39.6% |

  The two groups are not the same people (most users in the table reported no label), so part of the
  gap could be real. The direction is what class weighting in training produces.
- **As T2 audience input the profiles lower the score.** The T2 text + audience model trained on
  self-reported profiles drops from AUROC 0.737 to 0.706 when it is given these profiles instead, below
  the text-only model (0.719). See `analysis/2026-11-17_t2-audience-input/`.

What the run covers, measured without a model:

| | Users | Median tokens | Cut at 4,096 | Under 100 tokens |
|---|---|---|---|---|
| With text | 40,739 of 47,145 | | | |
| ... with at least one label | 18,397 | 1,404 | 26.5% | 7.6% |
| ... without any label | 22,342 | 336 | 7.2% | 22.6% |

- **The users this is for have much less text.** Half of the unlabelled users have under 336 tokens and
  almost a quarter have under 100. The baselines scored below the majority class for users with under
  100 words. Their profiles cannot be checked, because these users reported nothing.

## Notes

- **Keep the table private.** It holds stances the users did not report themselves, including the
  model's guess of their religion and politics. It belongs in the private dataset repo only.
- **Size of the run.** 54.6 million tokens to read, against 84.9 million in training (with a backward
  pass). The estimate before the run was 20 to 40 minutes on one GPU; the job's duration is not copied
  here.
- **The table is on the staging branch `t3` only.** Whether it is copied to `main` is not decided. Given
  the accuracy above, these profiles should not be read as the users' stances.
- **Checked without a GPU:** on made-up users, the profiles of validation and test users equal the
  predictions the training script saved for the same model (largest difference 2e-7), users without a
  label get a profile and users without text do not. On the real tables, a smoke run with an untrained
  stand-in model writes files with the right columns and class names.
- **The predicted class is always written,** however unsure the model is. The probabilities are in the
  table, so a confidence threshold can be tried later without running the model again.
