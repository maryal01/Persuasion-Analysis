# T2 text + audience with filled profiles: inferred labels where the voter reported none

**Status:** run on 2026-11-21 with three seeds (42, 43, 44). With filled profiles the text + audience
model reaches AUROC 0.764 on the `ddo_switch` test votes, against 0.744 with self-reported profiles
only: +0.020 (95% interval +0.007 to +0.030, permutation test p = 0.0015). The gain is on the voters
who reported no profile (0.654 to 0.728, in every seed). On the voters who reported one, the two
models are level.

## Goal

The proposal uses the T3 model to give a profile to the DDO voters who left theirs blank, and says we
"report results with and without inferred profiles". So far there are two results: the text + audience
model with self-reported profiles (AUROC 0.744 over three seeds), and the same model given inferred
profiles for every voter at test time, which lowers its score (0.737 to 0.706, seed 42;
`analysis/2026-11-19_t2-compare-models/`). The second replaces profiles the voters reported themselves.
This run is the intended use: reported labels stay, and the T3 model's labels go only where a voter
reported nothing, in training and at test time.

## What it does

`--audience filled` in `scripts/train_delta.py` writes each voter's profile line from two tables
(`scripts/common/audience.py`): the label the voter reported (`ddo_users`) where there is one, and
otherwise the label the T3 model predicted from the voter's text (`ddo_profiles_inferred`, on branch
`t3` of the dataset repo, so the run needs `--audience-revision t3`). Nothing else changes: the line
sits after the voter's side and in front of the debate, as for `--audience self`. The run is named
`ddo_switch-audience-filled`, with its own `outputs/` folder and markers.

- **A reported label always wins.** A predicted label is used only for a label the voter left out.
- **Partly filled profiles are completed too,** not only blank ones. A voter who reported some of the
  six labels gets the missing ones from the T3 model.
- **A voter without text has no prediction** and keeps the labels they reported, or
  `Voter profile: unknown`.

`scripts/t1_t2/filled_subsets.py` scores saved models separately on the test votes of voters with no
reported profile (the group the proposal names) and of voters with one, and inside each on the votes
whose profile line changes with filled profiles and on those where it stays. The table, the intervals
and the paired tests are the functions of `scripts/t1_t2/compare_models.py`. Nothing is trained, and no
voter name is printed or written.

## Run

Coverage, on a laptop (reads the two tables from the Hub; changes nothing):

```bash
python scripts/common/audience.py --dataset-config ddo_switch --audience filled --audience-revision t3
```

On Gilbreth, a smoke run first and the training run behind it in chained standby parts (about 7 hours
in two parts for the runs with self-reported profiles; a third part is a spare and exits at once when
the run is finished). Without `--push`, the model stays in `outputs/ddo_switch-audience-filled/`:

```bash
smoke=$(sbatch --parsable -A <account> jobs/train.sub ddo_switch --audience filled --audience-revision t3 --smoke --bf16)
jid=$(sbatch --parsable -A <account> --dependency=afterok:$smoke jobs/train.sub ddo_switch --audience filled --audience-revision t3 --bf16)
jid=$(sbatch --parsable -A <account> --dependency=afterany:$jid jobs/train.sub ddo_switch --audience filled --audience-revision t3 --bf16)
jid=$(sbatch --parsable -A <account> --dependency=afterany:$jid jobs/train.sub ddo_switch --audience filled --audience-revision t3 --bf16)
```

Seeds 43 and 44 are the same commands with `--seed 43` or `--seed 44` added (runs
`ddo_switch-audience-filled-seed43` and `-seed44`).

Afterwards, with the prediction files copied to `outputs/<run>/`. Both commands read from the private
dataset repo, so they need the Hugging Face login:

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text + audience=outputs/ddo_switch-audience-self,outputs/ddo_switch-audience-self-seed43,outputs/ddo_switch-audience-self-seed44" \
    --model "Text + audience, filled profiles=outputs/ddo_switch-audience-filled,outputs/ddo_switch-audience-filled-seed43,outputs/ddo_switch-audience-filled-seed44" \
    --model "Text only=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44" \
    --compare "Text + audience, filled profiles" "Text only" \
    --output outputs/compare_models_filled.json

python scripts/t1_t2/filled_subsets.py --audience-revision t3 \
    --model "Text + audience=outputs/ddo_switch-audience-self,outputs/ddo_switch-audience-self-seed43,outputs/ddo_switch-audience-self-seed44" \
    --model "Text + audience, filled profiles=outputs/ddo_switch-audience-filled,outputs/ddo_switch-audience-filled-seed43,outputs/ddo_switch-audience-filled-seed44" \
    --model "Text only=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44" \
    --output outputs/filled_subsets.json
```

One seed alone: the first command with one folder per model (seed 42: the folders without a suffix).

## Results

### On all test votes

5,845 test votes from 2,504 debates, 368 of them switched. Three seeds per model. AUROC is the average
over the seeds; its 95% interval is from a bootstrap over debates. F1, precision and recall are those of
the switched class at the model's own decision, averaged over the seeds.

| Model | Validation AUROC | Test AUROC | 95% interval | Per seed (42, 43, 44) | F1 switched | Precision | Recall | Same-debate pairs |
|---|---|---|---|---|---|---|---|---|
| Text only | 0.699 | 0.708 | 0.679–0.736 | 0.719, 0.697, 0.709 | 0.200 | 0.122 | 0.579 | 48.7% |
| Text + audience, self-reported profiles | 0.729 | 0.744 | 0.716–0.767 | 0.737, 0.746, 0.749 | 0.241 | 0.158 | 0.504 | 56.4% |
| Text + audience, filled profiles | 0.745 | **0.764** | 0.736–0.788 | 0.776, 0.744, 0.771 | 0.253 | 0.164 | 0.557 | 61.2% |

Same-debate pairs: 801 pairs of a switcher and a stayer from one debate with the same side before; the
share in which the switcher scores higher.

| Comparison (first minus second) | AUROC difference | 95% interval | Permutation test |
|---|---|---|---|
| Filled profiles − self-reported profiles | +0.020 | +0.007 to +0.030 | p = 0.0015 |
| Filled profiles − text only | +0.055 | +0.039 to +0.070 | p = 0.0005 |
| Self-reported profiles − text only | +0.036 | +0.019 to +0.053 | p = 0.0005 |

Filled against self-reported profiles, one seed at a time:

| Seed | Self-reported | Filled | Difference | 95% interval | Permutation test |
|---|---|---|---|---|---|
| 42 | 0.737 | 0.776 | +0.039 | +0.020 to +0.055 | p = 0.0010 |
| 43 | 0.746 | 0.744 | −0.002 | −0.016 to +0.013 | p = 0.83 |
| 44 | 0.749 | 0.771 | +0.022 | +0.005 to +0.036 | p = 0.0065 |

### By what the filled profile changed for the voter

Test AUROC inside each group of votes, three seeds per model (`filled_subsets.py`). A score inside a
group ranks the votes of that group only, so the rows do not add up to the table above.

| Test votes | Votes | Switched | Text only | Self-reported | Filled | Filled − self-reported | Per seed | Permutation test |
|---|---|---|---|---|---|---|---|---|
| **No reported profile** | 1,267 (21.7%) | 103 (8.1%) | 0.671 | 0.654 | **0.728** | +0.074 (+0.045 to +0.105) | +0.106, +0.053, +0.064 | p = 0.0005 |
| – blank, now inferred | 1,024 (17.5%) | 72 (7.0%) | 0.675 | 0.651 | 0.728 | +0.077 (+0.042 to +0.113) | +0.114, +0.046, +0.072 | p = 0.0005 |
| – still unknown | 243 (4.2%) | 31 (12.8%) | 0.630 | 0.639 | 0.691 | +0.053 (−0.004 to +0.111) | +0.075, +0.053, +0.030 | p = 0.097 |
| **Some reported profile** | 4,578 (78.3%) | 265 (5.8%) | 0.719 | 0.765 | 0.772 | +0.006 (−0.005 to +0.017) | +0.021, −0.015, +0.012 | p = 0.30 |
| – some labels added | 1,658 (28.4%) | 106 (6.4%) | 0.708 | 0.724 | 0.735 | +0.011 (−0.011 to +0.034) | +0.025, −0.007, +0.014 | p = 0.35 |
| – unchanged | 2,920 (50.0%) | 159 (5.4%) | 0.724 | 0.785 | 0.792 | +0.007 (−0.005 to +0.020) | +0.028, −0.018, +0.011 | p = 0.29 |

### What the tables say

- **Filled profiles score higher than self-reported profiles alone:** 0.764 against 0.744, and 0.055
  above text only. The validation scores move the same way (0.745 against 0.729), and so does the
  same-debate comparison, in every seed (62.0%, 60.2%, 61.5% against 58.1%, 54.4%, 56.7%; no interval
  was computed for it).
- **The gain is on the voters the proposal names, those who reported no profile.** With self-reported
  profiles the model has nothing to read for them and scores no better than text only (0.654 against
  0.671; text only minus self-reported +0.017, p = 0.16). With an inferred profile it scores 0.728,
  and the difference is positive in every seed.
- **Completing a partly reported profile changes nothing measurable** (+0.011, p = 0.35), and neither
  does the new training on the votes whose line is the same (+0.007, p = 0.29).
- **The overall gain is not the same in every seed:** +0.039, −0.002, +0.022. Seed 43 with filled
  profiles is the weakest of the three runs on validation too (0.729). The interval and the test
  describe the test votes for these three runs per model; they do not cover the spread between seeds,
  which is larger with filled profiles (standard deviation 0.017 against 0.006).
- **A voter without a reported profile is still harder to predict** than one with a profile (0.728
  against 0.772 with filled profiles).
- **Together with the earlier control:** replacing reported profiles by inferred ones at test time
  lowers the score (0.737 to 0.706, seed 42, `analysis/2026-11-19_t2-compare-models/`); adding inferred
  ones where nothing was reported raises it. An inferred profile is worth less than a reported one and
  more than none.

### Coverage

Measured without a model, on the real `ddo_switch` votes: how the profile line of a vote changes from
self-reported to filled profiles.

| Split | Votes | Known profile, self | Known profile, filled | Line unchanged | Blank, now inferred | Some labels added | Still unknown |
|---|---|---|---|---|---|---|---|
| train | 48,479 | 79.1% | 95.8% | 54.9% | 16.8% | 28.3% | 4.2% |
| validation | 6,085 | 79.2% | 95.9% | 55.6% | 16.7% | 27.7% | 4.1% |
| test | 5,845 | 78.3% | 95.8% | 54.1% | 17.5% | 28.4% | 4.2% |

- **The line changes on 45.9% of the test votes,** not only on the fifth without any profile: 17.5%
  go from `Voter profile: unknown` to an inferred profile, and 28.4% have a partly reported profile
  that gets its missing labels added. All six labels are then known on 94.9% of the test votes,
  against 49.0% with self-reported profiles.
- **4.2% of the votes stay without a profile:** their voter wrote nothing the T3 model could read.
- **22.6% of the test votes use a label predicted for a user the T3 model was trained on.** Those are
  voters with a partly reported profile: the model was trained on their reported labels and predicts
  the others. A predicted label is never one the model saw for that user.
- **What was expected.** The inferred labels agree with self-reports for under half of the held-out
  users (0.31 to 0.49 per label, `analysis/2026-11-17_t3-inferred-profiles/` on the T3 branches), so
  no gain was expected before the run. The scores above show one for the voters without a reported
  profile.

## Notes

- **What the inferred labels carry is not tested.** Inside the group "blank, now inferred" every voter
  has text and an inferred line, so the gain there cannot come from the mere fact that a voter has
  text: the labels are what differs between these voters. Whether it is the stance, or something
  else that the T3 model's guess reflects (how much a voter wrote and about what), is not separated.
- **The inferred labels of these voters are out of sample.** None of the 1,267 test votes without a
  reported profile uses a label predicted for a user the T3 model was trained on. The 22.6% in the
  coverage section are all voters with a partly reported profile (28.9% of their votes).
- **The group "still unknown" is small** (243 votes, 31 switched), and its interval includes zero. The
  profile line of these votes is the same for both models, so a difference there cannot come from the
  test input.
- **Extension, not in the proposal: with the voter's voting record added,** the three models are level
  (combined AUROC over three seeds: 0.828 self-reported, 0.835 filled, 0.833 text only). These
  combined rows were not compared with each other in a paired test.
- **This is not the proposal's wording to the letter.** The proposal names voters who left their
  profile blank; `filled` also completes partly reported profiles. A variant that fills blank profiles
  only does not exist in the code. The row "No reported profile" is the proposal's group, and the
  rows under "Some reported profile" show that the completed profiles change little.
- **Both tables are read from one branch.** `ddo_users` is the same on `main` and on `t3` (checked: the
  47,145 profile lines are identical), and `ddo_profiles_inferred` is on `t3` only.
- **Checked without a GPU,** on made-up tables and rows: a reported label always wins and a missing
  one is predicted; a user with no prediction keeps what they reported; a run with `--audience filled`
  trains, saves its scores and is reproduced by `--eval-only`; the same model scored with
  self-reported profiles changes on exactly the rows whose line differs; `jobs/train.sub` and the
  script derive the same run name in 5 argument forms.
- **Checked after the run:** the AUROC that `compare_models.py` computes from each prediction file
  equals the one the Gilbreth job wrote (0.776, 0.744, 0.771). The groups of `filled_subsets.py` have
  the shares of the coverage table, and its numbers equal those of a separate computation written
  directly on the prediction files.
- **The models are on Gilbreth only.** The runs were made without `--push`; scratch is purged after 60
  days.
- **The inferred profiles are private.** They hold stances the voters did not report; the table and
  the prediction files stay out of git.
