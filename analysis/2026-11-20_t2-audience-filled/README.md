# T2 text + audience with filled profiles: inferred labels where the voter reported none

**Status:** prepared on 2026-11-20, not run. The run is checked on made-up rows and its profile
coverage is measured on the real votes. With filled profiles 95.8% of the `ddo_switch` test votes have
a profile, against 78.3% with self-reported profiles only; the profile line changes on 45.9% of them.
No model is trained yet, so there are no scores.

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

Afterwards, with the prediction files copied to `outputs/<run>/` (sandbox off):

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text + audience=outputs/ddo_switch-audience-self" \
    --model "Text + audience, filled profiles=outputs/ddo_switch-audience-filled" \
    --model "Text only=outputs/ddo_switch" \
    --output outputs/compare_models_filled.json
```

With three seeds, each `--model` takes its three folders separated by commas.

## Results

No scores yet. Measured without a model, on the real `ddo_switch` votes: how the profile line of a vote
changes from self-reported to filled profiles.

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
- **What to expect.** The inferred labels agree with self-reports for under half of the held-out
  users (0.31 to 0.49 per label, `analysis/2026-11-17_t3-inferred-profiles/` on the T3 branches), so a
  gain is not expected. The run completes the comparison the proposal names.

## Notes

- **One seed cannot show a small difference.** The gain from the self-reported profile was +0.019 with
  seed 42 and clear only over three seeds (+0.036). A filled run with seed 42 alone can show a large
  change in either direction and no more; three seeds cost about 21 GPU hours.
- **The comparison to make** is filled against self-reported profiles, paired on the same test votes,
  and each against text only. A comparison on the votes whose line changes would say more, but the
  comparison script has no option for that subset yet.
- **This is not the proposal's wording to the letter.** The proposal names voters who left their
  profile blank; `filled` also completes partly reported profiles. A variant that fills blank profiles
  only does not exist in the code.
- **Both tables are read from one branch.** `ddo_users` is the same on `main` and on `t3` (checked: the
  47,145 profile lines are identical), and `ddo_profiles_inferred` is on `t3` only.
- **Checked without a GPU,** on made-up tables and rows: a reported label always wins and a missing
  one is predicted; a user with no prediction keeps what they reported; a run with `--audience filled`
  trains, saves its scores and is reproduced by `--eval-only`; the same model scored with
  self-reported profiles changes on exactly the rows whose line differs; `jobs/train.sub` and the
  script derive the same run name in 5 argument forms.
- **The inferred profiles are private.** They hold stances the voters did not report; the table and
  the prediction files stay out of git.
