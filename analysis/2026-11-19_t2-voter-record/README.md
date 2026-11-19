# T2: the voter's voting record in the model input

**Status:** three models trained on Gilbreth on 2026-11-19 (`ddo_switch`, one seed each). One model
that reads the debate and the voter's voting record scores AUROC 0.838, the same as the text-only model
and the record combined after training (0.836). Adding the profile as well changes nothing (0.831).

## Goal

Glued together after training, the text-only model and the voter's voting record score AUROC 0.836 on
`ddo_switch` (`analysis/2026-11-19_t2-compare-models/`). That is two models and a logistic regression
over their two scores. This adds the record to the input of one model, so that the model reads the
debate and the record together.

The voting record is **not** part of the audience description in the proposal, which is the stance
profile. A run that uses it is an extension of the proposal and should be reported as one.

## What it does

- **`scripts/common/voter_record.py`** counts how often a voter's votes in the training debates ended
  in each class and writes the counts as one line, for example
  `Voting record in other debates: stayed: 11-20; switched: 2`. A voter without such votes gets
  `Voting record in other debates: none`. Class names are read from the dataset. Counts are written as
  ranges (0, 1, 2, 3-5, 6-10, 11-20, 21-50, 51-100, over 100), so the model sees few different values.
- **The counting rules** are those of the listener baseline, which now uses the same function: only
  votes in training debates count, never validation or test debates, and a training row never counts
  its own vote.
- **`--voter-record all`** in `scripts/train_delta.py` appends that line to the first text of each row,
  after the profile line if there is one. The model input is then: resolution, the voter's side before
  the debate, profile (with `--audience`), voting record, then the debate.
- **`--voter-record earlier`** counts only training debates from earlier years than the row's debate.
  The line then starts with `Voting record in debates of earlier years:`.
- **Run names.** Such a run has `-record-all` or `-record-earlier` in its name, after the audience part
  and before the seed, for example `ddo_switch-record-all` or `ddo_switch-audience-self-record-all-seed43`.
  It has its own `outputs/` folder, markers and model repo. `jobs/train.sub` derives the same name.
- **`--shuffle-audience`** (with `--eval-only`) now also swaps the voting record between voters, with
  the same swap as for the profile.

## Run

```bash
# text + voting record; add --smoke first for a quick check
sbatch -A <account> jobs/train.sub ddo_switch --voter-record all --bf16 --push
# the version that could have been known at the time of the debate
sbatch -A <account> jobs/train.sub ddo_switch --voter-record earlier --bf16 --push
# profile and voting record together
sbatch -A <account> jobs/train.sub ddo_switch --audience self --voter-record all --bf16 --push
# control: the same model, records swapped between voters
sbatch -A <account> jobs/train.sub ddo_switch --voter-record all --eval-only --shuffle-audience --bf16
```

Each training run should take about as long as the text-only run (about 7 hours in two or three
chained standby jobs); the added line is short. Not measured.

## Results

`ddo_switch`, 5,845 test votes (368 switched), seed 42, the settings of the text-only run (3 epochs, at
most 4,096 tokens, `--bf16`). Intervals are 95%, from a bootstrap over debates; the table comes from
`scripts/t1_t2/compare_models.py` with the three new models as `--test-scores` rows.

| Row | Test AUROC | 95% interval | Same-debate pairs |
|---|---|---|---|
| Text only | 0.719 | 0.687–0.749 | 50.8% |
| Text + audience (the proposal's model) | 0.737 | 0.709–0.764 | 58.1% |
| Listener: voting record, no text | 0.790 | 0.765–0.813 | 78.3% |
| Listener: profile + voting record + side × year, no text | 0.805 | 0.780–0.829 | 78.2% |
| Text only + voting record, combined after training | 0.836 | 0.812–0.858 | 78.4% |
| **One model: text + voting record** (`--voter-record all`) | **0.838** | 0.816–0.860 | 77.5% |
| One model: text + audience + voting record (`--audience self --voter-record all`) | 0.831 | 0.806–0.854 | 77.2% |
| Text only + voting record of earlier years, combined after training | 0.775 | 0.746–0.802 | 63.7% |
| One model: text + voting record of earlier years (`--voter-record earlier`) | 0.767 | 0.739–0.793 | 63.9% |

| Comparison (first minus second) | AUROC difference | 95% interval | p |
|---|---|---|---|
| One model: text + record − Text only | +0.120 | +0.095 to +0.147 | 0.0005 |
| One model: text + record − Listener: voting record | +0.048 | +0.029 to +0.070 | 0.0005 |
| One model: text + record − Listener: profile + voting record + side × year | +0.034 | +0.016 to +0.054 | 0.0020 |
| One model: text + record − Text only + voting record, combined after training | +0.002 | −0.009 to +0.016 | 0.70 |
| One model: text + audience + record − One model: text + record | −0.007 | −0.019 to +0.003 | 0.20 |
| One model: text + record, earlier years − Listener: voting record, earlier years only | +0.093 | +0.069 to +0.116 | 0.0005 |
| One model: text + record, earlier years − the same two combined after training | −0.008 | −0.020 to +0.005 | 0.24 |

- **One model with the record in its input scores 0.838.** That is 0.120 above the text-only model and
  0.048 above the record alone, both clear, and 0.034 above the best model without text.
- **Reading both together gives no more than combining them afterwards** (0.838 against 0.836; the
  interval of the difference includes 0). The debate and the record seem to add up; this run shows no
  sign that the model uses them jointly.
- **The profile adds nothing on top of the record** (0.831 with it, 0.838 without; not a clear
  difference). The listener baseline and the combinations showed the same.
- **Counting only earlier years, one model scores 0.767,** against 0.674 for that record alone.
- **One seed.** The three-seed runs of the other two models moved by up to 0.02 between seeds
  (`analysis/2026-11-19_t2-compare-models/`), so differences of that size between rows here mean
  little.
- Models: `nlp-final-project11/nlp-project-modernbert-ddo_switch-record-all`, `-record-earlier` and
  `-ddo_switch-audience-self-record-all` (private). The Slurm job ids and durations are not copied here.

Measured without a model, on the real `ddo_switch` rows:

| Kind | Rows whose voter has a record (train / validation / test) | Different lines in training | AUROC of the line alone (train / validation / test) |
|---|---|---|---|
| `all` | 96.9% / 97.0% / 97.3% | 58 | 0.781 / 0.787 / 0.780 |
| `earlier` | 61.9% / 62.2% / 60.7% | 55 | 0.678 / 0.700 / 0.673 |

"AUROC of the line alone" is a logistic regression with one feature per different line, fitted on the
training rows.

- **The line carries the record's signal.** The listener baseline scores 0.790 and 0.674 on the exact
  counts; the line with ranges scores 0.780 and 0.673.
- **The line does not give the label away.** Training and test AUROC are about equal. Counting a
  training row's own vote as well gives 0.910 on the training rows and 0.733 on the test rows, which is
  what a leak looks like.
- **Most test voters never switched in a training debate.** The most common test lines are
  `stayed: 21-50; switched: 0` (571 rows), `stayed: 11-20; switched: 0` (435) and
  `stayed: 6-10; switched: 0` (394). With `earlier`, 2,300 of 5,845 test rows have no record.

## Notes

- **`all` counts later votes too:** every vote of the voter in a training debate, also those cast
  after the debate of the row. `earlier` is the version that could have been known at the time. Report
  both.
- **What to compare a trained model with:** the combination rows of
  `analysis/2026-11-19_t2-compare-models/` (0.836 with `all`, 0.775 with `earlier`), through
  `scripts/t1_t2/compare_models.py --compare`.
- **Runs without `--voter-record` are unchanged.** On made-up rows, runs with a seed, with a profile,
  with swapped profiles and on a three-class config give exactly the same scores before and after this
  change. The listener baseline and all rows of the comparison are also identical after it moved to the
  shared counting function.
- **Checked without a GPU:** the counts equal a slow, direct version on made-up votes, for both kinds
  and with and without the row's own vote; a training row's line leaves its own vote out; a run with
  `--voter-record` trains and saves; `--eval-only` reproduces its test scores; with
  `--shuffle-audience` the scores change on exactly the rows whose line changed; `jobs/train.sub` and
  the script derive the same run name in 10 argument forms.
- **Not run:** the swapped-record control for the trained models, and seeds other than 42.
- **`ddo_undecided`** works the same way (the line then lists `tie`, `pro` and `con`); nothing is
  measured for it here.
