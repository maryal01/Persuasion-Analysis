# T2: the temporal split (train through 2015, test on later debates)

**Status:** built and checked on 2026-11-19; the two ModernBERT runs (text only, text + audience)
finished on 2026-11-20, one seed each. On later debates the text models keep their score: test AUROC
0.700 and 0.758, against 0.708 and 0.744 on the random split. The baselines without text lose much of
theirs. The profile's gain holds: +0.059 (95% interval +0.025 to +0.092, p = 0.002).

## Goal

The proposal splits DDO by debate at random (80/10/10) and names "a temporal split (train through
2015, test 2016–2017) as a robustness check". With the random split a model is tested on debates from
the same years, and often on the same voters, as it was trained on. The temporal split asks whether
the results hold on later debates.

## What it does

- **`scripts/common/splits.py`** makes the temporal split from the stored one, so no second dataset
  is needed:
  - train: debates through 2015 from the stored train and test splits;
  - validation: debates through 2015 from the stored validation split;
  - test: debates after 2015 from all three stored splits.

  The stored splits never share a debate, so these parts do not either. Every row with a year is used
  exactly once.
- **`--split temporal`** in `scripts/train_delta.py` trains and tests on it. Such a run has `-temporal`
  after the config in its name, for example `ddo_switch-temporal-audience-self`, with its own
  `outputs/` folder, markers and model repo. `jobs/train.sub` derives the same name.
- **`--split temporal`** in `scripts/t1_t2/listener_baseline.py` and `scripts/t1_t2/compare_models.py`
  scores the same rows, so the baselines and the paired comparisons can be run on it.
- Without `--split`, everything is as before.

## Run

```bash
# the baselines without text (laptop)
python scripts/t1_t2/listener_baseline.py --users ddo/users.json --split temporal \
    --output outputs/listener_baseline_temporal.json
# the two models of the proposal; add --smoke first for a quick check
sbatch -A <account> jobs/train.sub ddo_switch --split temporal --bf16
sbatch -A <account> jobs/train.sub ddo_switch --split temporal --audience self --bf16
# afterwards, with the prediction files copied to outputs/<run>/
python scripts/t1_t2/compare_models.py --users ddo/users.json --split temporal \
    --model "Text only=outputs/ddo_switch-temporal" \
    --model "Text + audience=outputs/ddo_switch-temporal-audience-self" \
    --compare "Text only" "Listener: profile + voting record + side x year" \
    --compare "Text + audience" "Listener: profile + voting record + side x year" \
    --output outputs/compare_models_temporal.json
```

The two runs were made without `--push` (see Notes); add it to upload a model to the Hub.

## Results

The split, for `ddo_switch`:

| Part | Years | Votes | Debates | Switched |
|---|---|---|---|---|
| train | 2007–2015 | 52,385 | 21,161 | 3,388 |
| validation | 2007–2015 | 5,876 | 2,379 | 348 |
| test | 2016–2018 | 2,148 | 1,695 | 131 |

For comparison, the random split has 48,479 / 6,085 / 5,845 votes and 368 switchers in its test part.

Baselines without text, test AUROC with a 95% interval from a bootstrap over debates
(`listener_baseline.py`; the last column is the same model on the random split, from
`analysis/2026-11-16_t2-listener-baseline/` and `analysis/2026-11-19_t2-compare-models/`):

| Model | Temporal split | 95% interval | Random split |
|---|---|---|---|
| Side before × year | 0.568 | 0.522–0.616 | 0.643 |
| Profile (the proposal's profile-only baseline) | 0.580 | 0.529–0.628 | 0.678 |
| Voting record (extension) | 0.616 | 0.581–0.650 | 0.790 |
| Voting record, earlier years only (extension) | 0.620 | 0.586–0.655 | 0.674 |
| Profile + voting record + side × year (extension) | 0.675 | 0.631–0.720 | 0.805 |

- **Every baseline without text scores lower on later debates.** The profile alone falls from 0.678
  to 0.580 and the voting record from 0.790 to 0.616.
- **Most test voters are new.** Only 43.2% of the test votes come from a voter who also votes in a
  training debate; 94.8% come from a voter with a profile record. The voting record cannot help for a
  voter it has never seen.
- **The year tells nothing here:** the test years do not occur in training, so "side before × year"
  can use the side only.

### The two models of the proposal

One run each: seed 42, 3 epochs, at most 4,096 tokens, `--bf16`. Test AUROC on the 2,148 votes of
2016–2018, with a 95% interval from a bootstrap over debates. The validation column is on debates
through 2015. The last column is the same row on the random split, averaged over three seeds (the
three-seed comparison of `analysis/2026-11-19_t2-voter-record/`, `outputs/compare_models_all_seeds.json`);
with seed 42 alone the two models have 0.719 and 0.737 there.

| Row | Validation | Temporal split | 95% interval | Random split |
|---|---|---|---|---|
| Text only | 0.688 | 0.700 | 0.655–0.748 | 0.708 |
| Text + audience (the proposal's model) | 0.750 | 0.758 | 0.715–0.799 | 0.744 |
| Text only + voting record, combined after training (extension) | | 0.754 | 0.713–0.798 | 0.833 |
| Text + audience + voting record, combined after training (extension) | | 0.782 | 0.741–0.820 | 0.828 |

| Comparison on the temporal split (first minus second) | AUROC difference | 95% interval | p |
|---|---|---|---|
| Text + audience − Text only | +0.059 | +0.025 to +0.092 | 0.0020 |
| Text only − best baseline without text (0.675, uses the voting record) | +0.025 | −0.029 to +0.076 | 0.37 |
| Text + audience − best baseline without text | +0.084 | +0.038 to +0.132 | 0.0010 |
| Text only + voting record − Text only (extension) | +0.055 | +0.033 to +0.078 | 0.0005 |
| Text + audience + voting record − Text + audience (extension) | +0.024 | +0.005 to +0.042 | 0.0105 |

- **The text models keep their score on later debates.** Text only has 0.700 and text + audience
  0.758, against 0.708 and 0.744 on the random split. The two test parts hold different votes and the
  intervals here are about ±0.045, so this says "about the same" and no more.
- **The baselines without text do not, so the order turns around.** On the random split the voting
  record alone (0.790) is above both text models. Here the best baseline without text has 0.675:
  text + audience is 0.084 above it (p = 0.0010), and text only is 0.025 above it with an interval
  that includes 0.
- **The profile's gain holds.** +0.059 on later debates, against +0.036 (+0.019 to +0.053) on the
  random split. The two intervals overlap, so this does not show a larger gain.
- **The voting record adds less here (extension of the proposal).** Combined with a text model after
  training it adds 0.055 to text only and 0.024 to text + audience; on the random split it lifts both
  to about 0.83. Most test voters are new, and the record has nothing on them.
- **Validation and test agree for the text models** (0.688 and 0.700, 0.750 and 0.758), although the
  validation debates are from the training years. For the voting record alone they do not (0.800 and
  0.616).

## Notes

- **The test part runs to 2018.** The proposal says 2016–2017; the corpus has 223 `ddo_switch` votes
  from 2018 debates, and they are in the test part too.
- **131 switchers in the test part,** against 368 in the random split, so the intervals are wider
  (for the voting record 0.07 wide, against 0.05). Small differences between models will not be clear
  on this split.
- **Validation is from the training years,** not from later ones. The checkpoint and the baselines' C
  are chosen on it, so nothing about the test years is used before the test.
- **One seed** for each of the two runs: this is a check, not a second main result. Between seeds the
  random-split runs differ by up to 0.022 (text only) and 0.012 (text + audience).
- **Where the two models are.** They first ran with `--push`; training finished, and the upload at
  the end failed because the org's private storage on the Hub was full. Started again without
  `--push`, each run continued from its last checkpoint and wrote its scores. The models are in
  `outputs/ddo_switch-temporal/` and `outputs/ddo_switch-temporal-audience-self/` on Gilbreth scratch,
  which is purged after 60 days, and not on the Hub. Not copied from the job logs: the selected
  checkpoints and the run times.
- **The voting record on this split** counts only training debates, so it never counts a vote cast
  after the test debate. "All" and "earlier years only" count the same votes for a test row; the two
  rows differ only in how the model was fitted on the training rows.
- **`ddo_undecided`** works the same way (`--dataset-config ddo_undecided --split temporal`); nothing
  is measured for it here.
- **Runs without `--split` are unchanged.** On made-up rows, a three-class run with profiles, its
  re-score, its swapped-profile control and a text-only re-score give exactly the same scores before
  and after this change (six files, difference 0).
- **Checked without a GPU,** on made-up rows: every row with a year lands in exactly one part; train
  and validation hold only debates through 2015 and test only later ones; validation rows all come
  from the stored validation split; no debate is in two parts; the version for datasets and the
  version for the baseline's columns give the same rows in the same order; a run with
  `--split temporal` trains, saves predictions for exactly the temporal test and validation rows and
  records the split; a smoke run has rows in every part; `jobs/train.sub` and the script derive the
  same run name in 20 argument forms.
