# T2: the temporal split (train through 2015, test on later debates)

**Status:** built and checked on 2026-11-19. The baselines without text are measured on it and lose
much of their score. The two ModernBERT runs (text only, text + audience) are not run yet.

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
sbatch -A <account> jobs/train.sub ddo_switch --split temporal --bf16 --push
sbatch -A <account> jobs/train.sub ddo_switch --split temporal --audience self --bf16 --push
# afterwards, with the prediction files copied to outputs/<run>/
python scripts/t1_t2/compare_models.py --users ddo/users.json --split temporal \
    --model "Text only=outputs/ddo_switch-temporal" \
    --model "Text + audience=outputs/ddo_switch-temporal-audience-self" \
    --output outputs/compare_models_temporal.json
```

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
- **The ModernBERT runs are open.** Whether the text models hold up better than the baselines is what
  the two runs above will show.

## Notes

- **The test part runs to 2018.** The proposal says 2016–2017; the corpus has 223 `ddo_switch` votes
  from 2018 debates, and they are in the test part too.
- **131 switchers in the test part,** against 368 in the random split, so the intervals are wider
  (for the voting record 0.07 wide, against 0.05). Small differences between models will not be clear
  on this split.
- **Validation is from the training years,** not from later ones. The checkpoint and the baselines' C
  are chosen on it, so nothing about the test years is used before the test.
- **One seed** is planned for each of the two runs: this is a check, not a second main result.
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
