# T2: paired comparisons, and the text model combined with the listener baseline

**Status:** run on 2026-11-19 on the scores of the trained `ddo_switch` models, first for seed 42 and
then for three seeds (42, 43, 44). The text-only model combined with the voter's voting record scores
AUROC 0.833 on average (0.836 for seed 42), above the record alone (0.790) and the text alone (0.708).
The voter's profile adds 0.036 on average (0.708 to 0.744); the gain is clear in two of the three seeds.

## Goal

Two questions about `ddo_switch` that the single AUROC numbers cannot answer:

1. **Is one model really better than another?** For example text + audience (0.737) against text only
   (0.719). The proposal asks for 95% bootstrap intervals and paired tests between models.
2. **Does the debate text add to what the voter's record already tells?** The voting record alone
   scores 0.790 and all listener features 0.805, both without any text
   (`analysis/2026-11-16_t2-listener-baseline/`). No model has used the text and the record together.

The voting record is **not** part of the audience description in the proposal, which is the stance
profile. Every row that uses it is an extension of the proposal and should be reported as one.

## What it does

`scripts/t1_t2/compare_models.py` reads the per-row scores that `scripts/train_delta.py` saves and
builds one table for the test split:

- **A row per trained model** (`--model NAME=FOLDER`), from `predictions_validation.parquet` and
  `predictions_test.parquet` in the run's `outputs/` folder. The score is the logit of `switched` minus
  the logit of `stayed`. Rows are matched on debate and voter, and the labels must agree with the
  dataset.
- **Three listener rows**, fitted here with the code of the listener baseline: the voting record, the
  voting record from earlier years only, and profile + voting record + side × year.
- **A combination of every trained model with every listener row.** A logistic regression on two
  numbers, the text model's score and the listener model's score, is fitted on the validation split and
  scored once on the test split. The test split is never used for fitting.
- **Optional control rows** (`--test-scores NAME=FILE`) that have test scores only, such as the
  swapped-profile and predicted-profile runs.

Every row gets its test AUROC with a 95% interval (bootstrap over debates, 1,000 resamples) and the
same-debate pairs of the listener baseline. Each comparison gets the AUROC difference, its 95% interval
on the same resamples, and a paired permutation test. Compared by default: every later model with the
first, and every combination with both of its parts. `--compare ROW OTHER_ROW` adds more.

**Several seeds of one model** are given as folders separated by commas. The row then shows the AUROC
averaged over the seeds and the standard deviation between them. Its 95% interval is that of the
average (the seeds are averaged inside every bootstrap resample), so it describes the test votes, not
the spread between seeds. A combination is fitted once per seed. In a comparison, seeds are paired in
the order given, and a row with one set of scores is compared with every seed of the other row.

`scripts/train_delta.py` now also saves `predictions_validation.parquet`, for training runs and for
`--eval-only` runs. Before this change only the test scores were saved.

## Run

```bash
# On Gilbreth: save the validation and test scores of the two trained models (a short job each).
sbatch -A <account> jobs/train.sub ddo_switch --eval-only --bf16
sbatch -A <account> jobs/train.sub ddo_switch --audience self --eval-only --bf16

# On a laptop with ddo/users.json, after copying outputs/ddo_switch*/predictions_*.parquet from Gilbreth:
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text only=outputs/ddo_switch" \
    --model "Text + audience=outputs/ddo_switch-audience-self" \
    --test-scores "Text + audience, swapped profiles=outputs/ddo_switch-audience-self/predictions_test_shuffled.parquet" \
    --test-scores "Text + audience, predicted profiles=outputs/ddo_switch-audience-inferred/predictions_test.parquet" \
    --compare "Text + audience" "Text + audience, swapped profiles" \
    --compare "Text + audience" "Text + audience, predicted profiles" \
    --output outputs/compare_models.json
```

The script streams the id and label columns of `ddo_switch` from the Hub (branch `main`) and takes
about a minute plus the permutation tests (2,000 per comparison by default).

**More seeds.** `--seed <N>` in `scripts/train_delta.py` sets the training seed (default 42, the seed of
the first runs). A run with another seed has `-seed<N>` at the end of its name, so it has its own
`outputs/` folder, markers and model repo and does not touch the first run:

```bash
sbatch -A <account> jobs/train.sub ddo_switch --seed 43 --bf16 --push                    # text only
sbatch -A <account> jobs/train.sub ddo_switch --audience self --seed 43 --bf16 --push    # text + audience
```

Each run saves its validation and test scores. Seeds 43 and 44 were run for both models on 2026-11-19.
The comparison over the three seeds (see "Three seeds" under Results):

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text only=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44" \
    --model "Text + audience=outputs/ddo_switch-audience-self,outputs/ddo_switch-audience-self-seed43,outputs/ddo_switch-audience-self-seed44" \
    --compare "Text only + voting record" "Text + audience + voting record" \
    --output outputs/compare_models_seeds.json
```

## Results

`ddo_switch`, 5,845 test votes from 2,504 debates (368 switched). The two trained models are the
seed-42 runs, scored again with `--eval-only` on 2026-11-19 to save their validation scores. Their four
file rows reproduce the AUROC printed by the Gilbreth jobs (0.719, 0.737, 0.682, 0.706). The script and
the way of combining were committed before the real scores were read.

| Row | Test AUROC | 95% interval | Same-debate pairs |
|---|---|---|---|
| Text only | 0.719 | 0.687–0.749 | 50.8% |
| Text + audience (the proposal's model) | 0.737 | 0.709–0.764 | 58.1% |
| Text + audience, swapped profiles | 0.682 | 0.650–0.714 | 51.6% |
| Text + audience, predicted profiles | 0.706 | 0.678–0.732 | 55.6% |
| Listener: voting record | 0.790 | 0.765–0.813 | 78.3% |
| Listener: voting record, earlier years only | 0.674 | 0.646–0.700 | 62.9% |
| Listener: profile + voting record + side × year | 0.805 | 0.780–0.829 | 78.2% |
| **Text only + voting record** | **0.836** | 0.812–0.858 | 78.4% |
| Text only + voting record, earlier years only | 0.775 | 0.746–0.802 | 63.7% |
| Text only + all listener features | 0.831 | 0.807–0.854 | 78.0% |
| Text + audience + voting record | 0.824 | 0.801–0.846 | 78.5% |
| Text + audience + voting record, earlier years only | 0.771 | 0.743–0.798 | 65.5% |
| Text + audience + all listener features | 0.821 | 0.796–0.845 | 78.3% |

The rows from "Text only + voting record" down are combinations fitted on the validation split. All of
them use the voting record, so they go beyond the proposal's audience description.

Paired comparisons on the test split (first minus second; 2,000 permutations, so the smallest
possible p-value is 0.0005):

| Comparison | AUROC difference | 95% interval | p |
|---|---|---|---|
| Text + audience − Text only | +0.019 | −0.005 to +0.045 | 0.14 |
| Text + audience − the same model with swapped profiles | +0.055 | +0.035 to +0.075 | 0.0005 |
| Text + audience − the same model with predicted profiles | +0.031 | +0.014 to +0.049 | 0.0010 |
| Text only + voting record − Listener: voting record | +0.046 | +0.031 to +0.061 | 0.0005 |
| Text only + voting record − Text only | +0.117 | +0.092 to +0.143 | 0.0005 |
| Text only + voting record − Listener: profile + voting record + side × year | +0.031 | +0.018 to +0.046 | 0.0005 |
| Text only + voting record − Text only + all listener features | +0.005 | −0.002 to +0.011 | 0.17 |
| Text only + voting record − Text + audience + voting record | +0.011 | +0.001 to +0.021 | 0.018 |
| Text only + voting record, earlier years only − Listener: voting record, earlier years only | +0.101 | +0.076 to +0.126 | 0.0005 |
| Text only + voting record, earlier years only − Text + audience | +0.038 | +0.013 to +0.061 | 0.0025 |
| Text + audience + all listener features − Listener: profile + voting record + side × year | +0.017 | +0.009 to +0.025 | 0.0005 |

- **The text adds to what the voting record tells.** Text only + voting record scores 0.836, which is
  0.046 above the record alone and 0.117 above the text alone. Both differences are clear. The record
  gets about twice the weight of the text score (1.33 against 0.73 on standardised scores).
- **The gain from the voter's profile is not shown.** Text + audience is 0.019 above text only, with an
  interval that includes 0 (p = 0.14). This is one seed per model; with three seeds the gain is clear
  (see "Three seeds" below).
- **The model does read the profile.** With swapped profiles it loses 0.055, and with the T3 model's
  predicted profiles 0.031; both differences are clear.
- **With the record, the profile no longer helps.** Combined with the voting record, the text-only
  model is 0.011 above the text + audience model. The listener baseline showed the same: the profile
  adds nothing once the record is known.
- **Counting only earlier years,** text + record scores 0.775. That is still above the text + audience
  model (0.737) and far above that record alone (0.674).
- **Same-debate pairs of the text-only model are at 50.8%, not exactly 50%.** Voters of one debate with
  the same side get the same input, and their scores differ by at most 0.05 (the scores have a standard
  deviation of 1.5), so their order is close to random. The text + audience model separates them with
  the profile (58.1%).

### Three seeds

Both trained models were run again with seeds 43 and 44 (`--seed`, same settings; model repos
`nlp-project-modernbert-ddo_switch-seed43`, `-seed44`, `-ddo_switch-audience-self-seed43`, `-seed44`).
Test AUROC; the last column is the mean and the standard deviation over the three seeds. The listener
rows do not depend on the seed. The 95% intervals of the three-seed averages are 0.679–0.736 for text
only, 0.716–0.767 for text + audience and 0.808–0.854 for text only + voting record.

| Row | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|---|---|---|---|---|
| Text only | 0.719 | 0.697 | 0.709 | 0.708 ± 0.011 |
| Text + audience (the proposal's model) | 0.737 | 0.746 | 0.749 | 0.744 ± 0.006 |
| Listener: voting record | 0.790 | 0.790 | 0.790 | 0.790 |
| Listener: profile + voting record + side × year | 0.805 | 0.805 | 0.805 | 0.805 |
| Text only + voting record | 0.836 | 0.829 | 0.834 | 0.833 ± 0.004 |
| Text only + voting record, earlier years only | 0.775 | 0.763 | 0.770 | 0.769 ± 0.006 |
| Text only + all listener features | 0.831 | 0.824 | 0.831 | 0.829 ± 0.004 |
| Text + audience + voting record | 0.824 | 0.830 | 0.830 | 0.828 ± 0.003 |
| Text + audience + voting record, earlier years only | 0.771 | 0.778 | 0.780 | 0.777 ± 0.005 |
| Text + audience + all listener features | 0.821 | 0.826 | 0.827 | 0.825 ± 0.003 |

Paired comparisons (AUROC difference with its 95% interval). The per-seed columns come from one run of
the script per seed, the last column from the run over all three seeds:

| Comparison | Seed 42 | Seed 43 | Seed 44 | Three seeds together |
|---|---|---|---|---|
| Text + audience − Text only | +0.019 (−0.005 to +0.045) | +0.049 (+0.030 to +0.070) | +0.040 (+0.021 to +0.059) | +0.036 (+0.019 to +0.053), p = 0.0005 |
| Text only + voting record − Listener: voting record | +0.046 (+0.031 to +0.061) | +0.039 (+0.026 to +0.052) | +0.044 (+0.031 to +0.057) | +0.043 (+0.030 to +0.057), p = 0.0005 |
| Text only + voting record − Text only | +0.117 (+0.092 to +0.143) | +0.132 (+0.106 to +0.157) | +0.124 (+0.097 to +0.151) | +0.125 (+0.099 to +0.150), p = 0.0005 |
| Text + audience + voting record − Text + audience | +0.087 (+0.066 to +0.110) | +0.085 (+0.063 to +0.107) | +0.081 (+0.061 to +0.102) | +0.084 (+0.064 to +0.106), p = 0.0005 |
| Text only + voting record − Text + audience + voting record | +0.011 (+0.001 to +0.021) | not run | not run | +0.005 (−0.002 to +0.011), p = 0.17 |

- **The profile helps.** Text + audience is above text only in all three seeds, and clearly so in
  seeds 43 and 44 (p = 0.0005 each; p = 0.14 for seed 42). Over the three seeds together the difference
  is +0.036, with a 95% interval of +0.019 to +0.053.
- **Seed 42 was the least favourable seed for this comparison:** it has the best text-only model
  (0.719) and the weakest text + audience model (0.737).
- **The text adds to the voting record in every seed** (+0.039 to +0.046), and the combination is
  stable: 0.829 to 0.836.
- **With the record, the profile makes no clear difference:** 0.833 without it and 0.828 with it, on
  average (+0.005, interval −0.002 to +0.011). The seed-42 difference of 0.011 between the two does not
  repeat (seed 43: 0.829 against 0.830; seed 44: 0.834 against 0.830).
- **Seed averaging, checked:** with one folder per model the script gives exactly the rows, intervals
  and p-values of the seed-42 run above; the three-seed rows equal the averages of the three per-seed
  runs; the same folder given twice gives a standard deviation of 0 and a difference of exactly 0 to
  itself; rows with 3 and 2 seeds are refused.
- **Not recorded here:** the Slurm job ids and durations of the four new runs.

## Notes

- **The voting record counts later votes too.** It uses all of a voter's votes in training debates,
  also those cast after the debate of the row. The "earlier years only" rows are the version that
  could have been known at the time; on its own that record scores 0.674, against 0.790. Report both.
- **The combination has three numbers to fit** (two weights and an intercept) on 6,085 validation rows
  with 366 switchers. It cannot use anything about a row except the two scores.
- **The text models chose their checkpoint on validation AUROC,** so their validation scores are a
  little better than they would be on new rows. The combination is fitted on those scores and may lean
  on the text score slightly too much. Not measured.
- **The permutation test** swaps the two models' scores for whole debates with probability 1/2. The
  scores are replaced by their ranks first, which puts both models on one scale and leaves each
  model's own AUROC unchanged. With 2,000 permutations the smallest possible p-value is 0.0005.
- **Checked with made-up scores for the real rows:** the three listener rows equal the committed
  baseline in AUROC, interval and same-debate pairs (0.790, 0.674, 0.805); a file's row gets the AUROC
  of its scores although the file is in another row order; a model with one score per debate and side
  is at 50.0% on same-debate pairs; a second model with the same ranking gives a difference of exactly
  0 with p = 1; a model that knows more gives a positive difference with an interval above 0.
  `scripts/train_delta.py` on made-up rows writes the validation file in every mode.
- **What the seed controls:** the start weights of the classifier on top of the encoder, and the order
  of the training rows. In the runs made before `--seed` existed, the order was seeded with 42 but the
  classifier's start weights were not, so those runs cannot be repeated exactly.
- **The seed option, checked on made-up rows:** the same seed twice gives identical test scores;
  another seed gives different ones; no `--seed` equals `--seed 42`; `jobs/train.sub` and the script
  derive the same run name in 13 argument forms. The T3 scripts write to one fixed folder whatever
  their seed, so `jobs/train.sub` gives names starting with `t3_` no seed suffix.
- **Three-class configs** (`ddo_undecided`) have no single score per row. Since
  `analysis/2026-11-19_t2-undecided-audience/` the script compares their predicted classes on macro-F1
  instead and fits no combination. Everything on this page is about `ddo_switch` and is unchanged by
  that.
- **The prediction files hold voter names.** They stay in `outputs/`, which git ignores.
