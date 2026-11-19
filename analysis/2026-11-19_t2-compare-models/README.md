# T2: paired comparisons, and the text model combined with the listener baseline

**Status:** run on 2026-11-19 on the scores of the two trained `ddo_switch` models (one seed each).
The text-only model combined with the voter's voting record scores AUROC 0.836, above the record alone
(0.790) and the text alone (0.719). The gain from the voter's profile (0.719 to 0.737) is not
significant with one seed.

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
  interval that includes 0 (p = 0.14). This is one seed per model.
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
- **Two-class configs only.** `ddo_undecided` has three classes and no single score per row.
- **The prediction files hold voter names.** They stay in `outputs/`, which git ignores.
