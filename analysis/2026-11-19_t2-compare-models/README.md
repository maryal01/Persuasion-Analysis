# T2: paired comparisons, and the text model combined with the listener baseline

**Status:** script written and checked on a laptop with made-up model scores (2026-11-19). Not run on
the real model scores yet: they are saved on Gilbreth and have to be copied first.

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

**More seeds.** `--seed <N>` in `scripts/train_delta.py` sets the training seed (default 42, the seed of
the first runs). A run with another seed has `-seed<N>` at the end of its name, so it has its own
`outputs/` folder, markers and model repo and does not touch the first run:

```bash
sbatch -A <account> jobs/train.sub ddo_switch --seed 43 --bf16 --push                    # text only
sbatch -A <account> jobs/train.sub ddo_switch --audience self --seed 43 --bf16 --push    # text + audience
```

Each run saves its validation and test scores, so the comparison can be run per seed, for example with
`--model "Text only=outputs/ddo_switch-seed43" --model "Text + audience=outputs/ddo_switch-audience-self-seed43"`.
No run with another seed exists yet.

## Results

Not run on real model scores yet.

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
- **Two-class configs only.** `ddo_undecided` has three classes and no single score per row.
- **The prediction files hold voter names.** They stay in `outputs/`, which git ignores.
