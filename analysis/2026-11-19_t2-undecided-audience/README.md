# T2: text + audience for `ddo_undecided`, and the comparison for three classes

**Status:** prepared on 2026-11-19. The comparison script now handles three-class configs and is
checked on made-up scores. The Gilbreth runs for `ddo_undecided` (text + audience, its swapped-profile
control and a re-score of the text-only model) were not finished when this was written, so that task
has no results yet. One result is in: on `ddo_switch`, the voters who filled in the other criteria
(98.2% of the test votes) give the same picture as all voters.

## Goal

The proposal's T2 has two parts: did a decided voter switch (`ddo_switch`), and where does a voter who
was tied before the debate end (`ddo_undecided`: `tie`, `pro` or `con`). The text + audience model, the
debate with the voter's stance profile in front of it, exists for `ddo_switch` only
(`analysis/2026-11-17_t2-audience-input/`). This adds it for `ddo_undecided`, with what the proposal
promises around it:

- the same model with a shuffled description (its score should drop),
- 95% bootstrap intervals and paired permutation tests between models,
- results per class,
- a rerun on the voters who filled in the other criteria, because a tie before the debate can mean
  "undecided" or "skipped".

## What it does

- **Training needs no change.** `scripts/train_delta.py --dataset-config ddo_undecided --audience self`
  writes the profile line in front of the debate as for `ddo_switch`. Three classes are scored with
  macro-F1, and the checkpoint is chosen on validation macro-F1.
- **`scripts/t1_t2/compare_models.py` now follows the number of classes,** as the listener baseline
  does. For `ddo_switch` nothing changes (AUROC of the score, combinations with the listener
  baseline). For a config with more classes:
  - a model's output per vote is its predicted class, the one with the largest logit;
  - every row gets test macro-F1 with a 95% interval, accuracy with its interval, and the F1 of each
    class; with several seeds these are averages over the seeds, as for two classes;
  - a comparison is the macro-F1 difference with a 95% interval and a paired permutation test. Both
    resample whole debates. The permutation test swaps the two models' predicted classes for all
    votes of a debate with probability 1/2;
  - the rows of the listener baseline are added to compare with: always the largest training class,
    and logistic regression on the year, the profile, the voting record (all, and earlier years
    only) and their combinations. The first two of these are the proposal's baselines without text.
    The rows with the voting record are an extension of the proposal;
  - no combination of a text model with the listener baseline is fitted.
- **`--subset COLUMN`** reports every row and comparison a second time, on the test rows where that
  true/false column of the dataset is true. With `--subset filled_other_criteria` that is the
  proposal's rerun: the voters who marked a debater on at least one other criterion. Nothing is fitted
  again; the models are the ones trained on all rows.
- **`load_rows`** of `scripts/t1_t2/listener_baseline.py` can read extra columns (used for the subset).

## Run

On Gilbreth (a smoke run first; the training run needs two or three chained standby jobs):

```bash
# text + audience; add --smoke first for a quick check
sbatch -A <account> jobs/train.sub ddo_undecided --audience self --bf16 --push
# control, after training: the same model, profiles swapped between voters
sbatch -A <account> jobs/train.sub ddo_undecided --audience self --eval-only --shuffle-audience --bf16
# the text-only model of RESULTS.md scored again, to save its score per vote
sbatch -A <account> jobs/train.sub ddo_undecided --eval-only --bf16
```

Then, with the `predictions_*.parquet` files of both runs copied to `outputs/<run>/` (sandbox off; the
script streams ids and labels from the Hub and reads `ddo/users.json`):

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json --dataset-config ddo_undecided \
    --model "Text only=outputs/ddo_undecided" \
    --model "Text + audience=outputs/ddo_undecided-audience-self" \
    --test-scores "Text + audience, swapped profiles=outputs/ddo_undecided-audience-self/predictions_test_shuffled.parquet" \
    --compare "Text + audience" "Text + audience, swapped profiles" \
    --compare "Text + audience" "Listener: profile" \
    --compare "Text + audience" "Listener: year" \
    --compare "Text + audience" "Always tie" \
    --subset filled_other_criteria \
    --output outputs/compare_models_undecided.json
```

The `ddo_switch` rerun on the subset is the three-seed command of
`analysis/2026-11-19_t2-voter-record/` with `--subset filled_other_criteria` added and
`--output outputs/compare_models_all_seeds_subset.json`.

## Results

### `ddo_switch` on the voters who filled in the other criteria

`filled_other_criteria` is true for 5,739 of the 5,845 test votes (98.2%), from 2,468 of the 2,504 test
debates; 357 of them switched. Test AUROC averaged over seeds 42, 43 and 44, with the 95% interval of
the average. The listener row does not depend on the seed.

| Row | All test votes | Voters who filled in the other criteria |
|---|---|---|
| Text only | 0.708 (0.679–0.736) | 0.711 (0.685–0.739) |
| Text + audience (the proposal's model) | 0.744 (0.716–0.767) | 0.746 (0.720–0.772) |
| Listener: voting record, no text (extension) | 0.790 (0.765–0.813) | 0.789 (0.763–0.813) |
| Text only + voting record, combined after training (extension) | 0.833 (0.808–0.854) | 0.832 (0.811–0.855) |
| One model: text + voting record (extension) | 0.830 (0.809–0.850) | 0.831 (0.807–0.854) |

| Comparison on the subset (first minus second) | AUROC difference | 95% interval | p |
|---|---|---|---|
| Text + audience − Text only | +0.035 | +0.018 to +0.051 | 0.0010 |
| One model: text + record − Text only | +0.120 | +0.096 to +0.143 | 0.0005 |
| One model: text + record − Listener: voting record | +0.042 | +0.023 to +0.062 | 0.0005 |

- **Nothing changes.** Every row moves by at most 0.003, and the profile's gain is +0.035 against
  +0.036 on all votes. Decided voters almost all filled in another criterion, so this check says
  little for `ddo_switch`; it matters for `ddo_undecided`, where a tie before the debate is the label.
- **The numbers on all test votes are those already reported.** Run with the changed script, every row
  and comparison of `outputs/compare_models_all_seeds.json` comes out identical.

### `ddo_undecided`

None yet. What the new model will be set against, on the same 7,091 test votes (6,508 `tie`, 256 `pro`,
327 `con`):

| Row | Test macro-F1 | Test accuracy | Source |
|---|---|---|---|
| Always `tie` | 0.319 | 0.918 | listener baseline; reproduced by this script |
| Listener: year | 0.382 | 0.788 | same |
| Listener: profile | 0.342 | 0.584 | same |
| Listener: voting record (extension) | 0.423 | 0.748 | same |
| ModernBERT-base, text only | 0.360 | 0.644 | `RESULTS.md` on branch `t2-ddo-switch-results`; its scores per vote are not saved yet |

The subset: `filled_other_criteria` is true for 6,741 of the 7,091 test votes (95.1%), from 3,536 of
the 3,649 test debates: 6,169 `tie`, 252 `pro`, 320 `con`. The rerun therefore leaves out one vote in
twenty, almost all of them `tie`.

## Notes

- **The rerun is an evaluation, not a new training run.** The proposal says "we rerun T2 on voters who
  filled in the other criteria". Here the models trained on all voters are scored on those voters.
  Training on them only would be a separate run, which is not done here.
- **One seed.** The run prepared here is seed 42. On `ddo_switch` the gain from the profile was clear
  only over three seeds (`analysis/2026-11-19_t2-compare-models/`).
- **The text-only model predates `--seed`,** so its classifier's start weights were not seeded. Its
  re-score should reproduce the recorded 0.360 / 0.644; if it does not, the dataset or the model
  changed since.
- **Profile coverage:** 82.0% / 81.5% / 81.7% of the train / validation / test rows have a voter with
  a known self-reported profile (`analysis/2026-11-17_t2-audience-input/`).
- **Checked without a GPU,** on made-up rows with three classes: a run with `--audience self` trains and
  saves test and validation scores; `--eval-only` on the saved model reproduces its test logits
  exactly, for the model with profiles and for the text-only one; with `--shuffle-audience` the logits
  change on exactly the rows whose profile line changed, and the control does not overwrite the
  model's own files; a model with other class names is refused.
- **Checked on the Hub:** `ddo_undecided` has the same rows on branches `t1-t2` and `main` (58,156 /
  7,163 / 7,091) with the classes `tie`, `pro`, `con`, and the saved text-only model has the same
  class names.
- **The comparison, checked with made-up logits for the real rows:** a file's row gets the macro-F1
  and accuracy that scikit-learn computes for its predictions, on all test rows and on the subset,
  although the file is in another row order; two seeds give the average, the per-seed values and
  their standard deviation; the seven rows without text equal the committed listener baseline in
  macro-F1, interval, accuracy and F1 per class; a second model with the same predicted classes gives
  a difference of exactly 0 with p = 1; a model that knows more gives a positive difference with an
  interval above 0.
- **The prediction files hold voter names.** They stay in `outputs/`, which git ignores.
