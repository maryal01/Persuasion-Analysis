# T2: text + audience for `ddo_undecided`, and the comparison for three classes

**Status:** run on Gilbreth and compared on 2026-11-20, three seeds per model. On `ddo_undecided` the
profile in front of the debate brings no clear gain: test macro-F1 is 0.380 with it and 0.374 without
(difference +0.006, 95% interval −0.002 to +0.014, p = 0.15). Neither model is above logistic
regression on the year of the debate (0.382). The model does read the profile: with the profiles
swapped between voters its macro-F1 drops by 0.011 (p = 0.0035, one seed). On the voters who filled in
the other criteria the picture is the same, for `ddo_undecided` and for `ddo_switch`.

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
# text + audience; add --smoke first for a quick check, and --seed 43 / --seed 44 for the other seeds
sbatch -A <account> jobs/train.sub ddo_undecided --audience self --bf16
# control, after training: the same model, profiles swapped between voters
sbatch -A <account> jobs/train.sub ddo_undecided --audience self \
    --eval-only outputs/ddo_undecided-audience-self --shuffle-audience --bf16
# the text-only model of RESULTS.md scored again, to save its score per vote
sbatch -A <account> jobs/train.sub ddo_undecided --eval-only --bf16
# the text-only model with the other seeds
sbatch -A <account> jobs/train.sub ddo_undecided --seed 43 --bf16
```

These runs were made without `--push`, so their models are in `outputs/<run>/` on Gilbreth and not on
the Hub (see Notes). With `--push`, `--eval-only` needs no folder: it reads the model from the Hub.

Then, with the `predictions_*.parquet` files of the runs copied to `outputs/<run>/` (sandbox off; the
script streams ids and labels from the Hub and reads `ddo/users.json`):

```bash
# seed 42, with the swapped-profile control
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
# three seeds
python scripts/t1_t2/compare_models.py --users ddo/users.json --dataset-config ddo_undecided \
    --model "Text only=outputs/ddo_undecided,outputs/ddo_undecided-seed43,outputs/ddo_undecided-seed44" \
    --model "Text + audience=outputs/ddo_undecided-audience-self,outputs/ddo_undecided-audience-self-seed43,outputs/ddo_undecided-audience-self-seed44" \
    --compare "Text + audience" "Listener: profile" \
    --compare "Text + audience" "Listener: year" \
    --compare "Text + audience" "Always tie" \
    --subset filled_other_criteria \
    --output outputs/compare_models_undecided_seeds.json
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

Both models with seeds 42, 43 and 44 (3 epochs, at most 4,096 tokens, `--bf16`, class weights). The
text-only model with seed 42 is the one of `RESULTS.md`, scored again; the other five runs finished
on 2026-11-20. The test split has 7,091 votes: 6,508 `tie`, 256 `pro`, 327 `con`. A model row is the
average over its three seeds, ± the standard deviation between them, with the 95% interval of the
average. The rows without text do not depend on the seed.

| Row | Test macro-F1 | 95% interval | Per seed | Accuracy | F1 `tie` | F1 `pro` | F1 `con` |
|---|---|---|---|---|---|---|---|
| Always `tie` | 0.319 | 0.318–0.320 | | 0.918 | 0.957 | 0.000 | 0.000 |
| Listener: profile (the proposal's profile-only baseline) | 0.342 | 0.330–0.353 | | 0.584 | 0.741 | 0.116 | 0.168 |
| Listener: year | 0.382 | 0.367–0.398 | | 0.788 | 0.887 | 0.118 | 0.141 |
| Text only | 0.374 ± 0.019 | 0.360–0.389 | 0.359, 0.368, 0.395 | 0.715 | 0.832 | 0.134 | 0.157 |
| Text + audience (the proposal's model) | 0.380 ± 0.017 | 0.364–0.396 | 0.363, 0.382, 0.396 | 0.743 | 0.852 | 0.137 | 0.151 |
| Listener: voting record, no text (extension) | 0.423 | 0.408–0.439 | | 0.748 | 0.867 | 0.191 | 0.213 |

| Comparison (first minus second) | Macro-F1 difference | 95% interval | p |
|---|---|---|---|
| Text + audience − Text only | +0.006 | −0.002 to +0.014 | 0.15 |
| Text + audience − Listener: profile | +0.038 | +0.021 to +0.054 | 0.0005 |
| Text + audience − Listener: year | −0.002 | −0.023 to +0.017 | 0.84 |
| Text + audience − Always `tie` | +0.061 | +0.045 to +0.078 | 0.0005 |
| Seed 42 only: Text + audience − the same model with swapped profiles | +0.011 | +0.003 to +0.018 | 0.0035 |

- **The profile brings no clear gain on this task.** With it the macro-F1 is 0.006 higher, the
  interval includes 0 and p = 0.15. Per seed the gain is +0.003, +0.014 and +0.001. On `ddo_switch`
  the same change gave +0.036 AUROC (p = 0.0005).
- **The model reads the profile all the same.** With the profiles swapped between voters, the seed-42
  model falls from 0.363 to 0.351. So the profile changes its answers, but a model without any
  profile does about as well.
- **Neither text model is above the year of the debate.** Logistic regression on the year alone has
  0.382; text + audience is 0.002 below it (p = 0.84) and text only has 0.374 (not tested against
  it). `RESULTS.md` already said this for the text-only model with one seed; three seeds and the
  profile do not change it.
- **Both are above always answering `tie` and above the profile-only baseline.**
- **The seeds differ more than the two models do.** Macro-F1 runs from 0.359 to 0.395 between seeds,
  against a difference of 0.006 between the models. The seeds differ in how often they answer `tie`:
  for 4,560, 4,780 and 6,222 of the 7,091 test votes (text only) and for 4,714, 5,271 and 6,302
  (text + audience). Accuracy follows that: 0.644, 0.670, 0.833 and 0.661, 0.727, 0.841. Seed 44
  answers `tie` most often and has the highest macro-F1 with both models.
- **`pro` and `con` are found poorly by every model** (F1 0.12 to 0.21). The seed-42 text + audience
  model finds 83 of the 256 `pro` votes and 135 of the 327 `con` votes, and answers `pro` or `con`
  for 2,041 of the 6,508 votes that stayed tied.
- **The best row uses no text (extension of the proposal).** Logistic regression on the voter's
  voting record has 0.423; its interval lies above those of both text models. It was not compared
  with them in a paired test, and no model with text and the voting record was trained for this task.
- Seed 42 alone (`outputs/compare_models_undecided.json`): text only 0.359 (0.343–0.376),
  text + audience 0.363 (0.347–0.378), difference +0.003 (−0.007 to +0.013, p = 0.55).

#### The voters who filled in the other criteria

`filled_other_criteria` is true for 6,741 of the 7,091 test votes (95.1%), from 3,536 of the 3,649
test debates: 6,169 `tie`, 252 `pro`, 320 `con`. The rerun leaves out one vote in twenty, almost all
of them `tie`. Test macro-F1, three seeds for the model rows:

| Row | All test votes | Voters who filled in the other criteria |
|---|---|---|
| Always `tie` | 0.319 | 0.319 |
| Listener: profile | 0.342 (0.330–0.353) | 0.345 (0.333–0.357) |
| Listener: year | 0.382 (0.367–0.398) | 0.386 (0.368–0.401) |
| Text only | 0.374 (0.360–0.389) | 0.378 (0.364–0.394) |
| Text + audience | 0.380 (0.364–0.396) | 0.385 (0.370–0.401) |
| Listener: voting record, no text (extension) | 0.423 (0.408–0.439) | 0.432 (0.415–0.449) |

| Comparison on the subset (first minus second) | Macro-F1 difference | 95% interval | p |
|---|---|---|---|
| Text + audience − Text only | +0.007 | −0.002 to +0.015 | 0.095 |
| Text + audience − Listener: profile | +0.040 | +0.023 to +0.057 | 0.0005 |
| Text + audience − Listener: year | −0.001 | −0.021 to +0.019 | 0.95 |
| Text + audience − Always `tie` | +0.067 | +0.050 to +0.083 | 0.0005 |
| Seed 42 only: Text + audience − the same model with swapped profiles | +0.012 | +0.004 to +0.019 | 0.0030 |

- **Nothing changes here either.** Every row moves by at most 0.009 on the subset and every
  comparison keeps its sign and its reading. A tie that means "skipped" and not "undecided" is
  therefore not what holds the scores down, as far as this column can tell: it marks 350 test votes,
  339 of them `tie`.

## Notes

- **The rerun is an evaluation, not a new training run.** The proposal says "we rerun T2 on voters who
  filled in the other criteria". Here the models trained on all voters are scored on those voters.
  Training on them only would be a separate run, which is not done here.
- **Three seeds for both models; the swapped-profile control is seed 42 only.**
- **The text-only model with seed 42 predates `--seed`,** so its classifier's start weights were not
  seeded. Its re-score gives macro-F1 0.3595 and accuracy 0.6436, with validation macro-F1 0.345;
  `RESULTS.md` records 0.360, 64.4% and 0.345. The macro-F1 differs by at most 0.001; whether any
  vote is answered differently was not checked.
- **Where the models are.** Only the text-only model with seed 42 is on the Hub
  (`nlp-final-project11/nlp-project-modernbert-ddo_undecided`). The other five runs first ran with
  `--push`; training finished, and the upload at the end failed because the org's private storage on
  the Hub was full (a run with `--push` uploads the weights at every checkpoint). Started again
  without `--push`, each run continued from its last checkpoint and wrote its scores. Their models
  are in `outputs/<run>/` on Gilbreth scratch, which is purged after 60 days.
- **Not copied from the job logs:** which checkpoint each run selected, and the run times.
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
