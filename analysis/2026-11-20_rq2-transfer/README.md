# RQ2 transfer between CMV and DDO: common format, runs and a length-only baseline

**Status:** built and checked on 2026-11-20; no model is trained yet. CMV persuasion (T1) and DDO
switching (T2) are written in one format, the training script can start from another model and keep
transfer scores apart, and four runs are prepared. A baseline that sees only text lengths scores AUROC
0.53 on DDO and 0.52 on CMV, whichever platform it was fitted on. The three configs are on branch `rq2`
of the private dataset repo (commit `007570a`, uploaded 2026-11-20).

## Goal

The second half of RQ2 in the proposal: do persuasion signals transfer between the platforms, and does
training on CMV's plentiful mind-change data help predict DDO's rarer switches? The proposal casts both
tasks "in a common format, the inferred profile plus the arguments for the other side, labeled persuaded
or not", and compares (a) training on one platform and testing on the other and (b) training on both,
including CMV first and DDO second. A length-only baseline rules out surface cues.

This first step is text only. The inferred profile is left out until CMV posters have one.

## What it does

**The common format** (`scripts/t1_t2/build_common.py`):

| | DDO | CMV |
|---|---|---|
| First text | `Claim: <resolution>` and `Position before: Pro` or `Con` (the voter's side) | `Claim: <post title>` and `Position before: Pro` (a poster holds their own claim) |
| Second text | every turn of the debater on the side the voter did not hold, in round order | the reply branch of one challenger, as in `cmv` |
| Label | `persuaded` = switched sides | `persuaded` = the reply earned a delta |

Three configs:

- **`ddo_common`:** the rows, splits and other columns of `ddo_switch`, so its test votes are the ones
  the T2 models are scored on and the two can be compared vote by vote.
- **`cmv_common`:** the rows, splits and other columns of `cmv` on Hub branch `main` (split by the
  post's author).
- **`joint_common`:** both training splits together; validation and test are those of `ddo_common`,
  because the question is whether CMV data helps on DDO.

Rows, labels and splits are read from the Hub configs. Only the texts are rebuilt from the raw corpora:
the turns of each side from `debates.json`, the post titles from `pairs.jsonl`.

**Two options in `scripts/train_delta.py`:**

- **`--init-from MODEL`** starts training from a fine-tuned model (a path or a Hub repo) instead of the
  pretrained encoder: the CMV model, before it is trained on DDO. The model's classes must be the
  dataset's. The run is named `<config>-init-<model name>`.
- **`--tag TEXT`** appends `-<TEXT>` to the run name. With `--eval-only MODEL` it gives a model that
  was trained on another dataset its own `outputs/` folder, so it does not overwrite the scores of the
  model trained on this config.

`jobs/train.sub` derives the same run names. Runs without the two options are unchanged.

**`scripts/t1_t2/length_baseline.py`:** logistic regression on two numbers per row, the words of the
arguments and the words of the claim, fitted on one platform or on both and scored on each.

## Run

Build and check the datasets on a laptop with the raw files, then upload them to a staging branch:

```bash
python scripts/t1_t2/build_common.py --out outputs/common               # build, upload nothing
python scripts/t1_t2/length_baseline.py --data outputs/common --output outputs/length_baseline.json
python scripts/t1_t2/build_common.py --out outputs/common --push rq2    # upload to Hub branch rq2
```

The four runs on Gilbreth (a smoke run first; chain parts as for the T2 runs). They run without
`--push`, so each model stays in `outputs/<run>/`:

```bash
sbatch -A <account> jobs/train.sub cmv_common   --dataset-revision rq2 --bf16     # CMV only
sbatch -A <account> jobs/train.sub ddo_common   --dataset-revision rq2 --bf16     # DDO only, the reference
sbatch -A <account> jobs/train.sub joint_common --dataset-revision rq2 --bf16     # both training sets
# CMV first, then DDO: after the CMV run has finished
sbatch -A <account> jobs/train.sub ddo_common   --dataset-revision rq2 --init-from outputs/cmv_common --bf16
```

Transfer scores, a short job each, after the runs:

```bash
# the CMV model on the DDO test votes, and the DDO models on the CMV test pairs
sbatch -A <account> jobs/train.sub ddo_common --dataset-revision rq2 --eval-only outputs/cmv_common --tag by-cmv_common --bf16
sbatch -A <account> jobs/train.sub cmv_common --dataset-revision rq2 --eval-only outputs/ddo_common --tag by-ddo_common --bf16
sbatch -A <account> jobs/train.sub cmv_common --dataset-revision rq2 --eval-only outputs/joint_common --tag by-joint_common --bf16
sbatch -A <account> jobs/train.sub cmv_common --dataset-revision rq2 --eval-only outputs/ddo_common-init-cmv_common --tag by-ddo_common-init-cmv_common --bf16
```

Then, with the prediction files copied to `outputs/<run>/` (sandbox off):

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json --dataset-config ddo_common --dataset-revision rq2 \
    --model "DDO only=outputs/ddo_common" \
    --model "CMV first, then DDO=outputs/ddo_common-init-cmv_common" \
    --model "CMV and DDO together=outputs/joint_common" \
    --model "Full debate (T2 text only)=outputs/ddo_switch" \
    --test-scores "CMV only, scored on DDO=outputs/ddo_common-by-cmv_common/predictions_test.parquet" \
    --output outputs/compare_models_transfer.json
```

The CMV side of the table (AUROC and pairwise accuracy of each model on the CMV test pairs) is in the
`test_metrics.json` of the four `cmv_common*` folders.

## Results

No model scores yet. The datasets, as built from Hub branch `main` on 2026-11-20:

| Config | Split | Rows | Persuaded | Rows without arguments | Median tokens | Over 4,096 tokens |
|---|---|---|---|---|---|---|
| `ddo_common` | train | 48,479 | 3,133 (6.5%) | 61 | 1,124 | 12.8% |
| `ddo_common` | validation | 6,085 | 366 (6.0%) | 1 | 1,075 | 13.6% |
| `ddo_common` | test | 5,845 | 368 (6.3%) | 5 | 1,074 | 11.7% |
| `cmv_common` | train | 16,356 | 8,178 (50.0%) | 0 | 320 | 0.2% |
| `cmv_common` | validation | 2,060 | 1,030 (50.0%) | 0 | 311 | 0.2% |
| `cmv_common` | test | 2,148 | 1,074 (50.0%) | 0 | 324 | 0.3% |
| `joint_common` | train | 64,835 | 11,311 (17.4%) | 61 | 727 | 9.6% |

- **A DDO input is less than half as long as in `ddo_switch`** (median 1,074 tokens against 2,370 on
  the test split), and 12% of the votes are cut at 4,096 tokens against 32%.
- **CMV replies are short** (median about 320 tokens), so the two platforms differ in length as well
  as in how often people are persuaded (50% of the CMV rows by construction, 6% of the DDO rows).

Length-only baseline: test AUROC with a 95% interval, and for CMV the share of pairs in which the reply
that earned the delta scores higher.

| Fitted on | DDO test AUROC | CMV test AUROC | CMV pairwise accuracy |
|---|---|---|---|
| DDO | 0.533 (0.500–0.564) | 0.516 (0.498–0.531) | 0.529 |
| CMV | 0.532 (0.500–0.564) | 0.523 (0.501–0.543) | 0.529 |
| both | 0.499 (0.466–0.533) | 0.496 (0.490–0.502) | 0.471 |

- **Length is a weak cue, and it does transfer.** Longer arguments go a little more often with
  persuasion on both platforms, and a length model fitted on one platform scores the same on the other
  (0.53 on DDO, 0.52 on CMV). A text model that transfers has to beat these numbers.
- **Fitted on both platforms, the length model is at chance within each.** It learns to tell the
  platforms apart instead (short arguments and long claims are CMV, where half the rows are
  persuaded). The joint text model can do the same, which is why it is scored within one platform.

## Notes

- **Design choices that are ours, not the proposal's.** The CMV claim is the post title only; the body
  of the post, where the poster gives their reasons, is left out so that both platforms give the model
  the same kind of first text. `Position before` is always `Pro` on CMV. The joint run selects its
  checkpoint on DDO validation.
- **No audience profile yet.** The proposal's common format includes the inferred profile. CMV posters
  have none so far, and on DDO the inferred profiles lowered the score
  (`analysis/2026-11-19_t2-compare-models/`).
- **Class weights follow the training set of each run.** In the joint run 17.4% of the rows are
  persuaded, most of them CMV rows.
- **Rows without arguments are kept:** 67 DDO votes whose other side forfeited every round. The model
  then sees the claim alone. They stay so that `ddo_common` has exactly the rows of `ddo_switch`.
- **Three turns in three debates have a side that is neither Pro nor Con** in the raw corpus (a slip
  in its parsing). They belong to nobody and are left out.
- **Cost, estimated from the token counts, not measured:** the T2 runs took about 7 hours for 48,479
  votes of 2,431 tokens on average (after the cut). By that measure: DDO only about 4.5 hours, CMV
  only under 1 hour, both together about 5 hours, CMV then DDO about 4.5 hours; 15 to 17 GPU hours for
  one seed.
- **One seed first.** The text models of T2 moved by up to 0.02 AUROC between seeds, so differences of
  that size between the four runs will need more seeds.
- **Checked on the real data:** `ddo_common` has the rows, order, ids and labels of `ddo_switch`;
  every paragraph of a sampled reply (every seventh row) is in the full transcript and starts a turn
  of the other side; against `ddo_history`, which another script built, the text of a debate side is
  equal for 29,931 of the 32,078 sides and differs for none (the other 2,147 sides have no row
  there); `cmv_common` has the rows, replies, labels and pairs of `cmv`, and its claim is the start of
  the post on every row.
- **The upload, checked:** every column of every split on Hub branch `rq2` equals the local build;
  `main` and the other branches are unchanged; a smoke run of the training script on the Hub copy of
  `cmv_common` and `joint_common` passes on a CPU.
- **Checked without a GPU,** on made-up rows: runs without the new options give exactly the same
  scores before and after this change; a run with `--init-from` starts from the named model, has its
  own folder and saves its model; a model scored on another config with `--tag` does not overwrite
  that config's own scores; a model scored again on its own test split reproduces its scores; the
  joint config trains with empty DDO columns on its CMV rows; a start model with other classes is
  refused; `jobs/train.sub` and the script derive the same run name in 12 argument forms.
- **The prediction files hold voter names** and stay in `outputs/`, which git ignores.
