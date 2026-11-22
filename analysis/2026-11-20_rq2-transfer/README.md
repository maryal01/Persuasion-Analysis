# RQ2 transfer between CMV and DDO: common format, runs and a length-only baseline

**Status:** four runs trained and compared with three seeds each (42, 43, 44) on 2026-11-21, text only.
Persuasion signals transfer between the platforms, weakly: a model trained on CMV scores AUROC 0.586 on
the DDO test votes and a model trained on DDO 0.556 on the CMV test pairs. Both are above a length-only
baseline (0.53 and 0.52) and far below a model trained on the platform itself (0.703 and 0.648). CMV
data does not help on DDO: −0.003 AUROC for CMV first, then DDO (p = 0.43) and −0.003 for training on
both (p = 0.47). No audience profile is in the input yet. The three configs are on branch `rq2` of the
private dataset repo (commit `007570a`, uploaded 2026-11-20).

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

**`scripts/t1_t2/transfer_scores.py`:** the result tables. It reads the saved scores of the runs on one
platform's test split and adds the length-only baseline as rows. Every row gets a 95% interval from a
bootstrap over debates (DDO) or posts (CMV), and every comparison a paired permutation test that swaps
whole debates or posts. Both are the functions of `scripts/t1_t2/compare_models.py`. On CMV it also
reports the pairwise accuracy of T1: how often the reply that earned the delta scores above the other
reply of its pair.

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

Seeds 43 and 44 are the same eight commands with `--seed <N>` added and the model of that seed named:
`--init-from outputs/cmv_common-seed<N>`, `--eval-only outputs/ddo_common-seed<N>` and so on. The runs
get `-seed<N>` at the end of their name. "CMV first, then DDO" gets it twice, once from the name of its
start model: `outputs/ddo_common-init-cmv_common-seed<N>-seed<N>`.

Then, with the prediction files copied to `outputs/<run>/`, the two result tables. They read local files
only: the build in `outputs/common` and the prediction files.

```bash
python scripts/t1_t2/transfer_scores.py --platform ddo \
    --model "DDO only=outputs/ddo_common,outputs/ddo_common-seed43,outputs/ddo_common-seed44" \
    --model "CMV first, then DDO=outputs/ddo_common-init-cmv_common,outputs/ddo_common-init-cmv_common-seed43-seed43,outputs/ddo_common-init-cmv_common-seed44-seed44" \
    --model "CMV and DDO together=outputs/joint_common,outputs/joint_common-seed43,outputs/joint_common-seed44" \
    --model "CMV only, scored on DDO=outputs/ddo_common-by-cmv_common,outputs/ddo_common-by-cmv_common-seed43,outputs/ddo_common-by-cmv_common-seed44" \
    --model "Full debate (T2 text only)=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44" \
    --compare "CMV only, scored on DDO" "Length only, fitted on CMV" \
    --compare "DDO only" "Length only, fitted on DDO" \
    --compare "CMV first, then DDO" "CMV and DDO together" \
    --output outputs/transfer_scores_ddo.json

python scripts/t1_t2/transfer_scores.py --platform cmv \
    --model "CMV only=outputs/cmv_common,outputs/cmv_common-seed43,outputs/cmv_common-seed44" \
    --model "CMV and DDO together, scored on CMV=outputs/cmv_common-by-joint_common,outputs/cmv_common-by-joint_common-seed43,outputs/cmv_common-by-joint_common-seed44" \
    --model "DDO only, scored on CMV=outputs/cmv_common-by-ddo_common,outputs/cmv_common-by-ddo_common-seed43,outputs/cmv_common-by-ddo_common-seed44" \
    --model "CMV first, then DDO, scored on CMV=outputs/cmv_common-by-ddo_common-init-cmv_common,outputs/cmv_common-by-ddo_common-init-cmv_common-seed43,outputs/cmv_common-by-ddo_common-init-cmv_common-seed44" \
    --compare "CMV only" "Length only, fitted on CMV" \
    --compare "DDO only, scored on CMV" "Length only, fitted on DDO" \
    --compare "CMV first, then DDO, scored on CMV" "DDO only, scored on CMV" \
    --output outputs/transfer_scores_cmv.json
```

Validation scores and the combination with the voter's voting record (an extension of the proposal)
come from the T2 comparison script, on the DDO side only (sandbox off, it reads ids from the Hub):

```bash
python scripts/t1_t2/compare_models.py --users ddo/users.json --dataset-config ddo_common --dataset-revision rq2 \
    --model "DDO only=outputs/ddo_common,outputs/ddo_common-seed43,outputs/ddo_common-seed44" \
    --model "CMV first, then DDO=outputs/ddo_common-init-cmv_common,outputs/ddo_common-init-cmv_common-seed43-seed43,outputs/ddo_common-init-cmv_common-seed44-seed44" \
    --model "CMV and DDO together=outputs/joint_common,outputs/joint_common-seed43,outputs/joint_common-seed44" \
    --model "Full debate (T2 text only)=outputs/ddo_switch,outputs/ddo_switch-seed43,outputs/ddo_switch-seed44" \
    --test-scores "CMV only, scored on DDO=outputs/ddo_common-by-cmv_common/predictions_test.parquet,outputs/ddo_common-by-cmv_common-seed43/predictions_test.parquet,outputs/ddo_common-by-cmv_common-seed44/predictions_test.parquet" \
    --output outputs/compare_models_transfer.json
```

One seed alone: the same commands with one folder per model.

## Results

Three runs per model, seeds 42, 43 and 44, trained on Gilbreth on 2026-11-20 and 21. Seed N of "CMV
first, then DDO" is the CMV model of seed N trained further on DDO, and each transfer score is from the
model of the same seed. A number is the average over the three seeds; its 95% interval is that of the
average on the test rows and does not cover the spread between seeds, which is listed. AUROC is how
often a persuaded row scores above one that was not. F1, precision and recall are for the model's own
answer, the class with the larger logit.

### On the DDO test votes

5,845 votes from 2,504 debates; 368 switched sides (6.3%).

| Trained on | Validation AUROC | Test AUROC | 95% interval | Per seed (42, 43, 44) | F1 of the persuaded class (precision, recall) |
|---|---|---|---|---|---|
| DDO only (the reference) | 0.703 | 0.703 | 0.674–0.732 | 0.696, 0.706, 0.707 | 0.191 (0.114, 0.584) |
| CMV first, then DDO | 0.700 | 0.700 | 0.671–0.729 | 0.707, 0.703, 0.690 | 0.197 (0.121, 0.586) |
| CMV and DDO together | 0.710 | 0.700 | 0.671–0.728 | 0.703, 0.696, 0.701 | 0.194 (0.189, 0.223) |
| CMV only | | 0.586 | 0.555–0.621 | 0.587, 0.580, 0.591 | 0.135 (0.076, 0.599) |
| Length only, fitted on DDO | | 0.533 | 0.500–0.564 | | |
| Length only, fitted on CMV | | 0.532 | 0.500–0.564 | | |
| Full debate: T2 text only on `ddo_switch` | 0.699 | 0.708 | 0.679–0.736 | 0.719, 0.697, 0.709 | |

| Comparison (first minus second) | AUROC difference | 95% interval | p |
|---|---|---|---|
| CMV first, then DDO − DDO only | −0.003 | −0.010 to +0.005 | 0.43 |
| CMV and DDO together − DDO only | −0.003 | −0.011 to +0.005 | 0.47 |
| CMV first, then DDO − CMV and DDO together | +0.000 | −0.009 to +0.009 | 0.94 |
| CMV only − DDO only | −0.117 | −0.144 to −0.088 | 0.0005 |
| CMV only − Length only, fitted on CMV | +0.054 | +0.014 to +0.096 | 0.017 |
| DDO only − Length only, fitted on DDO | +0.170 | +0.129 to +0.211 | 0.0005 |
| Full debate − DDO only | +0.005 | −0.019 to +0.029 | 0.66 |

### On the CMV test pairs

2,148 replies in 1,074 pairs, from 615 posts; half of the replies earned a delta by construction.

| Trained on | Test AUROC | 95% interval | Per seed (42, 43, 44) | Pairwise accuracy | 95% interval | Per seed (42, 43, 44) |
|---|---|---|---|---|---|---|
| CMV only (the reference) | 0.648 | 0.628–0.670 | 0.645, 0.650, 0.649 | 67.3% | 64.6–70.0 | 67.2, 68.3, 66.3 |
| CMV and DDO together | 0.648 | 0.627–0.669 | 0.647, 0.652, 0.647 | 67.2% | 64.3–69.7 | 67.4, 67.6, 66.6 |
| DDO only | 0.556 | 0.538–0.573 | 0.556, 0.556, 0.556 | 58.2% | 55.2–60.9 | 57.0, 57.8, 59.7 |
| CMV first, then DDO | 0.574 | 0.557–0.591 | 0.570, 0.583, 0.569 | 60.5% | 58.0–63.1 | 59.1, 61.6, 60.8 |
| Length only, fitted on DDO | 0.516 | 0.498–0.531 | | 52.9% | 49.6–56.0 | |
| Length only, fitted on CMV | 0.523 | 0.501–0.543 | | 52.9% | 49.6–56.0 | |

| Comparison (first minus second) | AUROC difference (95% interval) | p | Pairwise accuracy difference (95% interval) | p |
|---|---|---|---|---|
| CMV and DDO together − CMV only | +0.000 (−0.009 to +0.010) | 0.94 | −0.1 points (−1.9 to +1.6) | 0.96 |
| DDO only − CMV only | −0.092 (−0.114 to −0.071) | 0.0005 | −9.1 points (−12.6 to −5.9) | 0.0005 |
| CMV first, then DDO − CMV only | −0.074 (−0.093 to −0.056) | 0.0005 | −6.7 points (−9.5 to −4.2) | 0.0005 |
| CMV first, then DDO − DDO only | +0.018 (+0.011 to +0.027) | 0.0005 | +2.4 points (+0.7 to +4.2) | 0.005 |
| DDO only − Length only, fitted on DDO | +0.040 (+0.024 to +0.055) | 0.0005 | +5.3 points (+2.3 to +8.3) | 0.001 |
| CMV only − Length only, fitted on CMV | +0.125 (+0.098 to +0.154) | 0.0005 | +14.4 points (+10.0 to +18.6) | 0.0005 |

### What the two tables say

- **Persuasion signals transfer in both directions, weakly.** A model that never saw DDO scores 0.586
  on the DDO test votes, and a model that never saw CMV scores 0.556 on the CMV test pairs (pairwise
  58.2%). Both are above the length-only baseline fitted on the same training platform (+0.054,
  p = 0.017, and +0.040, p = 0.0005), so text length does not explain the transfer.
- **Most of the signal is specific to the platform.** The transferred models are 0.117 (on DDO) and
  0.092 (on CMV) below the model trained on the platform itself.
- **CMV data does not help on DDO.** CMV first, then DDO is −0.003 against DDO only (−0.010 to +0.005,
  p = 0.43), and training on both is −0.003 (−0.011 to +0.005, p = 0.47). The intervals leave room for
  a gain of at most 0.005. Seed 42 alone had shown +0.011 and +0.007; the other two seeds did not
  repeat it.
- **Training on both gives one model for both platforms at no measured cost:** 0.700 on DDO and 0.648
  on CMV, against 0.703 and 0.648 for the two models trained on one platform.
- **Training on DDO afterwards removes most of what the CMV training taught, not all of it.** The CMV
  model drops from 0.648 to 0.574 on CMV. That is still 0.018 above the model that never saw CMV
  (+0.011 to +0.027, p = 0.0005; pairwise +2.4 points, p = 0.005).
- **The common format loses nothing measurable against the full debate.** `ddo_common` shows the model
  only the arguments of the side the voter did not hold, less than half the tokens of `ddo_switch`.
  Its 0.703 is 0.005 below the full-debate model's 0.708 (−0.019 to +0.029, p = 0.66).
- **With the voting record (an extension of the proposal, not part of its audience description)** the
  four DDO models end at the same place: combined with the voter's voting record they score 0.834
  (DDO only), 0.833 (CMV first, then DDO), 0.833 (both) and 0.833 (full debate), against 0.790 for the
  voting record alone.

### The datasets

As built from Hub branch `main` on 2026-11-20:

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

### Length-only baseline

Test AUROC with a 95% interval, and for CMV the share of pairs in which the reply that earned the delta
scores higher.

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
  have none so far. On DDO, inferred profiles in place of the reported ones lowered the score
  (`analysis/2026-11-19_t2-compare-models/`), and inferred profiles only for the voters who reported
  none raised it (`analysis/2026-11-20_t2-audience-filled/`).
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
- **Three seeds.** The standard deviation between seeds on the DDO test votes is 0.006 (DDO only),
  0.009 (CMV first, then DDO), 0.004 (both) and 0.006 (CMV only), and on the CMV test pairs 0.003
  (CMV only), 0.003 (both), 0.000 (DDO only) and 0.008 (CMV first, then DDO). Every difference between
  the three DDO-trained models on DDO is smaller than that spread.
- **F1, precision and recall depend on where the score is cut,** and that differs between the runs.
  The joint model marks 99% of the persuaded CMV replies and 97% of the others as persuaded, and on
  DDO it finds 22% of the switchers where the DDO-only model finds 58% (averages over the seeds). A
  likely reason, not tested: class weights follow the share of persuaded rows in the training set,
  which is 6.5% on DDO, 50% on CMV and 17.4% in the joint set. AUROC and pairwise accuracy do not
  depend on the cut and are the numbers to compare across rows.
- **T1's first run reports a pairwise accuracy of 66.1% and an AUROC of 0.629** with the whole post
  as the first text (`RESULTS.md`); the CMV-only model here reaches 67.3% and 0.648 over three seeds
  with the title alone. The two are in the same range and are not a paired comparison: that run used
  another split (1,019 test pairs on Hub branch `t1-t2`, against 1,074 here).
- **The length-only baseline rules out length, not every surface cue.** Topic words or platform style
  were not tested.
- **Scores, checked:** every AUROC and pairwise accuracy of `transfer_scores.py`, seed by seed, equals
  the `test_metrics.json` that the run wrote on Gilbreth; its DDO rows and intervals equal those of
  `compare_models.py`, and its length rows those of `length_baseline.py`; a direct calculation from
  three prediction files gives the same AUROC; a prediction file from another config, or with two rows
  swapped, is refused. The same comparison has slightly different p values in the two scripts (0.66
  and 0.69 for full debate − DDO only) because the random swaps are drawn in another order.
- **Runs, checked from their `test_metrics.json`:** every "CMV first, then DDO" run started from the
  CMV model of its own seed, and each transfer score names the model it scored, of the same seed.
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
