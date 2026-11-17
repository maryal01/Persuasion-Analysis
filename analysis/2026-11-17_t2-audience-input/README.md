# T2 text + audience: the voter's profile in front of the debate

**Status:** code written and checked on a laptop (2026-11-17). No text + audience model is trained yet:
the runs need GPU jobs on Gilbreth. This first version uses self-reported profiles only.

## Goal

The proposal's last model reads the debate and a description of the audience member. This adds that
input to the T1/T2 training script, plus the two things needed to compare models properly: test
predictions saved per row, and a way to score a trained model again without training.

## What it does

- **`scripts/common/audience.py`** turns a voter's six T3 labels in `ddo_users` (economy, social issues,
  personal liberties, foreign policy, religion, political ideology group) into one line, for example
  `Voter profile: economy: left; social issues: liberal; personal liberties: permissive; foreign policy:
  dovish; religion: nonreligious; political ideology group: liberal`. An unknown label is written as
  `unknown`; a voter with no label at all gets `Voter profile: unknown`. Column and class names are read
  from the dataset. It sits in `scripts/common/` because T4 and the simulated audience need the same line.
- **`--audience self`** in `scripts/train_delta.py` appends that line to the first text of each row, so
  the model input is: resolution, the voter's side before the debate, the voter's profile, then the
  debate. Such a run is named `<config>-audience-self`: its own `outputs/` folder and its own model repo,
  so it never mixes with the text-only run. `jobs/train.sub` derives the same name for its markers.
- **`--eval-only [MODEL]`** skips training and scores the test split with a trained model: a path or a
  Hub repo, by default the repo the run pushes to. It also works for the existing text-only models.
- **`--shuffle-audience`** (with `--eval-only`) is the control from the proposal: every test voter gets
  another voter's profile, the same one on all their rows. A model that uses the profile should score
  lower.
- **Every run now writes** `predictions_test.parquet` (ids, voter, label and the logits of every test
  row) and `test_metrics.json` to its `outputs/` folder, and prints precision, recall and F1 per class.
  The shuffled control writes `predictions_test_shuffled.parquet` and `test_metrics_shuffled.json`.

## Run

```bash
python scripts/common/audience.py --dataset-config ddo_switch          # coverage only, changes nothing

# train with the profile; add --smoke first for a quick check
sbatch -A <account> jobs/train.sub ddo_switch --audience self --bf16 --push
# control: the same model, profiles swapped between voters
sbatch -A <account> jobs/train.sub ddo_switch --audience self --eval-only --shuffle-audience --bf16
# save the test predictions of the existing text-only model
sbatch -A <account> jobs/train.sub ddo_switch --eval-only --bf16
```

The same commands work for `ddo_undecided`. Profiles are read from `ddo_users` on Hub branch `main`.

## Results

No model results yet. Coverage of the self-reported profiles, from `scripts/common/audience.py`:

| Config | Split | Rows | Voter has a known profile | All six labels known |
|---|---|---|---|---|
| `ddo_switch` | train | 48,479 | 79.1% | 49.9% |
| `ddo_switch` | validation | 6,085 | 79.2% | 50.6% |
| `ddo_switch` | test | 5,845 | 78.3% | 49.0% |
| `ddo_undecided` | train | 58,156 | 82.0% | 50.4% |
| `ddo_undecided` | validation | 7,163 | 81.5% | 50.7% |
| `ddo_undecided` | test | 7,091 | 81.7% | 50.5% |

- About a fifth of the rows get `Voter profile: unknown`. Those are the voters the T3 model is meant to
  profile from their text.
- Swapping profiles between voters changes the line on 92% to 96% of test rows; the rest are swaps
  between two voters with the same line.

## Notes

- **What to compare a trained model with.** The text-only model (`RESULTS.md`), and the listener-only
  baselines of `analysis/2026-11-16_t2-listener-baseline/`, which use the raw profile fields and the
  voter's voting record without any text.
- **The profile is coarse on purpose.** It uses the T3 labels, not the raw profile fields, so that a
  self-reported profile and one inferred by the T3 model have the same form and can be swapped.
- **Inferred profiles are not included yet.** `--audience` accepts only `self` until the T3 model exists.
- **Checked without a GPU,** on made-up rows: a run with `--audience self` trains and saves; `--eval-only`
  on the saved model reproduces its test logits exactly; with `--shuffle-audience` the logits change on
  exactly the rows whose profile line changed; a three-class config runs and prints per-class numbers.
  The run-name rule in `jobs/train.sub` was tested for each argument form.
- **Not checked:** a real run on a GPU, and `--eval-only` against the pushed text-only models.
