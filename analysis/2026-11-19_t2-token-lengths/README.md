# T2: how much of the debates the models read

**Status:** measured on 2026-11-19. The T2 runs read at most 4,096 tokens per vote. For a third of the
`ddo_switch` votes and a fifth of the `ddo_undecided` votes the input is longer, so the end of the
debate is cut. At 8,192 tokens, which ModernBERT can read, it would be about one vote in nine and one
in thirteen. One run at 8,192 tokens (`--max-length 8192`) finished on 2026-11-20: test AUROC 0.752,
against 0.744 ± 0.006 for the three runs at 4,096 (difference +0.008, 95% interval −0.006 to +0.022,
p = 0.26). Reading the whole debate brings no clear gain.

## Goal

The proposal describes the model as ModernBERT, "which reads up to 8k tokens". Every T2 run so far uses
`MAX_LENGTH = 4096` in `scripts/train_delta.py`. This measures how many votes that limit affects.

## What it does

`scripts/t1_t2/token_lengths.py` counts the tokens of the whole input of every vote: the resolution
and the voter's side before the debate, the debate, and the three special tokens the model adds.

- The rows (debate, split, side before) are streamed from the Hub dataset; the transcripts are not
  downloaded.
- The debate text is rebuilt from the raw corpus with `debate_transcript` of
  `scripts/create_dataset.py`, the code that built the Hub configs. Each of the 45,141 debates is
  tokenized once with the ModernBERT tokenizer.

- **`--max-length <N>`** in `scripts/train_delta.py` sets the limit for a run (default 4,096). A run
  with another limit has `-len<N>` in its name, before the seed, for example
  `ddo_switch-audience-self-len8192`, with its own `outputs/` folder, markers and model repo.
  `jobs/train.sub` derives the same name. A smoke run keeps its short inputs.

## Run

```bash
python scripts/t1_t2/token_lengths.py --debates ddo/debates.json --output outputs/token_lengths.json
# the proposal's model, reading up to 8,192 tokens; add --smoke first for a quick check
sbatch -A <account> jobs/train.sub ddo_switch --audience self --max-length 8192 --bf16 --push
# afterwards, with its prediction files copied to outputs/<run>/
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text + audience=outputs/ddo_switch-audience-self,outputs/ddo_switch-audience-self-seed43,outputs/ddo_switch-audience-self-seed44" \
    --test-scores "Text + audience, 8,192 tokens=outputs/ddo_switch-audience-self-len8192/predictions_test.parquet" \
    --compare "Text + audience, 8,192 tokens" "Text + audience" \
    --output outputs/compare_models_len8192.json
# the same against the seed-42 run at 4,096 tokens alone
python scripts/t1_t2/compare_models.py --users ddo/users.json \
    --model "Text + audience, seed 42=outputs/ddo_switch-audience-self" \
    --test-scores "Text + audience, 8,192 tokens=outputs/ddo_switch-audience-self-len8192/predictions_test.parquet" \
    --compare "Text + audience, 8,192 tokens" "Text + audience, seed 42" \
    --output outputs/compare_models_len8192_seed42.json
```

## Results

Tokens per vote, and the share of votes whose input is longer than each limit:

| Config | Split | Votes | Median | 90th percentile | Over 4,096 | Over 8,192 |
|---|---|---|---|---|---|---|
| `ddo_switch` | train | 48,479 | 2,428 | 8,804 | 33.7% | 11.7% |
| `ddo_switch` | validation | 6,085 | 2,380 | 8,997 | 32.8% | 12.7% |
| `ddo_switch` | test | 5,845 | 2,370 | 8,390 | 32.2% | 11.1% |
| `ddo_undecided` | train | 58,156 | 1,570 | 7,410 | 23.1% | 8.1% |
| `ddo_undecided` | validation | 7,163 | 1,548 | 7,209 | 22.6% | 8.1% |
| `ddo_undecided` | test | 7,091 | 1,555 | 7,115 | 21.4% | 7.3% |

- **Half of the votes are on short debates** (median about 2,400 tokens for `ddo_switch`, 1,600 for
  `ddo_undecided`), which the models read whole.
- **A third of the `ddo_switch` votes are cut at 4,096 tokens.** Going to 8,192 would let the model
  read all of the debate for another fifth of the votes; 11% to 13% would still be cut.
- **What is cut is the end of the debate.** The tokenizer shortens the longer of the two texts, which
  is the debate, from its end, so the last rounds are lost.

### The run at 8,192 tokens

Text + audience on `ddo_switch`, seed 42, the settings of the runs at 4,096 tokens except the limit.
AUROC on the 5,845 test votes with a 95% interval from a bootstrap over debates:

| Run | Validation | Test | 95% interval |
|---|---|---|---|
| 4,096 tokens, average of seeds 42, 43, 44 | 0.729 | 0.744 ± 0.006 (0.737, 0.746, 0.749) | 0.716–0.767 |
| 4,096 tokens, seed 42 | 0.713 | 0.737 | 0.709–0.764 |
| 8,192 tokens, seed 42 | 0.738 | 0.752 | 0.724–0.776 |

| Comparison (first minus second) | AUROC difference | 95% interval | p |
|---|---|---|---|
| 8,192 tokens − 4,096 tokens, three seeds | +0.008 | −0.006 to +0.022 | 0.26 |
| 8,192 tokens − 4,096 tokens, seed 42 | +0.015 | −0.003 to +0.033 | 0.12 |

- **No clear gain from reading twice as much.** On test the run at 8,192 tokens is above each of the
  three runs at 4,096; on validation it is above two of them (0.713, 0.733 and 0.741 at 4,096). Both
  intervals include 0. One seed cannot show a gain this small: the three seeds at 4,096 differ by
  0.012 among themselves on test.
- **The limit of 4,096 costs little, if anything.** The intervals end at +0.022 and +0.033. The T2
  results stand as reported at 4,096 tokens; that the runs read half of what the proposal names
  ("up to 8k tokens") is a deviation to state in the report, with this check.
- **Not looked at:** whether the gain sits in the third of the votes whose input is cut at 4,096.

## Notes

- **Cost of the run at 8,192 tokens.** It needed at least four chained 4-hour standby jobs (Slurm
  12446747 to 12446750; the first three hit the time limit), against two for a run at 4,096. A step
  takes longer, and a job that is cut off loses the steps since its last checkpoint. The exact GPU
  time is not copied from the job logs. The runs at 4,096 tokens peaked at 14.9 GiB of GPU memory
  with a batch of four.
- **Where its model is.** The final model is in `outputs/ddo_switch-audience-self-len8192/` on
  Gilbreth scratch, which is purged after 60 days. The Hub repo
  `nlp-final-project11/nlp-project-modernbert-ddo_switch-audience-self-len8192` holds a checkpoint
  from step 4,000 of the run, not the final model (checked on 2026-11-20).
- **`--max-length`, checked on made-up rows:** a higher limit changes the scores of debates that were
  cut; the default written out gives the plain name and identical scores; `--eval-only` reproduces
  the run; `jobs/train.sub` and the script derive the same run name in 12 argument forms. Runs without
  the option give exactly the same scores as before (14 files, difference 0).
- **The numbers are per vote, not per debate.** A debate with many voters counts once per voter.
- **The profile line adds 35 tokens** in a text + audience run (6 for a voter without a profile); it
  is not included above.
- **The count equals the training script's.** An input is the first text, the debate and three
  special tokens; the script measures that number from the tokenizer instead of assuming it. All
  debates of the two configs were found in the raw corpus.
