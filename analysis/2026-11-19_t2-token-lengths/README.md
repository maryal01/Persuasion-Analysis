# T2: how much of the debates the models read

**Status:** measured on 2026-11-19. The T2 runs read at most 4,096 tokens per vote. For a third of the
`ddo_switch` votes and a fifth of the `ddo_undecided` votes the input is longer, so the end of the
debate is cut. At 8,192 tokens, which ModernBERT can read, it would be about one vote in nine and one
in thirteen. One run at 8,192 tokens is prepared (`--max-length 8192`) and not finished.

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

## Notes

- **Whether this costs accuracy is not measured yet.** One run is planned: text + audience on
  `ddo_switch`, seed 42, at 8,192 tokens. If its score equals that of the runs at 4,096 (0.744 ± 0.006
  over three seeds), the shorter limit is justified; one seed cannot show a small gain.
- **Cost of that run, estimated, not measured.** The runs at 4,096 tokens peaked at 14.9 GiB of GPU
  memory with a batch of four; four times that still fits an 80 GB A100. A step should take about
  twice as long, and a 4-hour job then saves fewer checkpoints, so the run needs about five chained
  jobs instead of two.
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
