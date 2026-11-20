# T2: how much of the debates the models read

**Status:** measured on 2026-11-19. The T2 runs read at most 4,096 tokens per vote. For a third of the
`ddo_switch` votes and a fifth of the `ddo_undecided` votes the input is longer, so the end of the
debate is cut. At 8,192 tokens, which ModernBERT can read, it would be about one vote in nine and one
in thirteen. No run at 8,192 exists.

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

## Run

```bash
python scripts/t1_t2/token_lengths.py --debates ddo/debates.json --output outputs/token_lengths.json
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

- **Whether this costs accuracy is not measured.** There is no run at 8,192 tokens. Such a run needs
  an option for the limit in `scripts/train_delta.py`, more memory and more time per step.
- **The numbers are per vote, not per debate.** A debate with many voters counts once per voter.
- **The profile line adds 35 tokens** in a text + audience run (6 for a voter without a profile); it
  is not included above.
- **The count equals the training script's.** An input is the first text, the debate and three
  special tokens; the script measures that number from the tokenizer instead of assuming it. All
  debates of the two configs were found in the raw corpus.
