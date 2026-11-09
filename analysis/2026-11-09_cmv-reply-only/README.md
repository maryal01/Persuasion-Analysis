# T1 ablation: reply-only ModernBERT (no original post)

**Status:** implementation and smoke test only (2026-11-09).

- **Full training:** not started; waiting for the configuration to be approved.
- **Final test results:** pending. The official test split hasn't been scored by a reply-only model.
- **`RESULTS.md`:** no row yet; one will be added after the full training and evaluation.

## Goal

Does including the original r/ChangeMyView post help ModernBERT predict which reply earns a delta? The
published T1 model reads `tok(original, reply)` and gets 66.14% pairwise accuracy and AUROC 0.6286
on 1,019 test pairs. This run trains a fresh ModernBERT-base on `tok(reply)` only and changes as
little else as possible.

## What it does

`scripts/t1_t2/train_cmv_reply_only.py` has two stages:

- **`train`:**
  - Starts from `answerdotai/ModernBERT-base`, not the fine-tuned CMV checkpoint.
  - Checks that the splits are intact: every pair is complete within its split, and no post or pair
    appears in two splits.
  - Trains on `train`, evaluates on `validation` every 500 optimizer steps, keeps the best
    checkpoint by the selection metric, and saves it to `outputs/cmv_reply_only/full/final`.
  - Never loads model outputs for the test split.
- **`evaluate`:**
  - Scores the test split once; a guard refuses a second scoring unless `--allow-reevaluate` is passed.
  - Uses `train_delta.compute_metrics` and `train_delta.pairwise_accuracy`.
  - Reports 95% post-cluster bootstrap CIs, plus the paired difference to the published model's cached
    test scores (`outputs/cmv_length/predictions.csv`) on the same pairs.

### Settings compared with the published run

The published run used `train_delta.py` at commit `bf53e36`, transformers 5.20.1, on one Gilbreth
A100-80GB (Slurm job 12026112). The model was uploaded 2026-10-26 20:33 EDT, and the switch to AUROC
selection (`5df3a62`) was committed 20:38 EDT, after the upload.

| Setting | Published run | This run |
|---|---|---|
| Input | `tok(original, reply, truncation="longest_first", max_length=4096)` | `tok(reply, truncation="longest_first", max_length=4096)` |
| Base model | `answerdotai/ModernBERT-base` | same (fresh classification head) |
| Data | `persuasion-datasets`, `cmv`, branch `t1-t2` | same, pinned to commit `e236e8d9` (the `t1-t2` head; unchanged since before the published run) |
| Labels | `0 = nodelta`, `1 = delta` (from the dataset) | same |
| Loss | class-weighted CE (weights 1.0 / 1.0, balanced) | same function (`train_delta.weighted_cross_entropy`) |
| Optimizer | Trainer defaults: fused AdamW, β = (0.9, 0.999), ε = 1e-8, weight decay 0, linear decay, no warmup, grad-norm clip 1.0 | same (same transformers version) |
| Learning rate / epochs | 2e-5 / 3 | same |
| Batch | 4 per device × 8 accumulation = 32 | same |
| Eval + save interval | every 500 steps | same |
| Checkpoint selection | validation F1 (delta), `load_best_model_at_end` | **F1 by default; to be approved** (see below) |
| Precision | fp32 | fp32 |
| Seed | Trainer seed 42 (data order); head initialisation **unseeded** | `set_seed(42)` before model creation, plus Trainer seed 42 |
| Hardware | 1× A100-80GB, CUDA | Apple M5 (10-core CPU), 32 GB unified memory, MPS |
| Attention | not recorded (sdpa unless flash-attn was installed) | sdpa |
| Gradient checkpointing | off | on only for batches padded beyond 1,024 tokens (memory only) |
| Kept checkpoints | 2 + best | all evaluated checkpoints (disk only) |

**Unavoidable differences:**
- **Head initialisation.** PyTorch draws a fresh random seed in every process, and the published run
  created its model before the Trainer seeded. Its classifier-head initialisation therefore can't be
  reproduced. This run seeds the head, so the run itself is reproducible.
- **Floating-point results** on MPS won't match the A100 bit for bit.
- **Gradient checkpointing** recomputes activations instead of storing them. It changes memory use and
  speed, not the computed gradients (beyond floating-point order).

### Memory on Apple MPS

Measured with one fp32 training step of ModernBERT-base on synthetic tokens. The recommended maximum
working set is 25.0 GiB.

| Batch × length | Gradient checkpointing | Peak MPS memory | Time per step |
|---|---|---|---|
| 1 × 4096 | off | 24.6 GiB (at the limit) | 4.5 s |
| 1 × 4096 | on | 6.0 GiB | 5.7 s |
| 4 × 4096 | on | 17.0 GiB | 22.6 s |
| 4 × 2048 | off | out of memory | – |
| 4 × 1024 | off / on | 11.6 / 3.3 GiB | 1.95 / 2.6 s |
| 4 × 512 | off / on | 6.3 / 3.7 GiB | 0.80 / 1.04 s |

The baseline's batch of 4 at 4,096 tokens therefore fits only with gradient checkpointing.
Checkpointing only the long batches avoids its ~23% time cost on the rest. The MPS high watermark is
set to 1.0, so running out of memory raises an error instead of swapping.

**Reply-only lengths** (ModernBERT tokens, train split): mean 428, median 288, 99th percentile 2,357.
35 of 16,676 replies are longer than 4,096 tokens. The reply-only inputs total 48% of the baseline's
post + reply tokens.

## Run

```bash
python -m unittest tests/t1_t2/test_train_cmv_reply_only.py -v
python scripts/t1_t2/train_cmv_reply_only.py train --smoke && python scripts/t1_t2/train_cmv_reply_only.py evaluate --smoke
caffeinate -i python scripts/t1_t2/train_cmv_reply_only.py train   # resumes from the newest checkpoint if interrupted
python scripts/t1_t2/train_cmv_reply_only.py evaluate
```

## Results

### Smoke test (pipeline check only, not a result)

- **Data:** 32 train pairs (64 rows), including the 4 pairs with the longest replies; 10 validation
  pairs and 10 test pairs. Batch 4 × accumulation 2, 4 optimizer steps, evaluation every 2 steps.
- **Runs:** smoke training ran twice; the second run was an unintended repeat with identical settings.
  - Each took about 1.3 minutes of training (80 s in the second run).
  - Gradient checkpointing was on for 4 of 8 batches each time.
  - Peak MPS driver memory was 17.4 GiB in the first run and 14.4 GiB in the second. The peak is
    sampled every 50 ms, so it varies; the 4 × 4,096 probe (17.0 GiB) is the worst case.
- **Behaviour:**
  - Validation selection by F1 picked step 2.
  - The selected model was saved, and `evaluate --smoke` scored the 10 smoke test pairs and ran the
    paired comparison.
  - The guard refused a second test evaluation.
- **Smoke metrics aren't results** (10 pairs after 4 steps, from a throwaway model), and they weren't
  used to choose any setting. The official test split hasn't been evaluated with a reply-only model.

### Full run

**Pending.** Full training hasn't started, and no full-run validation or test metrics exist yet.

## Notes

- **Duration estimate:** about 7–9 hours of training on the M5, plus about 35 minutes of evaluation.
  This comes from simulating random batches of 4 over the real train lengths and the measured step
  times above. Treat it as a rough guide.
- With checkpoints every 500 steps, an interruption loses at most 500 steps.
