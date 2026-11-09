# T1: length-controlled pairwise evaluation of the CMV ModernBERT model

**Status:** done (2026-11-09). Baseline reproduced; length-controlled analysis run from the cached scores. One model, one seed.

## Goal

Can ModernBERT identify the persuasive (delta) reply when the delta and no-delta replies of a pair
have similar lengths? Length alone carries signal ("longer reply wins" = 55.3% pairwise on the test
set, `RESULTS.md`), so the 66.1% headline could partly reflect length. This analysis re-scores the
existing model, without retraining, on pairs whose replies are close in length.

## What it does

`scripts/t1_t2/evaluate_cmv_length.py` has two stages:

1. **`predict`**: loads the published checkpoint and the `cmv` test split at pinned revisions,
   scores every reply exactly as `scripts/train_delta.py` did, and checks the published numbers
   (pairwise accuracy within ±0.3 points of 66.1%, AUROC within ±0.005 of 0.629). The headline
   metrics use the team's `pairwise_accuracy` and `binary_auroc` from `scripts/train_delta.py`.
2. **`analyze`**: runs only if the check passed. For all pairs, word-length ratio ≥ 0.80, ≥ 0.90,
   ≥ 0.95 and < 0.80, it reports pairs, unique posts, ModernBERT pairwise accuracy, the
   longer-reply baseline, and their difference, each with a 95% **post-cluster bootstrap** CI
   (10,000 resamples of posts). It also gives Wilson intervals (descriptive only), accuracy by
   ratio bin with a logistic trend, the same tables with token and character lengths, and error
   breakdowns.

| Setting | Value |
|---|---|
| Dataset | `nlp-final-project11/persuasion-datasets`, config `cmv`, split `test`, revision `e236e8d98785ad4c80a3af422c656061f373f825` (branch `t1-t2`) |
| Model | `nlp-final-project11/nlp-project-modernbert-cmv`, revision `407888b2c6395a52498173ab686b91fdce3f9b73` (weights from upload `e4c86c70`) |
| Pairing | dataset `pair_id`; each pair checked to have one delta row and one no-delta row from the same `post_id` |
| Input | `tok(original, reply, truncation="longest_first", max_length=4096)`, batch size 1, fp32 |
| Reply score | `logit[delta] − logit[nodelta]`, label ids read from the model config; ties count half |
| Length ratio | shorter reply / longer reply; primary unit = whitespace words (reproduces the published 55.3%) |
| Device | Apple MPS, with a CPU fallback; 40 random rows re-scored on CPU to compare |

## Run

```bash
# unit tests (synthetic, no downloads)
python -m unittest tests/t1_t2/test_evaluate_cmv_length.py -v

# 1) score the test split and check the published numbers (needs HF_TOKEN for the private repos)
python scripts/t1_t2/evaluate_cmv_length.py predict --out-dir outputs/cmv_length

# 2) length-controlled analysis from the cached scores
python scripts/t1_t2/evaluate_cmv_length.py analyze --out-dir outputs/cmv_length
```

Outputs (git-ignored, keep local): `predictions.csv` (ids, scores, numeric features; no text),
`reproduction.json`, `metrics.json`, `summary.md`, `confident_errors.csv`. Reply text is only
written with `analyze --export-error-text`, and only into a git-ignored directory.

## Results

### Baseline reproduction

Passed. Every published number matches the original Gilbreth run:

| Metric | Published (`RESULTS.md`) | Reproduced (MPS) |
|---|---|---|
| Pairwise accuracy | 66.1% | **66.14%** (674 / 1,019 pairs) |
| AUROC | 0.629 | **0.6286** |
| Row accuracy | 55.9% | 55.94% |
| Share of replies labelled delta | 74% | 74.2% |

- Test set: 1,019 pairs, 578 posts, 11 truncated inputs.
- Inference took 226 s on Apple MPS (fp32, batch size 1).
- The same 40 random rows scored on CPU differed from MPS by at most 1.4 × 10⁻⁶ in score.
- Environment: torch 2.14.1, transformers 5.20.1.

### Length-controlled evaluation (primary: whitespace words)

- **Accuracy** is pairwise accuracy in %. "Longer" is the longer-reply-wins baseline in the same subset.
- **Diff** is ModernBERT minus longer, in percentage points.
- **Intervals** are 95% post-cluster bootstrap percentile intervals (10,000 resamples of posts, seed 577). Wilson intervals assume independent pairs and are descriptive only.
- The ≥ thresholds are nested subsets, not separate groups.

| Subset | Pairs | Posts | ModernBERT | 95% CI | Wilson (descr.) | Longer | 95% CI | Diff | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| all | 1019 | 578 | 66.1 | [63.1, 69.0] | [63.2, 69.0] | 55.3 | [51.8, 58.7] | +10.9 | [+6.6, +15.1] |
| ratio ≥ 0.80 | 208 | 178 | 67.8 | [61.1, 74.2] | [61.2, 73.8] | 51.0 | [44.3, 57.7] | +16.8 | [+7.2, +26.5] |
| ratio ≥ 0.90 | 94 | 90 | 71.3 | [62.0, 80.2] | [61.4, 79.4] | 47.9 | [38.0, 57.7] | +23.4 | [+10.4, +36.5] |
| ratio ≥ 0.95 | 51 | 49 | 72.5 | [60.8, 84.0] | [59.1, 82.9] | 56.9 | [44.0, 69.6] | +15.7 | [−1.0, +33.0] |
| ratio < 0.80 | 811 | 501 | 65.7 | [62.4, 69.0] | [62.4, 68.9] | 56.4 | [52.5, 60.2] | +9.4 | [+4.8, +13.9] |

### Robustness: other length units

ModernBERT accuracy is the same in every unit on "all" (66.1 [63.1, 69.0]); only the baseline changes there.

| Subset | Tokens: pairs | ModernBERT (95% CI) | Longer | Diff (95% CI) | Chars: pairs | ModernBERT (95% CI) | Longer | Diff (95% CI) |
|---|---|---|---|---|---|---|---|---|
| all | 1019 | 66.1 [63.1, 69.0] | 56.4 | +9.7 [+5.5, +13.8] | 1019 | 66.1 [63.1, 69.0] | 56.3 | +9.8 [+5.6, +13.9] |
| ratio ≥ 0.80 | 202 | 65.8 [58.6, 72.6] | 53.0 | +12.9 [+2.5, +22.9] | 204 | 65.7 [58.8, 72.4] | 53.4 | +12.3 [+1.6, +22.6] |
| ratio ≥ 0.90 | 88 | 63.6 [53.5, 73.6] | 53.4 | +10.2 [−3.5, +23.7] | 86 | 70.9 [61.2, 80.2] | 46.5 | +24.4 [+9.4, +39.5] |
| ratio ≥ 0.95 | 46 | 67.4 [53.3, 80.0] | 54.3 | +13.0 [−6.5, +31.1] | 45 | 60.0 [46.5, 73.3] | 48.9 | +11.1 [−11.1, +32.6] |
| ratio < 0.80 | 817 | 66.2 [62.9, 69.5] | 57.3 | +8.9 [+4.6, +13.2] | 815 | 66.3 [62.9, 69.6] | 57.1 | +9.2 [+4.8, +13.5] |

### Accuracy by word-length ratio (exploratory, all pairs)

| Ratio bin | Pairs | Posts | ModernBERT (95% CI) | Longer | Diff (95% CI) |
|---|---|---|---|---|---|
| [0.0, 0.2) | 110 | 96 | 70.0 [61.3, 78.4] | 60.0 | +10.0 [+1.0, +18.6] |
| [0.2, 0.4) | 223 | 184 | 63.7 [57.7, 69.6] | 57.4 | +6.3 [−0.9, +13.6] |
| [0.4, 0.6) | 257 | 209 | 65.0 [59.2, 70.7] | 54.1 | +10.9 [+2.7, +19.0] |
| [0.6, 0.8) | 221 | 189 | 66.5 [60.5, 72.3] | 56.1 | +10.4 [+1.8, +18.9] |
| [0.8, 0.9) | 114 | 98 | 64.9 [55.6, 74.0] | 53.5 | +11.4 [−2.0, +24.8] |
| [0.9, 1.0] | 94 | 90 | 71.3 [62.0, 80.2] | 47.9 | +23.4 [+10.4, +36.5] |

A logistic regression of pair correctness on word ratio gives a slope of +0.09 per unit of ratio, 95% post-cluster CI [−0.42, +0.62] (2,000 resamples). There is no detectable trend.

### Error analysis (word lengths, all pairs)

| Breakdown | Group: pairs, ModernBERT % [95% cluster CI] |
|---|---|
| Which reply is longer | delta longer: 561, 73.1 [69.3, 76.8] · no-delta longer: 454, 57.7 [52.9, 62.3] · same: 4, 50.0 |
| Links in replies | none: 685, 65.4 [61.7, 69.0] · delta only: 172, **83.1** [77.3, 88.6] · no-delta only: 97, **41.2** [31.2, 51.5] · both: 65, 66.2 [54.0, 77.8] |
| Quoted lines (`>`) | none: 523, 67.7 [63.7, 71.6] · a reply quotes: 496, 64.5 [60.0, 68.9] |
| Original post length (token terciles) | < 289: 340, 60.9 [55.5, 66.2] · 289–502: 338, 68.3 [62.9, 73.5] · ≥ 502: 341, 69.2 [64.5, 73.8] |
| Pairs per post | 1 pair: 332, 66.3 [61.1, 71.4] · 2+ pairs: 687, 66.1 [62.4, 69.6] |
| Truncated input (> 4,096 tokens) | none: 1010, 65.9 [63.0, 68.9] · a reply truncated: 9, 88.9 (too few to interpret) |
| \|Score gap\| quartile | Q1 (least confident): 56.5 · Q2: 58.3 · Q3: 69.0 · Q4: **80.8** [75.9, 85.4] |

- **Length preference:** the model picks the longer reply in 59.3% of pairs, while the delta reply is the longer one in 55.3%.
  - In the length-matched subsets it picks the longer reply about half the time: 50.5% (≥ 0.80), 51.1% (≥ 0.90), 56.9% (≥ 0.95).
  - Its score gap (delta − no-delta) correlates with the log word ratio (delta / no-delta) at Spearman ρ = +0.27.
- **Confidence tracks correctness.** The median |score gap| is 0.33 for correct pairs and 0.20 for wrong pairs.
- **The 25 most confident errors** have score gaps between −0.68 and −1.61 and come from 25 different posts.
  - In 13 of the 25 the no-delta reply is longer.
  - 4 of the 25 have word ratio ≥ 0.80.
  - 6 of the 25 have a link only in the no-delta reply, compared with 9.5% of all pairs.
  - None are truncated.
  - Their IDs and numbers are in `outputs/cmv_length/confident_errors.csv`. Their text is only in the local, git-ignored `outputs/cmv_length/confident_errors_text.jsonl`.

### Interpretation

- **ModernBERT's advantage is still there when the two replies have similar lengths.**
  - With word ratio ≥ 0.80 (208 pairs), the longer-reply baseline is at chance (51.0%) while ModernBERT scores 67.8% [61.1, 74.2]. That is 16.8 points above the baseline, with a CI of [+7.2, +26.5] that excludes 0.
  - The same holds with token counts (+12.9 [+2.5, +22.9]) and character counts (+12.3 [+1.6, +22.6]).
  - Accuracy on length-matched pairs isn't lower than on all pairs. The ratio bins and the logistic slope show no detectable change with length similarity.
- **The stricter thresholds are underpowered.** With ratio ≥ 0.90 / 0.95 (45–94 pairs), point estimates range from 60% to 72.5%, depending on the length unit.
  - The ModernBERT − baseline CI excludes 0 for words ≥ 0.90 and chars ≥ 0.90.
  - It includes 0 for tokens ≥ 0.90 and for all three units at ≥ 0.95.
  - These subsets can't show whether accuracy differs from the full-set 66%. They are compatible with roughly 55–80%.
- **The model's predictions are associated with length overall.** It favours the longer reply a little more often than the labels do, and it is more accurate when the delta reply is the longer one (73.1% vs. 57.7%). Both are associations; this analysis doesn't show that length causes the model's decisions.
- **Links show the strongest association in the breakdowns.** Accuracy is 83.1% when only the delta reply has a link and 41.2% when only the no-delta reply does. That is consistent with the model using link presence as a cue (the "more links wins" baseline is 54.3%). It is still observational and worth controlling for next.

## Limitations

- **One trained model and one seed,** evaluated on one test split of 1,019 pairs.
- **Selection, not intervention.** Length-matched pairs are a subset that may differ from other pairs in ways besides length (topic, reply style, links). The comparisons are associations and don't isolate an effect of length.
- **Small, nested subsets and many comparisons.**
  - The ≥ 0.90 / ≥ 0.95 subsets have 45–94 pairs.
  - The thresholds are nested, so their results aren't independent.
  - Several units, bins and breakdowns were examined with no multiple-comparison correction, so treat the breakdowns as exploratory.
  - Percentile bootstrap intervals are approximate for small subsets with many single-pair posts.
- **The length baseline's definition is inferred.** Whitespace words with ties counted as half reproduce the published 55.3% exactly, but the original code isn't committed.
- **Text features are heuristics.** Links are matched with `https?://`, `www.` or `](`. Quotes are lines starting with `>` or `&gt;`. The number of comments in a reply branch can't be recovered, because comments are joined with blank lines.
- **Truncation:** only 9 pairs involve a truncated input, so that breakdown says nothing.

## Notes

- **Pairs from the same post are correlated** (1,019 pairs from 578 posts). Primary intervals
  resample posts. Wilson intervals assume independent pairs and are descriptive only. No
  independence-based significance tests are reported.
- **Small subsets.** About 208 / 94 / 51 pairs reach word ratio ≥ 0.80 / 0.90 / 0.95, so those
  intervals are wide (roughly ±6 / ±10 / ±13 points before clustering).
- **Unverified assumption.** The committed code doesn't show how the 55.3% longer-reply baseline
  was computed. Whitespace word counts with ties counted as half reproduce it exactly, so that
  definition is used here.
- One trained model and one seed (see the `RESULTS.md` caveats). The checkpoint was selected by
  validation F1.
- 11 of 2,038 inputs are longer than 4,096 tokens and are truncated, as in training.
