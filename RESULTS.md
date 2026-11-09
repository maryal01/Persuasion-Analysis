# Results

Test-set results for each trained model. Each row is one training run; add new rows as runs finish.

## T1: CMV persuasion (`cmv`)

Test set: 1,019 delta/no-delta pairs (2,038 replies) from posts that never appear in training
(dataset branch `t1-t2`). The main metric is **pairwise accuracy**: how often the model scores the
delta reply above its no-delta partner. Chance is 50%.

| Model | Pairwise acc. | AUROC | Accuracy | F1 (delta) | Macro-F1 |
|---|---|---|---|---|---|
| Coin flip | 50.0% | 0.500 | – | – | – |
| More comments wins | 53.7% | – | – | – | – |
| More links wins | 54.3% | – | – | – | – |
| Longer reply wins | 55.3% | – | – | – | – |
| **ModernBERT-base, text only** | **66.1%** | **0.629** | 55.9% | 0.645 | 0.532 |

- Model: [`nlp-final-project11/nlp-project-modernbert-cmv`](https://huggingface.co/nlp-final-project11/nlp-project-modernbert-cmv)
  (private). Trained on Gilbreth (one A100, ~1h45m, Slurm job 12026112), 3 epochs, max 4,096 tokens.
- With 1,019 pairs, pairwise accuracy has a rough 95% interval of ±3 points (about 63–69%).
- Single-reply accuracy is low because the model labels 74% of replies "delta" (precision 0.54,
  recall 0.80). Its ranking is better than its yes/no cutoff, which is why the pairwise score is higher.
- No delta markers ("Δ", "!delta") leak into the replies: they appear in under 0.5% of replies on either side.

**Caveats**
- One run, one random seed. Final numbers should average a few seeds.
- This run picked its checkpoint by validation F1 (step 1000, validation AUROC 0.630). Later
  checkpoints had higher validation AUROC (0.644). The code now selects by AUROC.
- Text only. Audience features (RQ2) haven't been added yet.

**Prior work.** Al-Khatib et al. (ACL 2020), who built Webis-CMV-20, report per-comment accuracy
rather than pairwise: word count 51.9%, Tan et al.'s best text features 57.8%, their model with
debater traits 61.1%. Their metric differs from ours, so don't compare the numbers directly.
Tan et al. (2016)'s 65.1% is on a different corpus.

**Length-controlled evaluation** (same model, no retraining; reproduced 66.1% / AUROC 0.629 on Apple MPS).
Length ratio = shorter / longer reply, in whitespace words. 95% intervals come from a bootstrap
that resamples whole posts.

| Pairs | n pairs (posts) | ModernBERT | Longer reply wins | Difference (95% CI) |
|---|---|---|---|---|
| All | 1,019 (578) | 66.1% | 55.3% | +10.9 pts [+6.6, +15.1] |
| Word ratio ≥ 0.80 | 208 (178) | 67.8% | 51.0% | +16.8 pts [+7.2, +26.5] |

- Token- and character-based length ratios support this: at ratio ≥ 0.80 the difference is
  +12.9 [+2.5, +22.9] (tokens) and +12.3 [+1.6, +22.6] (characters).
- Observational: one model, one seed. The stricter subsets (ratio ≥ 0.90 / 0.95, 45–94 pairs) are
  small, and some of their intervals include 0.
- Details and error analysis: `analysis/2026-11-09_cmv-length-control/README.md`. Script:
  `scripts/t1_t2/evaluate_cmv_length.py`.

## T2: DDO voter switching (`ddo_switch`, `ddo_undecided`)

Not trained yet.
