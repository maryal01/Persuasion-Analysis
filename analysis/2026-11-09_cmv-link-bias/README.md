# T1: link sensitivity of the CMV ModernBERT model

**Status:** done (2026-11-09). Frozen published model, one seed, test split only. No retraining.

## Goal

Is the trained CMV classifier sensitive to URLs in a reply, and could links be a shortcut in its
persuasion predictions?

## What it does

`scripts/t1_t2/evaluate_cmv_links.py` has three stages:

- **`score`:** scores every test reply with the published model, checks that this reproduces the
  published 66.1% pairwise accuracy / 0.629 AUROC, and caches the per-reply scores.
- **`observe` (A, observational):** groups pairs by which reply contains a URL and reports pairwise
  accuracy next to the longer-reply baseline.
- **`intervene` (B, controlled edits):** re-scores each of the 399 test replies with a URL (of 2,038)
  in two edited forms, with the post and all other reply text unchanged:
  - **placeholder:** each URL replaced with `URL` (`[anchor](https://…)` becomes `[anchor](URL)`)
  - **remove:** each URL deleted (a Markdown link keeps its anchor text)

A URL is a Markdown link target or plain text starting with `http://`, `https://` or `www.`. Bare
domains (`example.com`, 10 test replies) and relative Reddit links (`/r/…`, 3 replies) aren't detected.
Every edit is verified to change nothing outside URL spans, and unedited replies must re-score to their
cached value (within 1e-3). Inputs match the published run: `tok(original, reply)`, 4,096 tokens, fp32.
A reply's score is `logit[delta] − logit[nodelta]`. All intervals are 95% bootstrap intervals that
resample whole posts (10,000 resamples, seed 577).

## Run

```bash
python scripts/t1_t2/evaluate_cmv_links.py score     --cache-dir outputs/cmv_length   # once
python scripts/t1_t2/evaluate_cmv_links.py observe   --out-dir outputs/cmv_links
python scripts/t1_t2/evaluate_cmv_links.py intervene --out-dir outputs/cmv_links      # ~3 min on M5 MPS
```

Outputs are git-ignored and hold ids and numbers only.

## Results

### A. Pairwise accuracy by which reply has a URL (observational)

| Group | Pairs | ModernBERT % | 95% CI | Longer reply wins % |
|---|---|---|---|---|
| all pairs | 1,019 | 66.1 | [63.1, 69.0] | 55.3 |
| neither has a URL | 685 | 65.4 | [61.7, 69.0] | 56.6 |
| both have URLs | 65 | 66.2 | [54.0, 77.8] | 46.2 |
| delta reply only | 172 | 83.1 | [77.3, 88.6] | 63.1 |
| no-delta reply only | 97 | 41.2 | [31.2, 51.5] | 38.1 |

- **Links and length are confounded.** The reply with the link is the longer one in about 62% of
  one-sided pairs, so the longer-reply baseline shows the same pattern (63.1% vs. 38.1%).
- Among length-matched pairs (word ratio ≥ 0.80) the pattern persists (93.8% vs. 30.0%), but on only
  32 and 10 pairs.
- In training data a link is mildly predictive of a delta: 20.7% of delta replies have one, against
  14.1% of no-delta replies (test: 23.3% vs. 15.9%).

### B. Controlled edits

Change in a reply's score:

| Edit | Replies | Mean tokens removed | Mean score change | 95% CI | Scores that fall |
|---|---|---|---|---|---|
| placeholder | 399 | 63.7 | −0.123 | [−0.138, −0.108] | 93.5% |
| remove | 399 | 71.0 | −0.160 | [−0.177, −0.142] | 96.2% |

- **The drop is the same for delta and no-delta replies** (difference −0.013 [−0.038, +0.013] for
  placeholder, −0.012 [−0.042, +0.018] for remove). URLs raise a reply's score whatever its label.
- Larger edits give larger drops: Spearman ρ between score change and tokens removed is about −0.3.

Pairwise accuracy before and after editing every URL reply:

| Group | Pairs | Before % | Placeholder % | Remove % |
|---|---|---|---|---|
| all pairs | 1,019 | 66.1 | 65.5 | 65.1 |
| both have URLs | 65 | 66.2 | 64.6 | 63.1 |
| delta reply only | 172 | 83.1 | 72.7 | 69.8 |
| no-delta reply only | 97 | 41.2 | 53.6 | 55.7 |

- **Overall accuracy barely moves:** −0.7 [−1.8, +0.5] points (placeholder) and −1.1 [−2.3, +0.2]
  (remove). The boost helps on "delta only" pairs and hurts on "no-delta only" pairs, and the two
  nearly cancel.
- **URLs explain part of the observational gap.** With URLs removed, the gap between the two one-sided
  groups shrinks from 41.9 to 14.1 points.
- Editing only one side moves accuracy on the affected pairs by 11 to 15 points.
- 37 (placeholder) and 43 (remove) pairs flip. They are pairs the model was unsure about: median
  score gap 0.07 to 0.09 before editing, against 0.27 for all pairs.

## Notes

- **URL content and length aren't separated.** Both edits shorten a reply by about 64–71 tokens, and
  the model mildly prefers longer replies. A length-matched control (deleting an equally long non-URL
  span) would separate them; it isn't run here.
- **These describe the model, not people.** Removing a link can change what a reply means, and nothing
  here says whether links persuade real readers.
- **Exploratory.** One model, one seed, one test split; many groups examined with no
  multiple-comparison correction; the smallest cells have 10–32 pairs.
- Five URL replies were already over 4,096 tokens and stay truncated after editing.
