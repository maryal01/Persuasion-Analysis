# T1: link-bias analysis of the CMV ModernBERT model

**Status:** done (2026-11-09). Observational analysis (A) and controlled URL edits (B) on the frozen
published model, one seed, test split only. No retraining.

## Goal

Is the trained CMV classifier sensitive to URL presence or URL content, and could links be a
superficial shortcut in its persuasion predictions? The length-controlled analysis
(`analysis/2026-11-09_cmv-length-control/`) found 83.1% pairwise accuracy when only the delta reply
has a link and 41.2% when only the no-delta reply has one. This analysis looks at that gap
observationally (A) and tests it with controlled edits (B).

## What it does

`scripts/t1_t2/evaluate_cmv_links.py`:

- **`observe` (Experiment A).**
  - Reuses the verified pairs and cached scores from `outputs/cmv_length/` (read only). It first checks
    that every cached row matches the dataset's pair id, label, character count and word count.
  - Groups pairs by which reply contains a URL and reports pairwise accuracy and the longer-reply
    baseline. All CIs are 95% post-cluster bootstrap intervals (10,000 resamples, seed 577).
  - Also reports the same groups split by word-length matching, observable characteristics per group,
    URL rates by label in each split, and reply scores with vs. without a URL.
- **`intervene` (Experiment B).**
  - For every test reply with a URL (399 of 2,038), it builds two edits:
    - **placeholder:** each URL span replaced with `URL`
    - **remove:** each URL span deleted
  - The original post and all other reply text stay unchanged.
  - Before scoring, each edit is verified to differ from the original only inside URL spans and to
    contain no remaining URL.
  - Unedited URL replies are re-scored first and must match the cached scores (max difference 1e-3).
  - Replies without URLs reuse their cached scores.
  - Inference settings are identical to the published baseline: `tok(original, reply,
    truncation="longest_first", max_length=4096)`, fp32, batch size 1, eval mode, Apple MPS. A reply's
    score is `logit[delta] − logit[nodelta]`, with label ids read from the model config.
  - Edit scenarios:
    - **all URL replies edited**
    - **delta reply only**: only the delta reply's URLs are edited
    - **no-delta reply only**: only the no-delta reply's URLs are edited
  - CIs for score and accuracy changes are paired: each replicate resamples posts and recomputes the
    within-pair change.

### URL detection

| Form | Example shape | Detected as | placeholder | remove |
|---|---|---|---|---|
| Markdown link with a URL target | `[anchor](https://…)`, optional `"title"` | one span | `[anchor](URL)` | `anchor` |
| Plain URL | `https://…`, `http://…`, `www.…` | one span | `URL` | deleted |

- Matching is case-insensitive.
- Balanced parentheses are allowed inside a URL, e.g. `…/wiki/Mercury_(planet)`.
- Trailing `.,;:!?'"*` is not part of a URL.
- `&amp;` inside a URL is kept.
- URLs inside a Markdown anchor are edited the same way as plain URLs.
- **Not detected:**
  - bare domains without a scheme or `www`, e.g. `example.com` (10 test replies mention one)
  - relative Reddit links such as `/r/…` and `](/u/…)` (3 replies)
  - URLs split by whitespace
- Detection flags the same 399 replies as the rough regex used in the length analysis, so the four
  link groups have the same sizes as there.
- **Edit verification** uses two checks; each one alone is sound. An edit passes if either passes:
  - a character diff whose changes must all lie inside URL spans
  - an alignment-free check: the text between spans must appear verbatim and in order, and each edited
    stretch may only reuse characters of its original span
- **Result:** 791 of the 798 edits pass both checks, and 7 pass exactly one. The first run stopped on
  one of those 7, a case where the diff alignment reached one character past a URL span; the text
  outside the span was in fact unchanged. That is why the second check was added.

## Run

```bash
python -m unittest tests/t1_t2/test_evaluate_cmv_links.py -v     # 20 synthetic tests
python scripts/t1_t2/evaluate_cmv_links.py observe   --out-dir outputs/cmv_links
python scripts/t1_t2/evaluate_cmv_links.py intervene --out-dir outputs/cmv_links  # ~3 min on M5 MPS
```

Prerequisite: `outputs/cmv_length/` from `scripts/t1_t2/evaluate_cmv_length.py predict` (a passed
reproduction). Outputs go to the git-ignored `outputs/cmv_links/` and contain ids and numbers only:
`pairs.csv`, `url_replies.csv`, `flips.csv`, `*_metrics.json`, `*_summary.md`.
`intervene --export-flip-text` also writes the flipped pairs' text to `flips_text.jsonl`, and refuses
unless the target is git-ignored.

## Results

### A1. Pairwise accuracy by link group (observational)

| Group | Pairs | Posts | ModernBERT % | 95% CI | Longer reply % | 95% CI | Diff (pts) | 95% CI |
|---|---|---|---|---|---|---|---|---|
| all pairs | 1019 | 578 | 66.1 | [63.1, 69.0] | 55.3 | [51.8, 58.7] | +10.9 | [+6.6, +15.1] |
| neither has a URL | 685 | 440 | 65.4 | [61.7, 69.0] | 56.6 | [52.6, 60.6] | +8.8 | [+3.9, +13.5] |
| both have URLs | 65 | 58 | 66.2 | [54.0, 77.8] | 46.2 | [33.9, 58.5] | +20.0 | [+1.4, +39.0] |
| delta only | 172 | 143 | 83.1 | [77.3, 88.6] | 63.1 | [55.1, 71.4] | +20.1 | [+10.1, +29.7] |
| no-delta only | 97 | 85 | 41.2 | [31.2, 51.5] | 38.1 | [28.1, 48.9] | +3.1 | [−10.5, +16.5] |

### A2. Link group × word-length matching (ratio ≥ 0.80)

| Group | Pairs | Posts | ModernBERT % | 95% CI | Longer reply % |
|---|---|---|---|---|---|
| neither, ratio ≥ 0.80 | 150 | 128 | 65.3 | [57.4, 73.0] | 51.7 |
| neither, ratio < 0.80 | 535 | 380 | 65.4 | [61.2, 69.5] | 57.9 |
| both, ratio ≥ 0.80 | 16 | 15 | 62.5 | [36.8, 86.7] | 43.8 |
| both, ratio < 0.80 | 49 | 47 | 67.3 | [54.9, 79.6] | 46.9 |
| delta only, ratio ≥ 0.80 | 32 | 30 | 93.8 | [83.9, 100.0] | 45.3 |
| delta only, ratio < 0.80 | 140 | 119 | 80.7 | [73.8, 87.0] | 67.1 |
| no-delta only, ratio ≥ 0.80 | 10 | 10 | 30.0 | [0.0, 60.0] | 70.0 |
| no-delta only, ratio < 0.80 | 87 | 77 | 42.5 | [32.0, 53.4] | 34.5 |

- Pairs without links: ModernBERT gets about 65% whether or not the lengths are matched (150
  length-matched pairs: 65.3% [57.4, 73.0]).
- The one-sided-link pattern persists among length-matched pairs, but those cells have only 10–32 pairs.

### A3. Possible confounders by link group

| Group | Median words (delta / no-delta) | Delta reply longer % | Word ratio ≥ 0.80 % | Pairs with a quote % | Median post tokens | Mean score (delta / no-delta) |
|---|---|---|---|---|---|---|
| neither | 215 / 192 | 56.4 | 21.9 | 44.8 | 367 | +0.22 / +0.07 |
| both | 292 / 276 | 46.2 | 24.6 | 58.5 | 380 | +0.61 / +0.45 |
| delta only | 316 / 233 | 62.8 | 18.6 | 53.5 | 385 | +0.54 / +0.15 |
| no-delta only | 276 / 372 | 38.1 | 10.3 | 60.8 | 363 | +0.29 / +0.31 |

- **Link presence and length are confounded.** The reply with the link also tends to be the longer
  one: in "delta only" pairs the delta reply is longer in 62.8%; in "no-delta only" pairs the no-delta
  reply is longer in 61.9%. So the longer-reply baseline also does well on "delta only" pairs (63.1%)
  and badly on "no-delta only" pairs (38.1%).
- Replies that contain URLs also contain quotes more often.

### A4–A5. URL rate by label, and scores of replies with URLs

| Split | Replies | Delta replies with URL | No-delta replies with URL |
|---|---|---|---|
| train | 16,676 | 20.7% | 14.1% |
| validation | 1,850 | 23.2% | 14.3% |
| test | 2,038 | 23.3% | 15.9% |

- In training data, a link is mildly predictive of the delta label, so the model could learn it.
- In the test set, replies with a URL score higher than replies without one, for both labels:
  - delta replies: +0.325 [+0.272, +0.375] (237 vs. 782 replies)
  - no-delta replies: +0.281 [+0.201, +0.361] (162 vs. 857 replies)

### B1. Controlled edits: change in reply score

| Variant | Label | Replies | Mean tokens removed | Mean score change | 95% CI | Predicted class flips |
|---|---|---|---|---|---|---|
| placeholder | all | 399 | 63.7 | −0.123 | [−0.138, −0.108] | 24 (6.0%) |
| placeholder | delta | 237 | 61.2 | −0.128 | [−0.145, −0.111] | 12 (5.1%) |
| placeholder | no-delta | 162 | 67.4 | −0.115 | [−0.139, −0.092] | 12 (7.4%) |
| remove | all | 399 | 71.0 | −0.160 | [−0.177, −0.142] | 32 (8.0%) |
| remove | delta | 237 | 68.4 | −0.165 | [−0.184, −0.146] | 18 (7.6%) |
| remove | no-delta | 162 | 74.8 | −0.152 | [−0.180, −0.126] | 14 (8.6%) |

- The score falls for 93.5% (placeholder) and 96.2% (remove) of URL replies.
- **The drop is the same for delta and no-delta replies.** Delta minus no-delta difference in mean score
  change: placeholder −0.013 [−0.038, +0.013]; remove −0.012 [−0.042, +0.018]. So URLs raise a reply's
  score whatever its label, which is what a label-independent shortcut would look like.
- Larger edits produce larger drops: Spearman ρ between score change and tokens removed is −0.30
  (placeholder) and −0.28 (remove).

### B2. Controlled edits: pairwise accuracy and preference flips

Paired post-cluster bootstrap. "Affected" means the pair has at least one edited reply.

| Variant | Scenario | Pairs | Acc. before % | After % | Change (pts) | 95% CI | Flips | Correct→wrong | Wrong→correct |
|---|---|---|---|---|---|---|---|---|---|
| placeholder | all URL replies (all pairs) | 1019 | 66.1 | 65.5 | −0.7 | [−1.8, +0.5] | 37 | 22 | 15 |
| placeholder | all URL replies (affected) | 334 | 67.7 | 65.6 | −2.1 | [−5.6, +1.5] | 37 | 22 | 15 |
| placeholder | delta reply only (affected) | 237 | 78.5 | 67.1 | −11.4 | [−15.5, −7.7] | 27 | 27 | 0 |
| placeholder | no-delta reply only (affected) | 162 | 51.2 | 63.6 | +12.3 | [+7.2, +17.7] | 20 | 0 | 20 |
| remove | all URL replies (all pairs) | 1019 | 66.1 | 65.1 | −1.1 | [−2.3, +0.2] | 43 | 27 | 16 |
| remove | all URL replies (affected) | 334 | 67.7 | 64.4 | −3.3 | [−7.3, +0.6] | 43 | 27 | 16 |
| remove | delta reply only (affected) | 237 | 78.5 | 64.1 | −14.3 | [−19.0, −10.0] | 34 | 34 | 0 |
| remove | no-delta reply only (affected) | 162 | 51.2 | 66.0 | +14.8 | [+9.3, +20.6] | 24 | 0 | 24 |

**All URL replies edited, by link group:**

| Variant | Group | Pairs | Before % | After % | Change 95% CI (pts) | Flips (C→W / W→C) |
|---|---|---|---|---|---|---|
| placeholder | both | 65 | 66.2 | 64.6 | [−9.5, +6.3] | 7 (4 / 3) |
| placeholder | delta only | 172 | 83.1 | 72.7 | [−15.2, −6.2] | 18 (18 / 0) |
| placeholder | no-delta only | 97 | 41.2 | 53.6 | [+6.5, +18.8] | 12 (0 / 12) |
| remove | both | 65 | 66.2 | 63.1 | [−10.6, +4.5] | 6 (4 / 2) |
| remove | delta only | 172 | 83.1 | 69.8 | [−18.5, −8.5] | 23 (23 / 0) |
| remove | no-delta only | 97 | 41.2 | 55.7 | [+8.2, +21.3] | 14 (0 / 14) |

- **Flips cluster where the model was unsure.** Flipped pairs had a median |score gap| before editing
  of 0.07 (placeholder) and 0.09 (remove), against 0.27 for all pairs; the largest was 0.37. They come
  from 37 and 42 distinct posts.
- Flip ids and numbers are in `outputs/cmv_links/flips.csv`; their text is only in the local,
  git-ignored `flips_text.jsonl`.

## Interpretation

- **The model is sensitive to URL spans.**
  - Replacing or deleting URLs lowers the delta score of nearly every URL reply, by 0.12–0.16 logits.
  - The drop is the same for delta and no-delta replies, so it doesn't depend on whether the reply
    persuaded anyone.
  - Editing only one side moves accuracy on the affected pairs by 11–15 points.
- **URL spans explain part of the observational gap.** With URLs removed, the "delta only" vs.
  "no-delta only" accuracy gap shrinks from 41.9 points (83.1% vs. 41.2%) to 14.1 points (69.8% vs.
  55.7%; placeholder: 72.7% vs. 53.6%).
- **On the full test set, editing every URL barely changes accuracy:** −0.7 [−1.8, +0.5] (placeholder)
  and −1.1 [−2.3, +0.2] (remove). The boost helps on "delta only" pairs and hurts on "no-delta only"
  pairs, and the two nearly cancel because links are only mildly more common on delta replies. The
  headline 66.1% therefore doesn't seem to depend on links, even though individual predictions do.
- **The edits don't separate URL content from length.**
  - Both edits shorten the reply by about 64–71 tokens; the placeholder is one token but a URL is many.
  - The model also has a mild length preference, and score drops grow with tokens removed.
  - So this experiment shows sensitivity to the URL spans as tokens. It doesn't show whether the model
    responds to "a link is present", to what the URL contains, or to the extra length.
  - A length-matched control would separate them: delete an equally long non-URL span from the same
    reply, or use a placeholder that keeps the token count. It isn't run here.
- These are statements about the model's behaviour. Editing links can change what a reply means (a
  cited source is no longer visible), and nothing here says whether links persuade real people.

## Limitations

- **Model and data:** one model, one seed, one test split. Experiment A's subgroups are observational
  and confounded with length and quoting. Several A2 cells have 10–32 pairs.
- **Detection:** heuristic, as described in "URL detection" above.
- **Edits:** both remove most of a URL's tokens. Edits only shorten replies, so they can't cause new
  truncation. Five URL replies were already over 4,096 tokens and stay truncated after editing. For
  those five, an edit can also shift which text falls inside the 4,096-token window.
- **Statistics:**
  - Many groups and scenarios were examined with no multiple-comparison correction, so treat the
    breakdowns as exploratory.
  - Bootstrap CIs resample posts; Wilson / binomial intervals assuming independent pairs aren't used.
  - Percentile intervals are approximate for small groups.
