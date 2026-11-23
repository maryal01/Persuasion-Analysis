# T5: can the Hub configs with the OP's history be used, and does the history help?

**Status:** running (23 November 2026: time check, logistic regression and ModernBERT seed 0 done; more seeds and a date control open).

## Goal
Manish uploaded configs with the OP's earlier CMV writing (`cmv_thread_history`, `cmv_thread_own_history` and its ablations `_base`, `_replies`; Hub `main`, commit 77c2ff53). Their build code is not in the repository. Checked: is the history written before the post, is the label ours, and how much do post, history and posting date predict.

## What it does
| script | question |
|---|---|
| `history_time_check.py` | finds every history item of `original` in Webis-CMV-20 `threads.jsonl` by its text and compares its creation / edit time with the post |
| `paired_test.py` | rescores the best checkpoints of the ModernBERT runs on test and compares each with a reference run, paired bootstrap over OPs |
| `scripts/t5/history_baselines.py` | logistic regression (validation grid L1/L2 x C, bootstrap over OPs) on date, number of history items, TF-IDF of post, history, both; only `original` is used, `reply` (the discussion) is not |

## Run
```bash
python analysis/2026-11-23_t5_thread_history/history_time_check.py <cmv_dir>/threads.jsonl.bz2 --output <out_dir>/history_time.json   # one core, about 15 minutes
cd scripts/t5 && python history_baselines.py --config cmv_thread_own_history --output-dir <out_dir>/history_baselines              # and --config cmv_thread_history
```

## Results

### The configs
| config | train / val / test | `original` | `reply` | rows without history |
|---|---|---|---|---|
| `cmv_thread_history` | 21,048 / 2,608 / 2,588 | `[comment]` (other threads), `[reply in own thread]`, `[post]`, then `[SEP]` + post | discussion with `OP:` turns | 38% (val) |
| `cmv_thread_own_history` | 8,446 / 999 / 923 | own earlier posts and their threads (`OP:` and `User:` comments), then `[SEP]` + post | discussion without `OP:` turns | 0% |
| `..._base` / `..._replies` | same posts | the post only | without / with `OP:` turns | - |
| `cmv_thread_matched` | same posts as `cmv_thread_history` | the post only | identical to `cmv_thread_history` | - |

`cmv_thread_history` is `cmv_thread_matched` with the history put in front: same posts, labels and `reply` in every split, and its `original` ends with the matched `original` (checked on all 26,244 rows). `cmv_thread_matched` also has `truncated`, so the flag probably refers to `reply`, not to the history (not verified).

- Same delta rate in both, about 0.5 (balanced sample), no OP in two splits.
- Label: identical to our DeltaBot label on every post that is also in `cmv_malleability_liwc` (7,216 and 3,904 posts).
- ModernBERT tokens (val): `cmv_thread_history` total median 821, 37% above 2,048, 20% above 8,192; `cmv_thread_own_history` median 5,681, 76% above 2,048, 40% above 8,192. Posts alone exceed 2,048 tokens in 0.3-0.4%.
- `truncated` is true for 48% of rows; what it means is not documented.

### The history is written before the post
| config | items | created before the post | created after | created before, edited after | not found |
|---|---|---|---|---|---|
| `cmv_thread_history` (all splits) | 608,000 | 99.2-99.7% per kind | 20 items (in 20 train rows) | 0 | 0.3-1.8% |
| `cmv_thread_own_history` (all splits) | 221,000 | 93.5-100% | 0 | 49 items (in 35 rows) | 0% posts, 6.4% replies |

The items are not always in time order (78% of rows of `cmv_thread_history`, 18% of `cmv_thread_own_history` are; the latter is grouped by thread). What cutting the history from the front relies on, the stored end being the newest (rows with at least two matched items, all splits):

| config | last item is the newest | last 5 items are the 5 newest |
|---|---|---|
| `cmv_thread_history` (15,023 rows) | 98.6% | 96.0% |
| `cmv_thread_own_history` (9,511 rows) | 83.8% | 73.3% |

So for `cmv_thread_history` "the last stored items" is a good stand-in for "the most recent"; for `cmv_thread_own_history` it is not, and the items would have to be sorted by time first.

### Logistic regression (test AUROC, 95% interval over OPs)
| input | `cmv_thread_history` (test 2,588) | `cmv_thread_own_history` (test 923) |
|---|---|---|
| posting date | 0.642 [0.617, 0.665] | 0.645 [0.600, 0.684] |
| number of history items | 0.505 | 0.572 |
| post (TF-IDF) | 0.593 [0.570, 0.617] | 0.617 [0.572, 0.658] |
| history (TF-IDF) | 0.553 [0.527, 0.577] | 0.636 [0.593, 0.678] |
| post + history | 0.598 [0.572, 0.623] | 0.644 [0.603, 0.682] |
| post + history + date | 0.665 [0.640, 0.689] | 0.671 [0.629, 0.709] |

- The date alone is as good as post + history in both configs; with balanced classes the delta rate still rises over time.
- History words add +0.005 (`cmv_thread_history`) and +0.027 (`cmv_thread_own_history`) over the post; both inside the intervals.

## Notes
- "Before" means at least one text match was created before the post; a text that occurs twice is counted by its earliest copy. 
- The stored history text is the crawled (possibly edited) version; only items whose edit time is known were checked for edits.
- Outputs (JSON, logs, checkpoints) stay out of git, in the `<out_dir>` of the commands.

### ModernBERT on `cmv_thread_history` (seed 0, only `original`)
`scripts/t5/train.py`, 3 epochs, lr 2e-5, head-lr 1e-3, effective batch 32, bf16, best validation epoch scored on test. "Grouped" = `--group-by-length` (batches of similar length; same results within seed noise, 2x faster). Paired difference: `paired_test.py`, AUROC minus that of A grouped on the same 2,588 test posts, 95% interval and share of OP-bootstrap resamples (2,000) with no gain. The rescored AUROCs equal those of the runs to 0.0001.

| run | input | val AUROC | test AUROC [95% CI] | vs A grouped [95% CI] | p (no gain) |
|---|---|---|---|---|---|
| A random | post (2,048) | 0.645 | 0.638 [0.614, 0.661] | +0.005 [-0.011, +0.021] | 0.28 |
| A grouped | post (2,048) | 0.646 | 0.633 [0.608, 0.655] | reference | - |
| B random | post + history as one text, cut at 2,048 | 0.656 | 0.646 [0.620, 0.669] | +0.013 [-0.005, +0.033] | 0.09 |
| B grouped | same | 0.668 | **0.657** [0.630, 0.680] | **+0.024 [+0.005, +0.043]** | 0.006 |
| E grouped | same, cut at 8,192 | 0.669 | 0.651 [0.626, 0.674] | +0.018 [-0.001, +0.036] | 0.03 |
| C mean | post + newest 32 items encoded apart (256 tokens each), mean of the item vectors | 0.652 | 0.636 [0.610, 0.659] | +0.003 [-0.012, +0.018] | 0.34 |
| C attention | same, attention pooling over the items | 0.649 | 0.637 [0.613, 0.659] | +0.004 [-0.013, +0.021] | 0.30 |

For comparison, logistic regression on the same test posts: posting date 0.642, post + history words + date 0.665.

- The history as text after the post (B, E) gives about +0.02 AUROC over the post alone; 8,192 tokens are not better than 2,048.
- The items encoded apart (C) give no gain, with mean or attention pooling (settings not tuned: 32 items, 256 tokens).
- One seed; the post-only models are at the level of the date alone, and no model here controls for the date.
- Peak GPU memory: A/B 13.4 GiB, C 27.2 GiB, E 32.9 GiB. Time per epoch (one RTX 6000 Ada, grouped): A 2.5 min, B 10, C 22, E 45.
