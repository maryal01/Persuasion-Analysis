# Persuasion Classification (CMV + DDO)

Fine-tune ModernBERT to predict persuasion on two platforms. Every config shares
the `{original, reply, label}` text schema, plus id columns used for splitting
and analysis:

| Config | Task | `original` | `reply` | `label` | Extra columns |
|--------|------|------------|---------|---------|---------------|
| `cmv` | T1: CMV persuasion (Webis-CMV-20, Al Khatib et al. 2020) | post title + selftext | the persuader's comments in one reply branch | `delta` vs `nodelta` | `pair_id`, `post_id` |
| `ddo_switch` | T2: did a decided voter switch sides? (DDO, Durmus & Cardie 2019) | resolution + voter's side before | full debate transcript | `switched` vs `stayed` | `debate_id`, `voter`, `before`, `after`, `year`, `filled_other_criteria` |
| `ddo_undecided` | T2: which side does an undecided (tied) voter end on? | resolution + voter's side before (`Tie`) | full debate transcript | `tie` / `pro` / `con` | same as `ddo_switch` |

Each example is a **(context, reply) pair** — a reply is only meaningful relative
to the view it argues against, so both are tokenized together as a text pair.

Splits are 80/10/10 by **group**: all rows of a CMV post, or of a DDO debate,
land in the same split (the two DDO configs share one assignment). On DDO, votes
by the debaters themselves are dropped, and so are side-to-tie votes in
`ddo_switch`. Empty turns and the placeholder text `forfeit` are removed from transcripts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate once (needed for private dataset repos). Create a write token at
<https://huggingface.co/settings/tokens>:

```bash
export HF_TOKEN=hf_xxx
```

## 1. Build & publish the datasets — `scripts/create_dataset.py`
Raw inputs (gitignored), relative to the project root:
- `cmv/pairs.jsonl` — decompressed `pairs.jsonl.bz2` from [Webis-CMV-20](https://zenodo.org/records/3778298)
- `ddo/debates.json` — from the DDO corpus

```bash
# build and print split/label counts without pushing
python scripts/create_dataset.py --dry-run

# build and push every config (or one source: --sources cmv)
python scripts/create_dataset.py
```

Pushes go to the `t1-t2` branch of the dataset repo, which is created on first
push, so the shared `main` branch is left alone. Pass `--revision main` to publish there.

## 2. Fine-tune & publish the model — `scripts/train_delta.py`
```bash
# train on a config, save model locally
python scripts/train_delta.py --dataset-config cmv

# train and push to the private repo nlp-final-project11/nlp-project-modernbert-<config>
python scripts/train_delta.py --dataset-config ddo_switch --push

# fast end-to-end pipeline check
python scripts/train_delta.py --smoke

# CMV thread outcome: keep the end of an over-long discussion instead of its start
python scripts/train_delta.py --dataset-config cmv_thread --truncation-side left --bf16

# T1 with vote pairs as extra training: train on cmv_vote_pairs, then fine-tune that model on cmv
python scripts/train_delta.py --dataset-config cmv_vote_pairs --bf16 --early-stopping 5 --push
python scripts/train_delta.py --dataset-config cmv --bf16 --init-model nlp-final-project11/nlp-project-modernbert-cmv_vote_pairs

# pairwise: every vote pair at every depth of a thread plus delta-vs-well-voted pairs, then cmv, both
# trained on comparisons; compare with the plain pairwise cmv run (outputs/cmv-pairwise)
python scripts/train_delta.py --dataset-config cmv_vote_pairs_all --pairwise --bf16 --early-stopping 5 --push
python scripts/train_delta.py --dataset-config cmv --pairwise --bf16 --init-model nlp-final-project11/nlp-project-modernbert-cmv_vote_pairs_all-pairwise
python scripts/train_delta.py --dataset-config cmv --pairwise --bf16
```

`--pairwise` trains on comparisons instead of single rows, as reward models are trained: each step
scores both replies of a pair (`pair_id`) with the same network, and the loss,
−log σ(score(label 1) − score(label 0)), is small when the label-1 reply (`delta`, `higher`) scores
higher. Only the order within a pair matters, so a reply can win one pair and lose another, and a
reply in several pairs is several lessons. A reply's score is logit 1 − logit 0, the score the test
ranks by. Validation and test are still scored row by row; the best checkpoint is the one with the
highest validation pairwise accuracy. `--batch-size` then counts pairs (two texts each), and the run is
named `<config>-pairwise`. Every test with pairs also reports `pairwise_acc_shorter_won`, the accuracy on
pairs whose label-1 reply is the shorter one: below 0.5 means the model mostly ranks by length.

Against overfitting, every run evaluates on the validation split every 500 steps and ends on its best
checkpoint there; `--early-stopping 5` also stops training once five evaluations in a row have not
improved on it, so a long run does not keep fitting the training replies. Validation and test posts
come from other OPs than the training posts, so they measure replies the model has not seen.

`--init-model` starts from a fine-tuned model (a Hub repo or a local folder) instead of
ModernBERT-base. The run is named `<config>-from-<model>` (here `cmv-from-cmv_vote_pairs`), for its
folder in `outputs/` and its Hub repo, so it never resumes from or overwrites the plain `cmv` run.
When both configs have two labels, the classifier head carries over with its class 1
(`higher` becomes `delta`); otherwise a new head is trained. Compare it with a plain `cmv` run
trained with the same flags.

Data is loaded from the dataset repo's `main` branch. Its `cmv` config has a different split from
the `t1-t2` branch's `cmv`, which the first T1 run in `RESULTS.md` used: only 56 of the 578 `t1-t2`
test posts are `main` test posts, so that run's 66.1% is not comparable with runs on `main`.

Inputs are cut to 8,192 tokens, the most ModernBERT reads. `cmv_thread` (a post and its whole
discussion up to the OP's first award attempt) has long inputs: 23% of the threads exceed that.
`--truncation-side left` cuts the start of the discussion rather than its end, which keeps the
comment that earned the delta in 97% of delta threads, against 93% with the default; a post longer
than half the window then loses its start too (10 posts).

The loss is class-weighted (inverse frequency), and the best checkpoint is chosen
by validation AUROC (binary configs) or macro-F1 (`ddo_undecided`). Test
metrics include AUROC for binary configs and, for `cmv`, pairwise accuracy: the
share of pairs whose delta reply scores higher than its no-delta reply.

Runs save a checkpoint every 500 steps and resume from the newest one in
`outputs/<run>/`; delete that directory to start over. At most two checkpoints are kept (the newest
and the best, about 1.8 GB each), and a finished run deletes them, keeping only its final model.

## 3. Train on Gilbreth — `jobs/train.sub`
Clone the repo onto scratch, then create the environment at `$RCAC_SCRATCH/persuasion-env`
with the `conda` module, as RCAC recommends; the commands are in the header of `jobs/train.sub`.
That header also shows how to submit, and how to chain several 4-hour standby jobs into
one longer run.

On Scholar, use `jobs/scholar.sub` the same way (`sbatch jobs/scholar.sub cmv`, no `-A`). It trains
on both A40s of a node with `torchrun`, one row per GPU per forward pass; `train_delta.py` sets the
gradient accumulation so every optimizer step still covers 32 rows (`--batch-size` changes the rows
per GPU).

Rules worth knowing (RCAC policies): run everything heavier than setup through `sbatch`,
since busy processes on login nodes are killed automatically. Never put tokens in scripts,
which RCAC logs. Scratch files untouched for 60 days are purged, so keep results elsewhere
(`--push`, Data Depot, or Fortress). To set the token without saving it in your shell history:

```bash
read -rs HF_TOKEN && export HF_TOKEN
```
