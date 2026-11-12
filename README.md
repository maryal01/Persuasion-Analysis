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
`ddo_switch`. `ddo_undecided` also leaves out tied votes cast before about September 2008 (dated
"10 years ago" in the corpus): 98.9% of the votes from that period share one fixed pattern, which
points to an older voting form that did not ask for a stance. Empty turns and the placeholder text `forfeit` are removed from transcripts.

T5 (will the OP award a delta, given their history?) uses two more configs, both split by OP:

| Config | Rows | Columns |
|--------|------|---------|
| `cmv_malleability` | one per CMV post that was not removed and has a known author (41,714) | `original` (title + selftext), `label` (`delta` if DeltaBot confirmed a delta the OP awarded), `post_id`, `author`, `created_utc` |
| `cmv_history` | every post and comment those OPs wrote in r/changemyview (1,684,662) | `author`, `kind` (`post`, `comment`, `own_thread_comment`), `created_utc`, `edited_utc`, `text` |

The history of a post is the `cmv_history` rows of its `author` with `created_utc` before the post's
and `edited_utc` null or before it. Apply both conditions: the stored text is the edited version, so a
later edit can reveal the outcome. Histories are not truncated (21% of posts have more than about
8,192 tokens of history; 39% have none). Delta symbols are removed from all text.

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
- `cmv/threads.jsonl` — decompressed `threads.jsonl.bz2` from Webis-CMV-20 (3.6 GB; only for `--sources cmv_history`)

```bash
# build and print split/label counts without pushing
python scripts/create_dataset.py --dry-run

# build and push every config (or one source: --sources cmv)
python scripts/create_dataset.py

# T5 configs only (about 2 minutes, 2.5 GB of memory)
python scripts/create_dataset.py --sources cmv_history
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
```

Data is loaded from the `t1-t2` branch by default; `--dataset-revision main`
loads the shared branch instead.

The loss is class-weighted (inverse frequency), and the best checkpoint is chosen
by validation AUROC (binary configs) or macro-F1 (`ddo_undecided`). Test
metrics include AUROC for binary configs and, for `cmv`, pairwise accuracy: the
share of pairs whose delta reply scores higher than its no-delta reply.

Runs save a checkpoint every 500 steps and resume from the newest one in
`outputs/<config>/`; delete that directory to start over.

## 3. Train on Gilbreth — `jobs/train.sub`
Clone the repo onto scratch, then create the environment at `$RCAC_SCRATCH/persuasion-env`
with the `conda` module, as RCAC recommends; the commands are in the header of `jobs/train.sub`.
That header also shows how to submit, and how to chain several 4-hour standby jobs into
one longer run.

Rules worth knowing (RCAC policies): run everything heavier than setup through `sbatch`,
since busy processes on login nodes are killed automatically. Never put tokens in scripts,
which RCAC logs. Scratch files untouched for 60 days are purged, so keep results elsewhere
(`--push`, Data Depot, or Fortress). To set the token without saving it in your shell history:

```bash
read -rs HF_TOKEN && export HF_TOKEN
```
