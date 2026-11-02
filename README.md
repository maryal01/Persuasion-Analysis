# Persuasion Classification (CMV + DDO)

Fine-tune ModernBERT to predict persuasion on two platforms. Every config shares
the `{original, reply, label}` text schema, plus id columns used for splitting
and analysis:

| Config | Task | `original` | `reply` | `label` | Extra columns |
|--------|------|------------|---------|---------|---------------|
| `cmv` | T1: CMV persuasion (Webis-CMV-20, Al Khatib et al. 2020) | post title + selftext | the persuader's comments in one reply branch | `delta` vs `nodelta` | `pair_id`, `post_id` |
| `ddo` | who won the debate (debate.org, "Exploring the Role of Prior Beliefs", Durmus & Cardie 2018) | debate resolution | one side's rounds | `won` vs `lost` | `debate_id` |

Each example is a **(context, reply) pair** — a reply is only meaningful relative
to the view it argues against, so both are tokenized together as a text pair.

Splits are 80/10/10 by **group**: all rows of a CMV post, or of a DDO debate,
land in the same split.

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

## 1. Build & publish the datasets — `create_dataset.py`
Raw inputs (gitignored), relative to the project root:
- `cmv/pairs.jsonl` — decompressed `pairs.jsonl.bz2` from [Webis-CMV-20](https://zenodo.org/records/3778298)
- `ddo/debates.json` — from the DDO corpus

```bash
# build and print split/label counts without pushing
python create_dataset.py --dry-run

# build and push every config (or one source: --sources cmv)
python create_dataset.py
```

Pushes go to the `t1-t2` branch of the dataset repo, which is created on first
push, so the shared `main` branch is left alone. Pass `--revision main` to publish there.

## 2. Fine-tune & publish the model — `train_delta.py`
```bash
# train on a config, save model locally
python train_delta.py --dataset-config cmv

# train and push to the private repo nlp-final-project11/nlp-project-modernbert-<config>
python train_delta.py --dataset-config ddo --push

# fast end-to-end pipeline check
python train_delta.py --smoke
```

Data is loaded from the `t1-t2` branch by default; `--dataset-revision main`
loads the shared branch instead.
