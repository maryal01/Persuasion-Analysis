# Persuasion Classification (CMV + DDO)

Fine-tune ModernBERT to predict whether an argument is persuasive, from two
sources with a shared `{original, reply, label}` schema:

| Config | Source | `original` | `reply` | `label` |
|--------|--------|------------|---------|---------|
| `cmv`  | Reddit r/changemyview ("Winning Arguments", Tan et al. 2016) | post title + selftext | a reply branch | `delta` (changed OP's mind) vs `nodelta` |
| `ddo`  | debate.org ("Exploring the Role of Prior Beliefs", Durmus & Cardie 2018) | debate resolution | one side's rounds | `won` vs `lost` |

Each example is a **(context, reply) pair** — a reply is only meaningful relative
to the view it argues against, so both are tokenized together as a text pair.

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
```bash
# run from the project root so the relative paths in SOURCES resolve
python create_dataset.py
```

## 2. Fine-tune & publish the model — `train_delta.py`
```bash
# train on a config, save model locally
python train_delta.py --dataset-config cmv

# train and push a model
python train_delta.py --dataset-config ddo --push

# fast end-to-end pipeline check
python train_delta.py --smoke
```
