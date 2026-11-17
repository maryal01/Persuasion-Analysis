"""T3: write a profile for every DDO user from their text, with the trained T3 model.

`scripts/t3/train.py` trains one ModernBERT for the six label columns of `ddo_users`. This script runs
that model over every user who has at least one usable text item, whether or not they reported any
stance, and writes one row per user (config `ddo_profiles_inferred`):

    user                   the DDO user name
    <label column>         the predicted class, with the class names of ddo_users (one column per label)
    <label column>_probs   the model's probability of each class
    tokens                 how many tokens of the user's text the model read
    in_training            True if the model was trained on this user (a training user with a label).
                           For those users the profile is not a fair test of the model.

The text rule is the one of training: debate arguments, comments, opinion arguments and poll votes with
their titles, without items from debates in T2's validation or test split, in the same seeded random
order and cut to the same number of tokens. So a user gets the same document here as in training.
Rows keep the split of `ddo_users`. Users without text are not in the table.

The table holds stances the users did not report themselves (the model's guesses, among others about
religion and politics). Keep it in the private dataset repo.

Usage (from the repository root):
    python scripts/t3/infer_profiles.py --smoke --cpu --model <path of a trained model>
On Gilbreth, after the T3 training run has pushed its model:
    sbatch -A <account> jobs/train.sub t3_profiles_infer --bf16 --push
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import SMOKE_ROWS, SPLITS, load_items, load_users  # noqa: E402
from train import CHARS_PER_TOKEN, HF_TOKEN, MAX_LENGTH, MODEL_NAME, MODEL_REPO, SMOKE_USERS, label_blocks, user_document  # noqa: E402

RUN_NAME = "t3_profiles_infer"  # outputs/<RUN_NAME>/ and the name for jobs/train.sub
HUB_CONFIG = "ddo_profiles_inferred"
HUB_REVISION = "t3"  # staging branch of the dataset repo; the config is copied to main afterwards


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict a profile for every DDO user with text, using the T3 model.")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo to read (default: %(default)s)")
    ap.add_argument("--users-config", default="ddo_users", help="one row per user with the label columns")
    ap.add_argument("--history-config", default="ddo_history", help="one row per activity item")
    ap.add_argument("--model", default=MODEL_REPO, help="trained T3 model: a path or a Hub repo (default: %(default)s)")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH, help="tokens read per user; use the training value (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42, help="item order; use the training seed (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=8, help="users per batch (default: %(default)s)")
    ap.add_argument("--push", action="store_true",
                    help=f"upload the table as config {HUB_CONFIG} to branch --hub-revision of the dataset repo")
    ap.add_argument("--hub-revision", default=HUB_REVISION, help="branch that --push uploads to (default: %(default)s)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check; never pushes")
    ap.add_argument("--bf16", action="store_true", help="run the model with bf16 mixed precision, as in a --bf16 training run")
    ap.add_argument("--cpu", action="store_true", help="do not use a GPU (local checks only)")
    return ap.parse_args()


def load_model(source: str, class_names: Dict[str, List[str]]) -> AutoModelForSequenceClassification:
    """The trained classifier, after checking that its outputs are the label columns and classes of the dataset."""
    model = AutoModelForSequenceClassification.from_pretrained(source, token=HF_TOKEN)
    expected = [f"{column}={cls}" for column, classes in class_names.items() for cls in classes]
    found = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if found != expected:
        raise SystemExit(f"{source} predicts {found}, but the users table has {expected}")
    return model


def predict(model: AutoModelForSequenceClassification, tok: AutoTokenizer, texts: List[str], max_length: int,
            batch_size: int, device: torch.device, bf16: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Logits and the number of tokens read, per text. Texts are batched by length to waste little padding."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    logits = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    tokens = np.zeros(len(texts), dtype=np.int32)
    model.to(device).eval()
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        batch = tok([texts[i] for i in rows], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(**batch).logits
        logits[rows] = out.float().cpu().numpy()
        tokens[rows] = batch["attention_mask"].sum(dim=1).cpu().numpy()
        if (start // batch_size) % 200 == 0:
            print(f"[PREDICT] {start + len(rows)} of {len(texts)} users", flush=True)
    return logits, tokens


def build_table(users: List[str], splits: List[str], in_training: List[bool], logits: np.ndarray, tokens: np.ndarray,
                class_names: Dict[str, List[str]]) -> DatasetDict:
    """One row per user, in the split the user has in the users table."""
    blocks = label_blocks(class_names)
    columns = {"user": users}
    features = {"user": Value("string")}
    for column, (start, end) in blocks.items():
        probs = torch.softmax(torch.tensor(logits[:, start:end]), dim=-1).numpy()
        columns[column], columns[f"{column}_probs"] = probs.argmax(-1).tolist(), probs.tolist()
        features[column], features[f"{column}_probs"] = ClassLabel(names=class_names[column]), Sequence(Value("float32"))
    columns.update(tokens=tokens.tolist(), in_training=in_training)
    features.update(tokens=Value("int32"), in_training=Value("bool"))
    table = Dataset.from_dict(columns, features=Features(features))
    return DatasetDict({split: table.select([i for i, s in enumerate(splits) if s == split]) for split in SPLITS})


def agreement(table: DatasetDict, users: Dict[str, List[str]], labels: Dict[str, Dict[str, np.ndarray]],
              class_names: Dict[str, List[str]]) -> Dict[str, dict]:
    """How often the predicted class equals the self-reported one, per label column, for users the model
    was trained on and for the others (held-out users: the fair number), and how the predictions for
    users without that label are spread over the classes."""
    known = {(split, user): i for split in SPLITS for i, user in enumerate(users[split])}
    out = {}
    for column, classes in class_names.items():
        hits = {"held_out": [], "in_training": []}
        unlabelled = np.zeros(len(classes), dtype=int)
        for split in SPLITS:
            rows = table[split]
            for user, predicted, trained in zip(rows["user"], rows[column], rows["in_training"]):
                reported = int(labels[split][column][known[split, user]])
                if reported < 0:
                    unlabelled[predicted] += 1
                else:
                    hits["in_training" if trained else "held_out"].append(predicted == reported)
        out[column] = {**{f"{group}_users": len(h) for group, h in hits.items()},
                       **{f"{group}_accuracy": float(np.mean(h)) if h else None for group, h in hits.items()},
                       "predicted_for_users_without_this_label": dict(zip(classes, unlabelled.tolist()))}
    return out


def main() -> None:
    args = parse_args()
    max_length = 256 if args.smoke else args.max_length
    users, labels, class_names = load_users(args.dataset_repo, args.users_config, args.dataset_revision)
    everyone = {user for split in SPLITS for user in users[split]}
    items, counts = load_items(args.dataset_repo, args.history_config, args.dataset_revision, everyone,
                               all_debates=False, titles=True, max_rows=SMOKE_ROWS if args.smoke else None)
    for name, n in counts.most_common():
        print(f"[{args.history_config}] {name}: {n}")

    names, splits, in_training, texts = [], [], [], []
    for split in SPLITS:
        for i, user in enumerate(users[split]):
            if user not in items:
                continue
            names.append(user)
            splits.append(split)
            # the rows scripts/t3/train.py trains on: training users with text and at least one label
            in_training.append(split == "train" and any(labels[split][column][i] >= 0 for column in class_names))
            texts.append(user_document(items[user], user, args.seed, max_length * CHARS_PER_TOKEN))
    if args.smoke:
        keep = [i for split in SPLITS for i in [j for j, s in enumerate(splits) if s == split][:SMOKE_USERS]]
        names, splits, in_training, texts = ([column[i] for i in keep] for column in (names, splits, in_training, texts))
    print(f"[USERS] {len(names)} of {len(everyone)} users have text; {sum(in_training)} of them were training rows of the model")

    model = load_model(args.model, class_names)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[MODEL] {args.model} on {device.type}, {'bf16 mixed precision' if use_bf16 else 'fp32'}, {max_length} tokens per user")
    logits, tokens = predict(model, tok, texts, max_length, args.batch_size, device, use_bf16)
    table = build_table(names, splits, in_training, logits, tokens, class_names)

    summary = agreement(table, users, labels, class_names)
    print("\n=== predicted class against the self-reported one ===")
    for column, s in summary.items():
        held, trained = s["held_out_accuracy"], s["in_training_accuracy"]
        print(f"  {column}: held-out users {'-' if held is None else f'{held:.3f}'} ({s['held_out_users']}), "
              f"training users {'-' if trained is None else f'{trained:.3f}'} ({s['in_training_users']}); "
              f"users without this label: {s['predicted_for_users_without_this_label']}")

    outputdir = f"outputs/{RUN_NAME}" + ("-smoke" if args.smoke else "")
    os.makedirs(outputdir, exist_ok=True)
    for split in SPLITS:
        table[split].to_parquet(f"{outputdir}/{split}.parquet")
    with open(f"{outputdir}/summary.json", "w") as fh:
        json.dump({"dataset": {"repo": args.dataset_repo, "revision": args.dataset_revision}, "model": args.model,
                   "options": {"max_length": max_length, "seed": args.seed, "bf16": use_bf16, "smoke": args.smoke},
                   "users": {split: len(table[split]) for split in SPLITS}, "labels": summary}, fh, indent=2)
    print(f"\nWrote {outputdir}/train.parquet, validation.parquet, test.parquet and summary.json: "
          + ", ".join(f"{split} {len(table[split])}" for split in SPLITS) + " users")

    if args.push and not args.smoke:
        HfApi().create_branch(args.dataset_repo, repo_type="dataset", branch=args.hub_revision, exist_ok=True, token=HF_TOKEN)
        table.push_to_hub(args.dataset_repo, config_name=HUB_CONFIG, revision=args.hub_revision, private=True, token=HF_TOKEN)
        print(f"Pushed config {HUB_CONFIG} -> https://huggingface.co/datasets/{args.dataset_repo} (branch {args.hub_revision}, private)")


if __name__ == "__main__":
    main()
