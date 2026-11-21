"""T3 (audience profiling): fine-tune ModernBERT to predict a DDO user's self-reported stances from their text.

One model for every label column of `ddo_users` (the four issue dimensions, religion and the political
ideology group). The encoder reads one document per user and the classifier outputs one block of logits
per label column. The loss is the class-weighted cross-entropy of each label the user has, so a user who
reported only some stances still trains the model.

Users, labels, class names, splits and the text rule are those of `scripts/t3/baselines.py`, so the test
numbers can be set next to its majority, random and TF-IDF rows:

  - only users with at least one label and at least one text item
  - a user's text items are debate arguments, comments, opinion arguments and poll votes, each with its
    title in front; votes have no text
  - items from debates in T2's validation or test split are left out, so the model can later write
    profiles for T2 voters without having read those debates

A user's document is their items in a random order (seeded per user) and is cut to --max-length tokens:
the model reads a sample of a long history, not its first items in table order.

The best checkpoint is the one with the highest validation macro-F1 averaged over the label columns.
After training, the validation and test predictions (class and probabilities per label column) are
written next to the checkpoints, and the test table is printed with 95% intervals from a bootstrap over
users.

A run with another seed than 42 (--seed <N>) keeps its own folder and model repo, named
t3_profiles-seed<N>. The seed sets the classifier's start weights, the order of the training users and
the order of each user's items, so for a user with a long history it also changes which items are read.

Usage (from the repository root):
    python scripts/t3/train.py --smoke
    python scripts/t3/train.py --bf16 --push
On Gilbreth (one 4-hour standby job should be enough; queue a second part to be safe):
    sbatch -A <account> jobs/train.sub t3_profiles --bf16 --push
    sbatch -A <account> jobs/train.sub t3_profiles --seed 43 --bf16 --push
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import SMOKE_ROWS, SPLITS, bootstrap_by_user, load_items, load_users, metrics  # noqa: E402
from train_delta import last_complete_checkpoint  # noqa: E402  (shared with the T1/T2 training script)

RUN_NAME = "t3_profiles"  # outputs/<RUN_NAME>/ and the config name for jobs/train.sub
DEFAULT_SEED = 42  # the seed of the first run; a run with this seed keeps the plain name
MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_REPO = f"nlp-final-project11/nlp-project-modernbert-{RUN_NAME}"
MAX_LENGTH = 4096
BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5
EVALS_PER_EPOCH = 2  # evaluation + checkpoint interval; bounds the work lost when a job hits its time limit
CHARS_PER_TOKEN = 8  # generous: a document is cut to max_length * this many characters before tokenizing
SMOKE_USERS = 32
UNKNOWN = -100  # label value the loss ignores (the user did not report this stance)

HF_TOKEN = os.environ.get("HF_TOKEN")

Blocks = Dict[str, Tuple[int, int]]  # label column -> (first, one past last) position of its logits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune ModernBERT for T3 audience profiling (one model, all labels).")
    ap.add_argument("--dataset-repo", default="nlp-final-project11/persuasion-datasets")
    ap.add_argument("--dataset-revision", default="main", help="branch of the dataset repo (default: %(default)s)")
    ap.add_argument("--users-config", default="ddo_users", help="one row per user with the label columns")
    ap.add_argument("--history-config", default="ddo_history", help="one row per activity item")
    ap.add_argument("--model-name", default=MODEL_NAME, help="encoder to fine-tune (default: %(default)s)")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH, help="tokens read per user (default: %(default)s)")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="plain cross-entropy; the default weights classes by inverse frequency within each label")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="training seed and item order (default: %(default)s). A run with another seed gets "
                         "-seed<N> at the end of its name: its own outputs/ folder and model repo")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples of the test users")
    ap.add_argument("--push", action="store_true",
                    help=f"push the model to the private repo {MODEL_REPO} (with -seed<N> for another seed)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check")
    ap.add_argument("--bf16", action="store_true", help="train with bf16 mixed precision (default: fp32)")
    ap.add_argument("--cpu", action="store_true", help="do not use a GPU (local checks only)")
    return ap.parse_args()


def label_blocks(class_names: Dict[str, List[str]]) -> Blocks:
    """Where each label column's logits sit in the classifier output."""
    blocks, start = {}, 0
    for column, classes in class_names.items():
        blocks[column] = (start, start + len(classes))
        start += len(classes)
    return blocks


def user_document(items: List[str], user: str, seed: int, max_chars: int) -> str:
    """The user's items in a random order (the same for a given seed and user), joined by blank lines.
    Items are added until the document has max_chars characters; the tokenizer makes the exact cut.
    """
    order = list(items)
    random.Random(f"{seed}:{user}").shuffle(order)
    parts, length = [], 0
    for item in order:
        parts.append(item)
        length += len(item) + 2
        if length >= max_chars:
            break
    return "\n\n".join(parts)


def build_splits(users: Dict[str, List[str]], labels: Dict[str, Dict[str, np.ndarray]], columns: List[str],
                 items: Dict[str, List[str]], seed: int, max_chars: int) -> DatasetDict:
    """One row per user who has text and at least one label: user, text, labels (one class index per
    label column, UNKNOWN where the user has none)."""
    splits = {}
    for split in SPLITS:
        rows = {"user": [], "text": [], "labels": []}
        for i, user in enumerate(users[split]):
            values = [int(labels[split][column][i]) for column in columns]
            if user not in items or all(v < 0 for v in values):
                continue
            rows["user"].append(user)
            rows["text"].append(user_document(items[user], user, seed, max_chars))
            rows["labels"].append([v if v >= 0 else UNKNOWN for v in values])
        splits[split] = Dataset.from_dict(rows)
    return DatasetDict(splits)


def class_weights(train_labels: np.ndarray, class_names: Dict[str, List[str]], weighted: bool) -> Dict[str, torch.Tensor]:
    """Inverse-frequency weights per label column, averaging 1 over the training users who have that label;
    all 1s when not weighted."""
    weights = {}
    for k, (column, classes) in enumerate(class_names.items()):
        known = train_labels[:, k][train_labels[:, k] != UNKNOWN]
        counts = np.bincount(known, minlength=len(classes))
        w = counts.sum() / (len(classes) * np.maximum(counts, 1)) if weighted else np.ones(len(classes))
        weights[column] = torch.tensor(w, dtype=torch.float32)
        print(f"[CLASSES] {column}: " + ", ".join(f"{n}={c} (weight {x:.2f})" for n, c, x in zip(classes, counts, w)))
    return weights


def multi_label_loss(blocks: Blocks, weights: Dict[str, torch.Tensor]) -> Callable:
    """Loss for Trainer(compute_loss_func=...): the class-weighted cross-entropy of every known label,
    summed over the label columns. The Trainer expects such a loss to be normalized over the whole
    accumulated batch, so sum it and divide by that batch's count of known labels.
    """
    def loss_fn(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        total = logits.new_zeros(())
        for k, (column, (start, end)) in enumerate(blocks.items()):
            total = total + F.cross_entropy(logits[:, start:end], labels[:, k], ignore_index=UNKNOWN, reduction="sum",
                                            weight=weights[column].to(logits.device, logits.dtype))
        known = num_items_in_batch if num_items_in_batch is not None else (labels != UNKNOWN).sum().clamp(min=1)
        return total / known
    return loss_fn


def make_compute_metrics(blocks: Blocks) -> Callable:
    """Accuracy and macro-F1 per label column over the users who have that label, and their means."""
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = np.asarray(eval_pred[0]), np.asarray(eval_pred[1])
        out, per_metric = {}, {"accuracy": [], "macro_f1": []}
        for k, (column, (start, end)) in enumerate(blocks.items()):
            known = labels[:, k] != UNKNOWN
            if not known.any():
                continue
            scores = metrics(labels[known, k], logits[known, start:end].argmax(-1), end - start)
            for name, value in scores.items():
                out[f"{column}_{name}"] = value
                per_metric[name].append(value)
        out.update({f"mean_{name}": float(np.mean(values)) if values else 0.0 for name, values in per_metric.items()})
        return out
    return compute_metrics


def save_predictions(path: str, users: List[str], logits: np.ndarray, labels: np.ndarray, blocks: Blocks) -> None:
    """One row per user: for each label column the true class (-1 if unknown), the predicted class and
    the class probabilities."""
    columns = {"user": users}
    for k, (column, (start, end)) in enumerate(blocks.items()):
        probs = torch.softmax(torch.tensor(logits[:, start:end], dtype=torch.float32), dim=-1).numpy()
        columns[column] = [int(v) if v != UNKNOWN else -1 for v in labels[:, k]]
        columns[f"{column}_pred"] = probs.argmax(-1).tolist()
        columns[f"{column}_probs"] = probs.tolist()
    Dataset.from_dict(columns).to_parquet(path)


def test_table(logits: np.ndarray, labels: np.ndarray, train_labels: np.ndarray, class_names: Dict[str, List[str]],
               blocks: Blocks, resamples: int) -> Dict[str, dict]:
    """Print and return the test metrics per label column, next to the majority class of the training users."""
    results = {}
    print("\n=== TEST, held-out users: accuracy and macro-F1 with 95% intervals ===")
    for k, (column, (start, end)) in enumerate(blocks.items()):
        classes, known = class_names[column], labels[:, k] != UNKNOWN
        train_known = train_labels[:, k][train_labels[:, k] != UNKNOWN]
        if not known.any() or not len(train_known):
            print(f"  {column}: skipped, no labelled users")
            continue
        y = labels[known, k]
        majority = int(np.bincount(train_known, minlength=len(classes)).argmax())
        preds = {"majority": np.full(len(y), majority), "modernbert": logits[known, start:end].argmax(-1)}
        intervals = bootstrap_by_user(y, preds, len(classes), resamples)
        results[column] = {"classes": classes, "test_users": int(len(y)),
                           "models": {name: {**metrics(y, pred, len(classes)), "ci95": intervals[name]}
                                      for name, pred in preds.items()}}
        print(f"  {column} ({' / '.join(classes)}): {len(y)} test users")
        for name, r in results[column]["models"].items():
            (acc_low, acc_high), (f1_low, f1_high) = r["ci95"]["accuracy"], r["ci95"]["macro_f1"]
            print(f"    {name:12s} accuracy {r['accuracy']:.3f} ({acc_low:.3f}-{acc_high:.3f})   "
                  f"macro-F1 {r['macro_f1']:.3f} ({f1_low:.3f}-{f1_high:.3f})")
        low, high = intervals["modernbert"]["accuracy_minus_majority"]
        difference = results[column]["models"]["modernbert"]["accuracy"] - results[column]["models"]["majority"]["accuracy"]
        print(f"    accuracy of modernbert minus majority: {difference:+.3f} ({low:+.3f} to {high:+.3f})")
    return results


def main() -> None:
    args = parse_args()
    epochs = 1 if args.smoke else EPOCHS
    max_steps = 4 if args.smoke else -1
    batch_size = 1 if args.smoke else BATCH_SIZE
    grad_accum = 1 if args.smoke else GRAD_ACCUM
    max_length = 256 if args.smoke else args.max_length

    users, labels, class_names = load_users(args.dataset_repo, args.users_config, args.dataset_revision)
    columns = list(class_names)
    blocks = label_blocks(class_names)
    wanted = {user for split in SPLITS for i, user in enumerate(users[split])
              if any(labels[split][column][i] >= 0 for column in columns)}
    items, counts = load_items(args.dataset_repo, args.history_config, args.dataset_revision, wanted,
                               all_debates=False, titles=True, max_rows=SMOKE_ROWS if args.smoke else None)
    for name, n in counts.most_common():
        print(f"[{args.history_config}] {name}: {n}")

    ds = build_splits(users, labels, columns, items, args.seed, max_length * CHARS_PER_TOKEN)
    if args.smoke:
        ds = DatasetDict({k: v.select(range(min(SMOKE_USERS, len(v)))) for k, v in ds.items()})
    print("[USERS] " + ", ".join(f"{split} {len(ds[split])}" for split in SPLITS) + " with text and at least one label")
    split_users = {split: list(ds[split]["user"]) for split in SPLITS}
    split_labels = {split: np.array(ds[split]["labels"]).reshape(len(ds[split]), len(columns)) for split in SPLITS}
    weights = class_weights(split_labels["train"], class_names, not args.no_class_weights)

    tok = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["user", "text"])
    lengths = np.array([len(ids) for ids in ds["train"]["input_ids"]])
    print(f"[TOKENS] training users: median {np.median(lengths):.0f} tokens, "
          f"{np.mean(lengths >= max_length):.1%} cut at {max_length}")

    names = [f"{column}={cls}" for column, classes in class_names.items() for cls in classes]
    # The classifier on top of the encoder starts from random weights, drawn here. The Trainer sets the
    # seed only when it is built, which is after this, so set it now.
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(names),
        id2label=dict(enumerate(names)),  # "<label column>=<class>", in the order of the logits
        label2id={name: i for i, name in enumerate(names)},
        token=HF_TOKEN,
    )

    use_bf16 = args.bf16 and not args.cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.bf16 and not use_bf16:
        print("[PRECISION] --bf16 requested but no GPU here supports it; using fp32")
    print(f"[PRECISION] {'bf16 mixed precision' if use_bf16 else 'fp32'}")
    steps_per_epoch = math.ceil(len(ds["train"]) / (batch_size * grad_accum))
    save_steps = 2 if args.smoke else max(1, steps_per_epoch // EVALS_PER_EPOCH)
    # a run with another seed keeps its own outputs and model repo; jobs/train.sub derives the same name
    run_name = RUN_NAME + ("" if args.seed == DEFAULT_SEED else f"-seed{args.seed}")
    model_repo = MODEL_REPO + ("" if args.seed == DEFAULT_SEED else f"-seed{args.seed}")
    # smoke runs get their own dir so a real run never resumes from a smoke checkpoint
    outputdir = f"outputs/{run_name}" + ("-smoke" if args.smoke else "")
    targs = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LR,
        bf16=use_bf16,
        use_cpu=args.cpu,
        seed=args.seed,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,  # the best checkpoint is kept on top of this
        logging_steps=25,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="mean_macro_f1",
        report_to="none",
        push_to_hub=args.push and not args.smoke,
        hub_model_id=model_repo,
        hub_private_repo=True,  # only applies when the repo is created; change visibility on the Hub
        # no hub_token: it would be pickled into every checkpoint's training_args.bin and pushed
        # with the model; the Hub client reads HF_TOKEN from the environment instead
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=make_compute_metrics(blocks),
        compute_loss_func=multi_label_loss(blocks, weights),
    )

    # Continue a run cut off by a time limit; delete outputs/<run name> to start over.
    resume: Optional[str] = None if args.smoke else last_complete_checkpoint(outputdir)
    if resume:
        print(f"[RESUME] continuing from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    if torch.cuda.is_available() and not args.cpu:  # peak over this process, to size future runs
        print(f"[VRAM] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")

    logits = {split: np.asarray(trainer.predict(ds[split]).predictions) for split in ("validation", "test")}
    results = test_table(logits["test"], split_labels["test"], split_labels["train"], class_names, blocks, args.bootstrap)

    # The model first. A push uploads everything in the output folder except checkpoints, so the files
    # written below (one row per user, with user names) stay on this machine and are not uploaded.
    if args.push and not args.smoke:
        trainer.push_to_hub()
        print(f"\nPushed model -> https://huggingface.co/{model_repo} (private)")
    elif not args.smoke:
        trainer.save_model(outputdir)
        tok.save_pretrained(outputdir)
        print(f"\nSaved model locally -> {outputdir}/  (add --push to upload)")

    for split in ("validation", "test"):
        save_predictions(f"{outputdir}/predictions_{split}.parquet", split_users[split], logits[split],
                         split_labels[split], blocks)
    with open(f"{outputdir}/test_metrics.json", "w") as fh:
        json.dump({"run": run_name, "dataset": {"repo": args.dataset_repo, "revision": args.dataset_revision},
                   "options": {"model": args.model_name, "max_length": max_length, "seed": args.seed,
                               "class_weights": not args.no_class_weights, "bf16": use_bf16, "smoke": args.smoke},
                   "users": {split: len(ds[split]) for split in SPLITS}, "labels": results}, fh, indent=2)
    print(f"\nWrote {outputdir}/test_metrics.json and predictions_validation/test.parquet")


if __name__ == "__main__":
    main()
