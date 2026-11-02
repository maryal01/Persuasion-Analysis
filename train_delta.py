import argparse
import os
from collections import defaultdict

import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATASET_REPO = "nlp-final-project11/persuasion-datasets"
DATASET_REVISION = "t1-t2"  # Hub branch holding the T1/T2 configs built by create_dataset.py
MODEL_NAME = "answerdotai/ModernBERT-base" 
MODEL_REPO = "nlp-final-project11/nlp-project-modernbert"  # pushed to {MODEL_REPO}-{config}
MAX_LENGTH = 4096
BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5

HF_TOKEN = os.environ.get("HF_TOKEN")

def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune ModernBERT on a persuasion dataset.")
    ap.add_argument("--dataset-config", default = "cmv", help="dataset config/subset name (cmv, ddo_switch, ddo_undecided); omit for default")
    ap.add_argument("--dataset-revision", default=DATASET_REVISION, help="branch of the dataset repo to load (default: %(default)s)")
    ap.add_argument("--push", action="store_true", help=f"push the model to a private repo, {MODEL_REPO}-<config>")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check")
    ap.add_argument("--freeze", action="store_true", help="freeze the encoder, train only the classifier head (cheap, lower accuracy)")
    return ap.parse_args()

def binary_auroc(scores, labels):
    """Chance a random positive outscores a random negative (ties count half); nan if a class is absent."""
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if not pos or not neg:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # 1-based ranks, averaged over ties
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits, labels = np.asarray(logits), np.asarray(labels)
    preds = logits.argmax(-1)
    per_class = []
    for c in range(logits.shape[1]):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class.append((prec, rec, 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0))
    metrics = {"accuracy": float((preds == labels).mean()), "macro_f1": float(np.mean([f for _, _, f in per_class]))}
    if logits.shape[1] == 2:  # report the positive class (delta, switched)
        prec, rec, f1 = per_class[1]
        metrics.update(precision=prec, recall=rec, f1=f1, auroc=binary_auroc(logits[:, 1] - logits[:, 0], labels))
    return metrics

def pairwise_accuracy(scores, labels, pair_ids):
    """T1: share of CMV pairs whose delta reply outscores its no-delta reply (ties count half)."""
    pairs = defaultdict(dict)
    for s, l, p in zip(scores, labels, pair_ids):
        pairs[p][int(l)] = s
    wins = [float(d[1] > d[0]) + 0.5 * float(d[1] == d[0]) for d in pairs.values() if len(d) == 2]
    return float(np.mean(wins)) if wins else float("nan")

def main():
    args = parse_args()
    login(token=HF_TOKEN)
    
    epochs     = 1   if args.smoke else EPOCHS
    max_steps  = 4   if args.smoke else -1
    batch_size = 1   if args.smoke else BATCH_SIZE
    grad_accum = 1   if args.smoke else GRAD_ACCUM
    max_length = 256 if args.smoke else MAX_LENGTH

    ds = load_dataset(DATASET_REPO, args.dataset_config, revision=args.dataset_revision, token=HF_TOKEN).rename_column("label", "labels")
    if args.smoke:
        ds = DatasetDict({k: v.select(range(32)) for k, v in ds.items()})

    label_names = ds["train"].features["labels"].names

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    def tokenize(batch):
        return tok(batch["original"], batch["reply"], truncation="longest_first", max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["original", "reply"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        token=HF_TOKEN,
    )

    if args.freeze:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("[FREEZE] encoder frozen; training classifier head only")

    collator = DataCollatorWithPadding(tokenizer=tok)
    outputdir = f"outputs/{args.dataset_config}"
    model_repo = f"{MODEL_REPO}-{args.dataset_config}"
    targs = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=not args.smoke,
        # AUROC ranks without a threshold; positive-class F1 rewards predicting that class often,
        # so on balanced cmv it preferred checkpoints that call most replies "delta"
        metric_for_best_model="auroc" if len(label_names) == 2 else "macro_f1",
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
        data_collator=collator, compute_metrics=compute_metrics,
    )

    trainer.train()

    pred = trainer.predict(ds["test"])
    metrics = {k[5:]: v for k, v in pred.metrics.items() if k.startswith("test_")}
    if "pair_id" in ds["test"].column_names:
        scores = pred.predictions[:, 1] - pred.predictions[:, 0]
        metrics["pairwise_acc"] = pairwise_accuracy(scores, pred.label_ids, ds["test"]["pair_id"])
    print("\n=== TEST METRICS ===")
    for name, v in metrics.items():
        print(f"  {name:10s}: {v:.4f}" if isinstance(v, float) else f"  {name}: {v}")

    if args.push and not args.smoke:
        trainer.push_to_hub()
        print(f"\nPushed model -> https://huggingface.co/{model_repo} (private)")
    elif not args.smoke:
        trainer.save_model(outputdir)
        tok.save_pretrained(outputdir)
        print(f"\nSaved model locally -> {outputdir}/  (add --push to upload)")

if __name__ == "__main__":
    main()
