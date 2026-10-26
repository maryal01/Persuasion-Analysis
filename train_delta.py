import argparse
import glob
import os

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
MODEL_NAME = "answerdotai/ModernBERT-base" 
MODEL_REPO = "maryal01/nlp-project-modernbert" 
MAX_LENGTH = 4096
BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5
SAVE_STEPS = 500  # eval + checkpoint interval; bounds the work lost when a job hits its time limit

HF_TOKEN = os.environ.get("HF_TOKEN")

def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune ModernBERT on a persuasion dataset.")
    ap.add_argument("--dataset-config", default = "cmv", help="dataset config/subset name (e.g. cmv, ddo); omit for default")
    ap.add_argument("--push", action="store_true", help=f"push the (public) model to {MODEL_REPO}")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check")
    ap.add_argument("--freeze", action="store_true", help="freeze the encoder, train only the classifier head (cheap, lower accuracy)")
    return ap.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    labels = np.asarray(labels)
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": float(acc), "precision": prec, "recall": rec, "f1": f1}

def last_complete_checkpoint(outputdir):
    """Newest checkpoint-N in outputdir, or None. The Trainer writes trainer_state.json last,
    so a checkpoint cut off mid-save (e.g. by a Slurm time limit) lacks it and is skipped.
    """
    states = glob.glob(os.path.join(outputdir, "checkpoint-*", "trainer_state.json"))
    if not states:
        return None
    return max((os.path.dirname(s) for s in states), key=lambda d: int(d.rsplit("-", 1)[1]))

def main():
    args = parse_args()
    login(token=HF_TOKEN)
    
    epochs     = 1   if args.smoke else EPOCHS
    max_steps  = 4   if args.smoke else -1
    batch_size = 1   if args.smoke else BATCH_SIZE
    grad_accum = 1   if args.smoke else GRAD_ACCUM
    max_length = 256 if args.smoke else MAX_LENGTH
    save_steps = 2   if args.smoke else SAVE_STEPS

    ds = load_dataset(DATASET_REPO, args.dataset_config, token=HF_TOKEN).rename_column("label", "labels")
    if args.smoke:
        ds = DatasetDict({k: v.select(range(32)) for k, v in ds.items()})

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    def tokenize(batch):
        return tok(batch["original"], batch["reply"], truncation="longest_first", max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["original", "reply"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
        token=HF_TOKEN,
    )

    if args.freeze:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("[FREEZE] encoder frozen; training classifier head only")

    collator = DataCollatorWithPadding(tokenizer=tok)
    # smoke runs get their own dir so a real run never resumes from a smoke checkpoint
    outputdir = f"outputs/{args.dataset_config}" + ("-smoke" if args.smoke else "")
    targs = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LR,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,  # the best checkpoint is kept on top of this
        logging_steps=25,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="f1",
        report_to="none",
        push_to_hub=args.push and not args.smoke,
        hub_model_id=MODEL_REPO,
        hub_private_repo=False,
        hub_token=HF_TOKEN,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=collator, compute_metrics=compute_metrics,
    )

    # Continue a run cut off by a time limit; delete outputs/<config> to start over.
    resume = None if args.smoke else last_complete_checkpoint(outputdir)
    if resume:
        print(f"[RESUME] continuing from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    print("\n=== TEST METRICS ===")
    for k, v in trainer.evaluate(ds["test"]).items():
        if k.startswith("eval_"):
            name = k[5:]
            print(f"  {name:10s}: {v:.4f}" if isinstance(v, float) else f"  {name}: {v}")

    if args.push and not args.smoke:
        trainer.push_to_hub()
        print(f"\nPushed model -> https://huggingface.co/{MODEL_REPO} (public)")
    elif not args.smoke:
        trainer.save_model(outputdir)
        tok.save_pretrained(outputdir)
        print(f"\nSaved model locally -> {outputdir}/  (add --push to upload)")

if __name__ == "__main__":
    main()
