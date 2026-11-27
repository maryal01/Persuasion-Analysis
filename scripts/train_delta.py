import argparse
import glob
import os
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATASET_REPO = "nlp-final-project11/persuasion-datasets"
MODEL_NAME = "answerdotai/ModernBERT-base" 
MODEL_REPO = "nlp-final-project11/nlp-project-modernbert"  # pushed to {MODEL_REPO}-{run_name}
MAX_LENGTH = 8192  # the most ModernBERT reads
EFFECTIVE_BATCH = 32  # rows (pairs with --pairwise) per optimizer step over all GPUs; gradient accumulation makes up the rest
BATCH_SIZE = 4  # default rows (pairs) per GPU per forward pass (--batch-size), sized for an 80 GB A100 at 8,192 tokens
EPOCHS = 3
LR = 2e-5
SAVE_STEPS = 500  # eval + checkpoint interval; bounds the work lost when a job hits its time limit

HF_TOKEN = os.environ.get("HF_TOKEN")

def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune ModernBERT on a persuasion dataset.")
    ap.add_argument("--dataset-config", default = "cmv", help="dataset config/subset name (cmv, cmv_thread, ddo_switch, ddo_undecided); omit for default")
    ap.add_argument("--truncation-side", choices=("right", "left"), default="right",
                    help="which end of an over-long text is cut; left keeps the end of the reply, "
                         "i.e. the latest comments of a cmv_thread discussion (default: %(default)s)")
    ap.add_argument("--push", action="store_true", help=f"push the model to a private repo, {MODEL_REPO}-<run> (<config>, or <config>-from-<model>)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check")
    ap.add_argument("--bf16", action="store_true", help="train with bf16 mixed precision (default: fp32, as the T1 baseline)")
    ap.add_argument("--freeze", action="store_true", help="freeze the encoder, train only the classifier head (cheap, lower accuracy)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"rows per GPU per forward pass; gradient accumulation keeps {EFFECTIVE_BATCH} rows per "
                         "optimizer step for any batch size and GPU count (default: %(default)s)")
    ap.add_argument("--epochs", type=float, default=EPOCHS, help="passes over the training split (default: %(default)s)")
    ap.add_argument("--early-stopping", type=int, default=0, metavar="EVALS",
                    help="stop once the validation metric (AUROC; macro-F1 for 3 classes; pairwise accuracy with "
                         f"--pairwise) has not improved for this many evaluations, {SAVE_STEPS} steps apart; "
                         "0 trains every epoch (default: %(default)s)")
    ap.add_argument("--pairwise", action="store_true",
                    help="train on pairs (configs with pair_id: cmv, cmv_vote_pairs, cmv_vote_pairs_all): each step "
                         "scores both replies of a pair and the loss pushes the label-1 reply's score above the "
                         "other's; the best checkpoint is picked by validation pairwise accuracy")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: score the test split with --init-model (e.g. the stage-1 vote model on cmv) "
                         "and print the test metrics; nothing is saved or pushed")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute activations in the backward pass: less GPU memory, about 30%% slower")
    ap.add_argument("--init-model", default=None,
                    help="start from this fine-tuned model (Hub repo or local folder) instead of ModernBERT-base, "
                         "e.g. nlp-final-project11/nlp-project-modernbert-cmv_vote_pairs; the run is named "
                         "<config>-from-<model> for its output folder and Hub repo")
    ap.add_argument("--print-output-dir", action="store_true",
                    help="print the run's output folder and exit (jobs/train.sub keeps its DONE/FAILED markers there)")
    return ap.parse_args()

def run_name(args):
    """<config>, plus -pairwise with --pairwise and -from-<model> with --init-model, so such a run never
    resumes from or overwrites the plain run of the same config (locally or on the Hub)."""
    name = args.dataset_config + ("-pairwise" if args.pairwise else "") + ("-eval" if args.eval_only else "")
    if not args.init_model:
        return name
    model = os.path.basename(args.init_model.rstrip("/")).removeprefix(MODEL_REPO.split("/")[1] + "-")
    return f"{name}-from-{model}"

def binary_auroc(scores, labels):
    """Chance a random positive outscores a random negative (ties count half); nan if a class is absent."""
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if not pos or not neg:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # 1-based ranks, averaged over ties
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

def make_compute_metrics(pair_ids=None):
    """compute_metrics for the Trainer; with pair_ids (one per evaluated row, in order) it adds pairwise_acc."""
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
            if pair_ids is not None:
                metrics["pairwise_acc"] = pairwise_accuracy(logits[:, 1] - logits[:, 0], labels, pair_ids)
        return metrics
    return compute_metrics

def pairwise_accuracy(scores, labels, pair_ids, lengths=None):
    """T1: share of pairs whose label-1 reply (delta, higher) outscores the other one (ties count half).
    With lengths (words per row), only the pairs whose label-1 reply is the shorter one count: a model
    that just prefers longer replies gets those wrong, so it shows what the model reads past length."""
    pairs = defaultdict(dict)
    for i, (s, l, p) in enumerate(zip(scores, labels, pair_ids)):
        pairs[p][int(l)] = (s, lengths[i] if lengths is not None else 0)
    wins = [float(d[1][0] > d[0][0]) + 0.5 * float(d[1][0] == d[0][0]) for d in pairs.values()
            if len(d) == 2 and (lengths is None or d[1][1] < d[0][1])]
    return float(np.mean(wins)) if wins else float("nan")

def to_pairs(rows):
    """Rows of a pairs config (two per pair_id with the same original, labels 1 and 0) -> one example per
    pair: original, higher (the label-1 reply) and lower. A pair missing a row is left out."""
    by_pair = defaultdict(dict)
    for original, reply, label, pair_id in zip(rows["original"], rows["reply"], rows["labels"], rows["pair_id"]):
        by_pair[pair_id][label] = (original, reply)
    pairs = [d for d in by_pair.values() if len(d) == 2]
    return Dataset.from_dict({"original": [d[1][0] for d in pairs], "higher": [d[1][1] for d in pairs],
                              "lower": [d[0][1] for d in pairs]})

class PairCollator:
    """Pads a batch of pairs into one batch of sequences: the pairs' higher replies, then their lower ones.
    Evaluation batches are rows with labels (the Trainer uses one collator for both) and are padded as rows."""
    def __init__(self, tok):
        self.tok = tok
        self.rows = DataCollatorWithPadding(tokenizer=tok)

    def __call__(self, features):
        if "input_ids_hi" not in features[0]:
            return self.rows(features)
        rows = [{"input_ids": p[f"input_ids_{side}"], "attention_mask": p[f"attention_mask_{side}"]}
                for side in ("hi", "lo") for p in features]
        return self.tok.pad(rows, return_tensors="pt")

class PairwiseTrainer(Trainer):
    """Trains with a pairwise (Bradley-Terry) loss, -log sigmoid(score(higher) - score(lower)), as reward
    models are trained: only which reply of a pair scores higher matters, so a reply can win one pair and
    lose another. A reply's score is logit[1] - logit[0], the score the test ranks by. Evaluation batches
    are single rows with labels and get plain cross-entropy, so validation and test work as before."""
    loss_is_scaled_for_ga = False  # the pairwise loss is a per-batch mean: the Trainer divides it by the accumulation steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:  # evaluation rows
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = F.cross_entropy(outputs.logits.float(), inputs["labels"])
            return (loss, outputs) if return_outputs else loss
        outputs = model(**inputs)  # PairCollator: the first half are the higher replies, the second half the lower ones
        score = (outputs.logits[:, 1] - outputs.logits[:, 0]).float()
        higher, lower = score.chunk(2)
        loss = -F.logsigmoid(higher - lower).mean()
        return (loss, outputs) if return_outputs else loss

def weighted_cross_entropy(weights):
    """Class-weighted loss for Trainer(compute_loss_func=...). The Trainer expects such a loss to be
    normalized over the whole accumulated batch, so sum it and divide by that batch's label count.
    """
    def loss_fn(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, weight=weights.to(logits.device, logits.dtype), reduction="sum")
        return loss / (num_items_in_batch if num_items_in_batch is not None else labels.numel())
    return loss_fn

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
    if args.print_output_dir:
        print(f"outputs/{run_name(args)}")
        return
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))  # set by torchrun, one process per GPU
    if not args.smoke and EFFECTIVE_BATCH % (args.batch_size * n_gpus):
        raise SystemExit(f"--batch-size {args.batch_size} x {n_gpus} GPUs does not divide {EFFECTIVE_BATCH}")
    epochs     = 1   if args.smoke else args.epochs
    max_steps  = 4   if args.smoke else -1
    batch_size = 1   if args.smoke else args.batch_size
    grad_accum = 1   if args.smoke else EFFECTIVE_BATCH // (args.batch_size * n_gpus)
    max_length = 256 if args.smoke else MAX_LENGTH
    save_steps = 2   if args.smoke else SAVE_STEPS
    unit = "pairs" if args.pairwise else "rows"
    print(f"[BATCH] {batch_size} per GPU x {n_gpus} GPUs x {grad_accum} accumulation = {batch_size * n_gpus * grad_accum} {unit} per step")

    ds = load_dataset(DATASET_REPO, args.dataset_config, token=HF_TOKEN).rename_column("label", "labels")
    if args.smoke:
        ds = DatasetDict({k: v.select(range(32)) for k, v in ds.items()})

    label_names = ds["train"].features["labels"].names
    counts = np.bincount(list(ds["train"]["labels"]), minlength=len(label_names))
    # inverse-frequency weights, averaging 1 over the training set; all 1s for balanced configs
    class_weights = torch.tensor(counts.sum() / (len(label_names) * np.maximum(counts, 1)), dtype=torch.float32)
    print("[CLASSES] " + ", ".join(f"{n}={c} (weight {w:.2f})" for n, c, w in zip(label_names, counts, class_weights.tolist())))
    has_pairs = "pair_id" in ds["test"].column_names
    if args.pairwise and (not has_pairs or len(label_names) != 2):
        raise SystemExit(f"--pairwise needs a config with pair_id and two labels; {args.dataset_config} has neither or more labels")
    pair_ids = {k: v["pair_id"] if "pair_id" in v.column_names else None for k, v in ds.items()}
    test_words = [len(reply.split()) for reply in ds["test"]["reply"]]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tok.truncation_side = args.truncation_side

    def tokenize(batch):
        return tok(batch["original"], batch["reply"], truncation="longest_first", max_length=max_length)

    def tokenize_pairs(batch):
        higher = tok(batch["original"], batch["higher"], truncation="longest_first", max_length=max_length)
        lower = tok(batch["original"], batch["lower"], truncation="longest_first", max_length=max_length)
        return {"input_ids_hi": higher["input_ids"], "attention_mask_hi": higher["attention_mask"],
                "input_ids_lo": lower["input_ids"], "attention_mask_lo": lower["attention_mask"]}

    if args.pairwise:
        # training sees whole pairs; validation and test stay rows, scored one by one as in row training
        train_pairs = to_pairs(ds["train"])
        print(f"[PAIRWISE] {len(train_pairs)} training pairs from {len(ds['train'])} rows")
        train_set = train_pairs.map(tokenize_pairs, batched=True, remove_columns=["original", "higher", "lower"])
        ds = DatasetDict({k: v for k, v in ds.items() if k != "train"})
    ds = ds.map(tokenize, batched=True, remove_columns=["original", "reply"])
    if args.pairwise:
        # the Trainer keeps every column (the pairs' columns are not model inputs), so pass only model inputs
        ds = DatasetDict({k: v.select_columns([c for c in ("input_ids", "attention_mask", "labels") if c in v.column_names])
                          for k, v in ds.items()})
    else:
        train_set = ds["train"]

    # With --init-model the encoder and, when the label count matches, the classifier head carry over; the
    # head's class 1 then keeps its meaning (cmv_vote_pairs "higher" -> cmv "delta"). Otherwise the head is new.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_model or MODEL_NAME, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        ignore_mismatched_sizes=bool(args.init_model),
        token=HF_TOKEN,
    )
    if args.init_model:
        print(f"[INIT] starting from {args.init_model}")

    if args.freeze:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("[FREEZE] encoder frozen; training classifier head only")

    collator = DataCollatorWithPadding(tokenizer=tok)
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.bf16 and not use_bf16:
        print("[PRECISION] --bf16 requested but this GPU does not support it; using fp32")
    print(f"[PRECISION] {'bf16 mixed precision' if use_bf16 else 'fp32'}")
    # smoke runs get their own dir so a real run never resumes from a smoke checkpoint
    outputdir = f"outputs/{run_name(args)}" + ("-smoke" if args.smoke else "")
    model_repo = f"{MODEL_REPO}-{run_name(args)}"
    targs = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LR,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,  # the newest checkpoint, plus the best one when it is older: at most 2 (about 1.8 GB each)
        logging_steps=25,
        load_best_model_at_end=not args.smoke,
        # AUROC ranks without a threshold; positive-class F1 rewards predicting that class often,
        # so on balanced cmv it preferred checkpoints that call most replies "delta"
        metric_for_best_model="pairwise_acc" if args.pairwise else "auroc" if len(label_names) == 2 else "macro_f1",
        remove_unused_columns=not args.pairwise,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        push_to_hub=args.push and not args.smoke and not args.eval_only,
        hub_model_id=model_repo,
        hub_private_repo=True,  # only applies when the repo is created; change visibility on the Hub
        # no hub_token: it would be pickled into every checkpoint's training_args.bin and pushed
        # with the model; the Hub client reads HF_TOKEN from the environment instead
    )

    common = dict(
        model=model, args=targs, train_dataset=train_set, eval_dataset=ds["validation"],
        compute_metrics=make_compute_metrics(pair_ids["validation"]),
        # the best validation checkpoint is loaded at the end either way; this also stops training once
        # validation stops improving (smoke runs skip it: they keep no best checkpoint)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)]
                  if args.early_stopping and not args.smoke else None,
    )
    if args.pairwise:  # pairs are balanced by construction, so no class weights
        trainer = PairwiseTrainer(data_collator=PairCollator(tok), **common)
    else:
        trainer = Trainer(data_collator=collator, compute_loss_func=weighted_cross_entropy(class_weights), **common)

    if args.eval_only:
        print(f"[EVAL ONLY] scoring the {args.dataset_config} test split with {args.init_model or MODEL_NAME}, no training")
    else:
        # Continue a run cut off by a time limit; delete outputs/<run> to start over.
        resume = None if args.smoke else last_complete_checkpoint(outputdir)
        if resume:
            print(f"[RESUME] continuing from {resume}")
        trainer.train(resume_from_checkpoint=resume)
    if torch.cuda.is_available():  # peak over this process, to size future runs
        print(f"[VRAM] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")

    pred = trainer.predict(ds["test"])
    metrics = {k[5:]: v for k, v in pred.metrics.items() if k.startswith("test_")}
    if has_pairs:
        scores = pred.predictions[:, 1] - pred.predictions[:, 0]
        metrics["pairwise_acc"] = pairwise_accuracy(scores, pred.label_ids, pair_ids["test"])
        # pairs whose label-1 reply is the shorter one: below 0.5 means the model mostly ranks by length
        metrics["pairwise_acc_shorter_won"] = pairwise_accuracy(scores, pred.label_ids, pair_ids["test"], test_words)
    print("\n=== TEST METRICS ===")
    for name, v in metrics.items():
        print(f"  {name:10s}: {v:.4f}" if isinstance(v, float) else f"  {name}: {v}")

    if args.eval_only:
        return
    if args.push and not args.smoke:
        trainer.push_to_hub()
        print(f"\nPushed model -> https://huggingface.co/{model_repo} (private)")
    elif not args.smoke:
        trainer.save_model(outputdir)
        tok.save_pretrained(outputdir)
        print(f"\nSaved model locally -> {outputdir}/  (add --push to upload)")
    # The final (best) model is now in outputdir itself, and on the Hub with --push; the checkpoints
    # were only for resuming, so free their space (one GPU process deletes them)
    if not args.smoke and trainer.is_world_process_zero():
        for checkpoint in glob.glob(os.path.join(outputdir, "checkpoint-*")):
            shutil.rmtree(checkpoint)
        print(f"[CLEANUP] deleted the checkpoints in {outputdir}/")

if __name__ == "__main__":
    main()
