"""T5 (CMV malleability): fine-tune ModernBERT on the post text, optionally fused with the author's LIWC/Receptiviti features.

Modes:
    post        ModernBERT on "title + selftext" only
    post_liwc   the same encoder; its mean-pooled output is concatenated with a small MLP over the standardized
                author features (late fusion) before the classifier

Class-weighted cross entropy, AdamW, validation AUROC after every epoch, best epoch kept, test scored once
at the end (with a 95% bootstrap interval). Data comes from scripts/t5/build_dataset.py and is loaded by t5_common.py, the same code as the baselines (same splits, same posts).

Usage (from the repository root, GPU recommended):
    python scripts/t5/train.py --data-dir <out_dir>/t5_dataset \
        --mode post_liwc --output-dir outputs/t5 --bf16
    python scripts/t5/train.py --data-dir ... --mode post --smoke      # tiny pipeline check (16 posts)
--data-dir must be a dataset built with the DeltaBot label (the script stops otherwise).  All posts are used; the post text is cleaned by t5_common.clean_text() (moderator footnote, the OP's later Edit:/Update: paragraphs and delta symbols removed).

Checkpoints in --output-dir: best_model_<mode>_seed<N>.pt (best validation epoch, written whenever it improves) and last_<...>.pt (model, optimizer,
scheduler after every epoch; training resumes from it when the same command is run again, and it is deleted after the test scoring unless --keep-last).
--stop-after-epoch K ends the process after epoch K (for time-limited jobs); run the same command again to continue.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from t5_common import bootstrap_by_group, check_label, load_rows, metrics, standardizer

MODEL_NAME = "answerdotai/ModernBERT-base"
Rows = List[Dict[str, Any]]  # rows of build_t5_dataset.py: {"text", "label", "features", "created_utc", ...}


class MalleabilityModel(nn.Module):
    """ModernBERT encoder with mean pooling; optional author-feature branch concatenated before the classifier."""

    def __init__(self, n_features: int, use_features: bool, feature_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.use_features = use_features
        self.feature_branch = nn.Sequential(nn.Linear(n_features, feature_dim), nn.GELU(), nn.Dropout(dropout)) if use_features else None
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + (feature_dim if use_features else 0), 2))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Logits (batch, 2). input_ids/attention_mask: (batch, seq); features: (batch, n_features)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden, mask = hidden.float(), attention_mask.unsqueeze(-1).float()  # float32 pooling: a bf16 token count above 256 would be rounded
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.use_features:
            pooled = torch.cat([pooled, self.feature_branch(features.float())], dim=-1)
        return self.classifier(pooled)


def make_loader(rows: Rows, tokenizer: Any, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    """DataLoader that tokenizes with dynamic padding per batch."""
    def collate(batch: Rows) -> Dict[str, torch.Tensor]:
        enc = tokenizer([r["text"] for r in batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "features": torch.tensor([r["features"] for r in batch], dtype=torch.float32),
                "labels": torch.tensor([r["label"] for r in batch])}
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model: MalleabilityModel, loader: DataLoader, device: torch.device, bf16: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Probability of the delta class and the labels for a whole loader."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["features"].to(device))
        scores.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="output of build_dataset.py")
    parser.add_argument("--mode", choices=["post", "post_liwc"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/t5"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None, help="default 3 (1 with --smoke)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate of the pretrained encoder")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new, randomly initialized layers (classifier and, in post_liwc, the feature MLP); "
                        "not tuned, set it equal to --lr for a single learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision (default: fp32)")
    parser.add_argument("--smoke", action="store_true", help="16 posts, 128 tokens, batch 4, grad-accum 3")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="stop the process after this epoch (the last checkpoint stays for the next run)")
    parser.add_argument("--keep-last", action="store_true", help="keep the resume checkpoint after the test scoring")
    args = parser.parse_args()
    args.epochs = args.epochs or (1 if args.smoke else 3)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[SETUP] device={device} precision={'bf16' if bf16 else 'fp32'} mode={args.mode} seed={args.seed}")

    names = json.loads((args.data_dir / "feature_names.json").read_text())
    train, val, test = (load_rows(args.data_dir / f"{s}.jsonl") for s in ("train", "validation", "test"))
    check_label(train, args.data_dir)
    transform = standardizer(np.array([r["features"] for r in train]), names)
    for rows in (train, val, test):
        for r, f in zip(rows, transform(np.array([r["features"] for r in rows]))):
            r["features"] = f.astype(np.float32).tolist()
    if args.smoke:
        train, val, test = train[:16], val[:16], test[:16]
        args.max_length, args.batch_size, args.grad_accum = 128, 4, 3
    labels_train = np.array([r["label"] for r in train])
    counts = np.bincount(labels_train, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    print(f"[DATA] {args.data_dir.name}, DeltaBot label; train/val/test = {len(train)}/{len(val)}/{len(test)}; train delta rate {labels_train.mean():.3f}; class weights {weights.tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loaders = {"train": make_loader(train, tokenizer, args.max_length, args.batch_size, True),
               "val": make_loader(val, tokenizer, args.max_length, 2 * args.batch_size, False),
               "test": make_loader(test, tokenizer, args.max_length, 2 * args.batch_size, False)}
    model = MalleabilityModel(len(names), args.mode == "post_liwc").to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    n_batches = len(loaders["train"])
    # one optimizer step per grad_accum batches, plus one for a last incomplete group: ceil per epoch
    steps_total = args.epochs * math.ceil(n_batches / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, args.head_lr], total_steps=steps_total, pct_start=0.1, anneal_strategy="linear")
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    last_group = n_batches % args.grad_accum or args.grad_accum  # size of the incomplete group at the end of an epoch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_seed{args.seed}{'_smoke' if args.smoke else ''}"
    best_path, last_path = args.output_dir / f"best_model_{tag}.pt", args.output_dir / f"last_{tag}.pt"
    signature = {"mode": args.mode, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "head_lr": args.head_lr, "max_length": args.max_length,
                 "batch_size": args.batch_size, "grad_accum": args.grad_accum, "bf16": bf16, "n_train": len(train)}
    history: List[Dict[str, float]] = []
    best_auroc, start_epoch = -1.0, 1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu")
        if checkpoint["signature"] != signature:
            raise SystemExit(f"{last_path} comes from a different run ({checkpoint['signature']}); delete it or use another --output-dir")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        history, best_auroc, start_epoch = checkpoint["history"], checkpoint["best_auroc"], checkpoint["epoch"] + 1
        print(f"[RESUME] {last_path.name}: epochs 1-{checkpoint['epoch']} done, continuing at epoch {start_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            group = args.grad_accum if step <= n_batches - last_group else last_group  # batches that share this optimizer step
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["features"].to(device))
            loss = loss_fn(logits.float(), batch["labels"].to(device)) / group
            loss.backward()
            running += loss.item() * group
            if step % args.grad_accum == 0 or step == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        val_scores, val_labels = predict(model, loaders["val"], device, bf16)
        val_metrics = metrics(val_scores, val_labels)
        history.append({"epoch": epoch, "train_loss": running / n_batches, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"[EPOCH {epoch}] train_loss={history[-1]['train_loss']:.4f} val_auroc={val_metrics['auroc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} val_macro_f1={val_metrics['macro_f1']:.3f} elapsed={time.time() - t_start:.0f}s")
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] best model so far (val AUROC {best_auroc:.4f}) -> {best_path}")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "best_auroc": best_auroc, "signature": signature}, last_path)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch and epoch < args.epochs:
            print(f"[STOP] stopped after epoch {epoch}; run the same command again to continue")
            return
    if device.type == "cuda":
        print(f"[VRAM] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_scores, test_labels = predict(model, loaders["test"], device, bf16)
    result = {"mode": args.mode, "data_dir": str(args.data_dir), "label": "deltabot", "seed": args.seed, "bf16": bf16, "max_length": args.max_length, "lr": args.lr, "head_lr": args.head_lr,
              "n": {"train": len(train), "validation": len(val), "test": len(test)}, "history": history, "best_val_auroc": best_auroc, "best_model": str(best_path),
              "test": metrics(test_scores, test_labels),
              "test_ci95_by_op": None if args.smoke else bootstrap_by_group(test_scores, test_labels, [r["author"] for r in test])}
    (args.output_dir / f"result_{tag}.json").write_text(json.dumps(result, indent=1))
    if not args.keep_last:
        last_path.unlink()
    print("=== TEST (best validation epoch) ===\n" + json.dumps(result["test"], indent=1), "\nCI95 (OP-level bootstrap):", result["test_ci95_by_op"])


if __name__ == "__main__":
    main()
