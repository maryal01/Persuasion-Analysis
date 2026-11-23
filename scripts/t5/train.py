"""T5 (CMV malleability): fine-tune ModernBERT on the post text, optionally together with the author's LIWC/Receptiviti numbers
or the OP's earlier CMV writing.

Modes:
    post          ModernBERT reads the cleaned post (title + body)
    post_liwc     the same, and the 211 author numbers go through a small network whose output is joined with the text
                  representation before the classifier (late fusion); only --config cmv_malleability_liwc
    post_history  the cleaned post, then the history items from the newest stored one backwards, cut at --max-length (the
                  earliest items are dropped); only the history configs (cmv_thread_history, cmv_thread_own_history)
    post_history_chunks  the post and each of the newest --history-items history items are encoded apart by the same ModernBERT
                  (an item cut at --item-length tokens); the item vectors are pooled (--history-pooling mean or attention, a learned
                  weight per item) and joined with the post vector; a post without history gets a learned "no history" vector

Class-weighted cross entropy, AdamW, the validation AUROC is computed after every epoch and the best epoch is kept; the test
split is scored once at the end (with a 95% bootstrap interval over authors). The data is read from the Hub
(scripts/t5/t5_common.py), the same data and splits as baselines.py.

Usage (from the repository root; GPU recommended; needs a Hugging Face login or HF_TOKEN):
    python scripts/t5/train.py --mode post_liwc --output-dir <out_dir>/t5_models --bf16
    python scripts/t5/train.py --mode post --smoke      # pipeline check on 16 posts
    python scripts/t5/train.py --config cmv_thread_history --mode post_history --max-length 8192 --batch-size 2 --grad-accum 16 --bf16

Files in --output-dir: best_model_<run>.pt (best validation epoch, written whenever it improves), last_<run>.pt (model, optimizer
and scheduler after every epoch; running the same command again resumes from it, and it is deleted after the test scoring unless
--keep-last) and result_<run>.json. --stop-after-epoch K ends the process after epoch K (for time-limited jobs).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from t5_common import (HISTORY_CONFIGS, HUB_CONFIG, HUB_REVISION, Split, T5Data, bootstrap_by_author, fit_scaler, load_history_data,
                       load_t5_data, metrics)

MODEL_NAME = "answerdotai/ModernBERT-base"
Batch = Dict[str, torch.Tensor]  # input_ids, attention_mask, features, labels, positions (+ item_ids, item_mask, item_owner)


class MalleabilityModel(nn.Module):
    """ModernBERT with mean pooling over the tokens. post_liwc: the output of a small network over the author numbers is joined to the
    text vector. post_history_chunks: the pooled vector of the history items is joined to the post vector."""

    def __init__(self, n_features: int, use_features: bool, history_pooling: Optional[str] = None, feature_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.use_features, self.history_pooling = use_features, history_pooling
        self.feature_branch = nn.Sequential(nn.Linear(n_features, feature_dim), nn.GELU(), nn.Dropout(dropout)) if use_features else None
        if history_pooling == "attention":
            self.item_scorer = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
        if history_pooling is not None:
            self.no_history = nn.Parameter(torch.zeros(hidden))
        width = hidden + (feature_dim if use_features else 0) + (hidden if history_pooling else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean of the token vectors of each text, (texts, hidden), in float32 (a bf16 token count above 256 would be rounded)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def pool_history(self, items: torch.Tensor, owner: torch.Tensor, n_posts: int) -> torch.Tensor:
        """One vector per post from the vectors of its items (items: (n_items, hidden); owner: the post of each item)."""
        counts = torch.bincount(owner, minlength=n_posts).unsqueeze(-1).float()
        if self.history_pooling == "attention":
            scores = self.item_scorer(items).squeeze(-1).float()  # float32 softmax (the layer runs in bf16 under autocast)
            highest = torch.full((n_posts,), -1e9, device=items.device).scatter_reduce(0, owner, scores.detach(), "amax")
            weights = torch.exp(scores - highest[owner])  # softmax over the items of each post
            totals = torch.zeros(n_posts, device=items.device).index_add(0, owner, weights)
            items = items * (weights / totals[owner]).unsqueeze(-1)
            counts = counts.clamp(max=1)  # the weights of a post already sum to 1
        pooled = torch.zeros(n_posts, items.size(-1), device=items.device).index_add(0, owner, items) / counts.clamp(min=1)
        return torch.where(counts > 0, pooled, self.no_history.float())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor, item_ids: Optional[torch.Tensor] = None,
                item_mask: Optional[torch.Tensor] = None, item_owner: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits (batch, 2). input_ids and attention_mask: (batch, tokens); features: (batch, n_features); item_ids and item_mask:
        (history items of the batch, tokens); item_owner: (history items,) the post of each item."""
        pooled = self.encode(input_ids, attention_mask)
        if self.history_pooling is not None:
            items = self.encode(item_ids, item_mask) if len(item_owner) else pooled.new_zeros((0, pooled.size(-1)))
            pooled = torch.cat([pooled, self.pool_history(items, item_owner, len(pooled))], dim=-1)
        if self.use_features:
            pooled = torch.cat([pooled, self.feature_branch(features.float())], dim=-1)
        return self.classifier(pooled)


# --- data ---------------------------------------------------------------------------------------------------------------

def prepare_data(args: argparse.Namespace) -> T5Data:
    """The Hub data (author features standardized with training statistics); in smoke mode only 16 posts per split."""
    if args.config in HISTORY_CONFIGS:
        history_format = {"post": "none", "post_history": "text", "post_history_chunks": "items"}[args.mode]
        data = load_history_data(args.config, history_format, args.hub_revision)
    else:
        data = load_t5_data(args.hub_revision)
        scale = fit_scaler(data.train.features, data.feature_names)
        for split in data.splits:
            split.features = scale(split.features).astype(np.float32)
    for split in data.splits:
        split.features = split.features.astype(np.float32)
    if args.smoke:
        data.train, data.validation, data.test = (split.first(16) for split in data.splits)
    return data


def length_grouped_batches(lengths: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator,
                           pool_batches: int = 50) -> List[List[int]]:
    """Batches of posts with similar token counts, so little padding is computed. Training: the posts are shuffled, cut into pools of
    pool_batches batches, each pool is sorted by length and cut into batches, and the order of all batches is shuffled (the scheme of
    the Hugging Face length-grouped sampler). Scoring: the posts are sorted by length."""
    if not shuffle:
        order = np.argsort(lengths, kind="stable")
        return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    order, pool = rng.permutation(len(lengths)), batch_size * pool_batches
    batches = []
    for start in range(0, len(order), pool):
        members = order[start:start + pool]
        members = members[np.argsort(-lengths[members], kind="stable")]
        batches += [members[i:i + batch_size].tolist() for i in range(0, len(members), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


class LengthGroupedSampler(torch.utils.data.Sampler):
    """A batch sampler that draws new length-grouped batches in every epoch."""

    def __init__(self, lengths: np.ndarray, batch_size: int, shuffle: bool, seed: int):
        self.lengths, self.batch_size, self.shuffle = lengths, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.n_batches = len(length_grouped_batches(lengths, batch_size, False, self.rng))

    def __iter__(self):
        return iter(length_grouped_batches(self.lengths, self.batch_size, self.shuffle, self.rng))

    def __len__(self) -> int:
        return self.n_batches


def make_loader(split: Split, tokenizer: Any, args: argparse.Namespace, batch_size: int, shuffle: bool) -> DataLoader:
    """A loader that tokenizes each batch (padded to the longest text of the batch). Each batch carries the positions of its posts in
    the split, so scores can be put back in split order when the batches are grouped by length. In chunk mode the newest
    args.history_items items of each post (the end of the stored order) are tokenized apart, all items of a batch in one tensor."""
    items = split.history_items or [[] for _ in split.texts]
    items = [post_items[-args.history_items:] for post_items in items]
    examples = list(zip(range(len(split)), split.texts, split.features, split.labels, items))

    def collate(batch: List[Tuple[int, str, np.ndarray, int, List[str]]]) -> Batch:
        positions, texts, features, labels, post_items = zip(*batch)
        encoded = tokenizer(list(texts), truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        result = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "positions": torch.tensor(positions),
                  "features": torch.tensor(np.stack(features), dtype=torch.float32), "labels": torch.tensor(labels)}
        if split.history_items is not None:
            flat = [item for one_post in post_items for item in one_post]
            owners = [post for post, one_post in enumerate(post_items) for _ in one_post]
            encoded_items = tokenizer(flat or [""], truncation=True, max_length=args.item_length, padding=True, return_tensors="pt")
            result.update(item_ids=encoded_items["input_ids"], item_mask=encoded_items["attention_mask"],
                          item_owner=torch.tensor(owners, dtype=torch.long))
        return result

    if not args.group_by_length:
        return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    lengths = np.array([len(ids) for ids in tokenizer(split.texts, truncation=True, max_length=args.max_length)["input_ids"]])
    if split.history_items is not None:  # the tokens of the items count too
        lengths += np.array([sum(len(ids) for ids in tokenizer(one_post, truncation=True, max_length=args.item_length)["input_ids"]) if one_post else 0
                             for one_post in items])
    return DataLoader(examples, batch_sampler=LengthGroupedSampler(lengths, batch_size, shuffle, args.seed), collate_fn=collate)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for the two classes (they average 1 over the training posts)."""
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)


# --- training -----------------------------------------------------------------------------------------------------------

class Trainer:
    """The model with its optimizer, learning-rate schedule, loaders and checkpoints."""

    def __init__(self, args: argparse.Namespace, data: T5Data, device: torch.device, bf16: bool):
        self.args, self.data, self.device, self.bf16 = args, data, device, bf16
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.loaders = {"train": make_loader(data.train, tokenizer, args, args.batch_size, True),
                        "validation": make_loader(data.validation, tokenizer, args, 2 * args.batch_size, False),
                        "test": make_loader(data.test, tokenizer, args, 2 * args.batch_size, False)}
        pooling = args.history_pooling if args.mode == "post_history_chunks" else None
        self.model = MalleabilityModel(len(data.feature_names), args.mode == "post_liwc", pooling).to(device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights(data.train.labels, device))
        self.optimizer, self.scheduler = self.build_optimizer()
        self.n_batches = len(self.loaders["train"])
        self.last_group = self.n_batches % args.grad_accum or args.grad_accum  # batches in the incomplete group that ends an epoch
        self.history: List[Dict[str, float]] = []
        self.best_auroc = -1.0
        self.start_epoch = 1
        run = (f"{args.mode}_seed{args.seed}" if args.config == HUB_CONFIG else f"{args.config}_{args.mode}_len{args.max_length}_seed{args.seed}")
        if args.mode == "post_history_chunks":
            run = f"{args.config}_{args.mode}_{args.history_pooling}_{args.history_items}x{args.item_length}_seed{args.seed}"
        run += "_grouped" if args.group_by_length else ""
        run += "_smoke" if args.smoke else ""
        self.best_path = args.output_dir / f"best_model_{run}.pt"
        self.last_path = args.output_dir / f"last_{run}.pt"
        self.result_path = args.output_dir / f"result_{run}.json"

    def build_optimizer(self) -> Tuple[torch.optim.Optimizer, Any]:
        """AdamW with a small learning rate for ModernBERT and a larger one for the new layers, and a one-cycle schedule."""
        args = self.args
        new_layers = [parameter for name, parameter in self.model.named_parameters() if not name.startswith("encoder.")]
        optimizer = torch.optim.AdamW([{"params": self.model.encoder.parameters(), "lr": args.lr}, {"params": new_layers, "lr": args.head_lr}],
                                      weight_decay=0.01)
        n_batches = len(self.loaders["train"])
        steps = args.epochs * math.ceil(n_batches / args.grad_accum)  # one optimizer step per grad_accum batches, the last group may be shorter
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                        pct_start=0.1, anneal_strategy="linear")
        return optimizer, scheduler

    def forward(self, batch: Batch) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            inputs = {name: batch[name].to(self.device) for name in ("input_ids", "attention_mask", "features", "item_ids", "item_mask", "item_owner")
                      if name in batch}
            return self.model(**inputs)

    def train_epoch(self, epoch: int) -> float:
        """One pass over the training posts; returns the mean loss per batch."""
        self.model.train()
        grad_accum = self.args.grad_accum
        loss_sum, started = 0.0, time.time()
        for step, batch in enumerate(self.loaders["train"], 1):
            group_size = grad_accum if step <= self.n_batches - self.last_group else self.last_group  # batches that share this optimizer step
            loss = self.loss_fn(self.forward(batch).float(), batch["labels"].to(self.device)) / group_size
            loss.backward()
            loss_sum += loss.item() * group_size
            if step % 50 == 0:
                spent = time.time() - started
                print(f"[STEP] epoch {epoch} batch {step}/{self.n_batches} mean loss {loss_sum / step:.4f} elapsed {spent:.0f}s, "
                      f"about {spent * (self.n_batches - step) / step:.0f}s left in this epoch")
            if step % grad_accum == 0 or step == self.n_batches:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        return loss_sum / self.n_batches

    @torch.no_grad()
    def predict(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """The probability of a delta and the label of every post of a split ("validation" or "test"), in split order."""
        self.model.eval()
        scores, labels, positions = [], [], []
        for batch in self.loaders[name]:
            scores.append(torch.softmax(self.forward(batch).float(), dim=-1)[:, 1].cpu().numpy())
            labels.append(batch["labels"].numpy())
            positions.append(batch["positions"].numpy())
        order = np.argsort(np.concatenate(positions))
        return np.concatenate(scores)[order], np.concatenate(labels)[order]

    # --- checkpoints ---

    def signature(self) -> Dict[str, Any]:
        """What identifies a run; a resume checkpoint is only used by a run with the same signature."""
        args = self.args
        return {"config": args.config, "mode": args.mode, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "head_lr": args.head_lr, "max_length": args.max_length,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum, "bf16": self.bf16, "n_train": len(self.data.train),
                "group_by_length": args.group_by_length, "history_input": [args.history_pooling, args.history_items, args.item_length]}

    def save_last(self, epoch: int) -> None:
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "history": self.history, "best_auroc": self.best_auroc, "signature": self.signature()}, self.last_path)

    def resume(self) -> None:
        """Continue after the last finished epoch if a checkpoint of the same run exists."""
        if not self.last_path.exists():
            return
        checkpoint = torch.load(self.last_path, map_location="cpu")
        if checkpoint["signature"] != self.signature():
            raise SystemExit(f"{self.last_path} comes from a different run ({checkpoint['signature']}); delete it or use another --output-dir")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.history, self.best_auroc, self.start_epoch = checkpoint["history"], checkpoint["best_auroc"], checkpoint["epoch"] + 1
        print(f"[RESUME] {self.last_path.name}: epochs 1-{checkpoint['epoch']} done, continuing at epoch {self.start_epoch}")

    # --- the run ---

    def finish_epoch(self, epoch: int, train_loss: float, started: float) -> None:
        """Score the validation split, keep the model if it is the best so far, and write the resume checkpoint."""
        validation = metrics(*self.predict("validation"))
        self.history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{name}": value for name, value in validation.items()}})
        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f} val_auroc={validation['auroc']:.4f} val_pr_auc={validation['pr_auc']:.4f} "
              f"val_macro_f1={validation['macro_f1']:.3f} elapsed={time.time() - started:.0f}s")
        if validation["auroc"] > self.best_auroc:
            self.best_auroc = validation["auroc"]
            torch.save(self.model.state_dict(), self.best_path)
            print(f"[SAVE] best model so far (val AUROC {self.best_auroc:.4f}) -> {self.best_path}")
        self.save_last(epoch)

    def fit(self) -> bool:
        """Train for the remaining epochs; False if --stop-after-epoch ended the process early."""
        args, started = self.args, time.time()
        for epoch in range(self.start_epoch, args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.finish_epoch(epoch, train_loss, started)
            if args.stop_after_epoch is not None and args.stop_after_epoch <= epoch < args.epochs:
                print(f"[STOP] stopped after epoch {epoch}; run the same command again to continue")
                return False
        return True

    def score_test(self) -> Dict[str, Any]:
        """Load the best epoch and score the test split."""
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        scores, labels = self.predict("test")
        result = {"config": self.args.config, "mode": self.args.mode, "hub": {"revision": self.args.hub_revision, "commit": self.data.hub_commit}, "seed": self.args.seed,
                  "bf16": self.bf16, "group_by_length": self.args.group_by_length,
                  "history_input": [self.args.history_pooling, self.args.history_items, self.args.item_length], "max_length": self.args.max_length, "lr": self.args.lr, "head_lr": self.args.head_lr,
                  "n": {"train": len(self.data.train), "validation": len(self.data.validation), "test": len(self.data.test)},
                  "history": self.history, "best_val_auroc": self.best_auroc, "best_model": str(self.best_path), "test": metrics(scores, labels),
                  "test_ci95_by_op": None if self.args.smoke else bootstrap_by_author(scores, labels, self.data.test.authors)}
        self.result_path.write_text(json.dumps(result, indent=1))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=[HUB_CONFIG, *HISTORY_CONFIGS], default=HUB_CONFIG, help="Hub config (default: %(default)s)")
    parser.add_argument("--mode", choices=["post", "post_liwc", "post_history", "post_history_chunks"], required=True)
    parser.add_argument("--history-pooling", choices=["mean", "attention"], default="mean", help="post_history_chunks: how the item vectors are pooled")
    parser.add_argument("--history-items", type=int, default=32, help="post_history_chunks: newest history items used per post")
    parser.add_argument("--item-length", type=int, default=256, help="post_history_chunks: tokens per history item")
    parser.add_argument("--hub-revision", default=HUB_REVISION, help="branch or commit of the Hub dataset (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/t5"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None, help="default 3 (1 with --smoke)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate of the pretrained encoder")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new, randomly initialized layers (classifier and, in post_liwc, "
                        "the feature network); not tuned, set it equal to --lr for a single learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision (default: fp32)")
    parser.add_argument("--group-by-length", action="store_true", help="batches of posts with similar token counts (less padding)")
    parser.add_argument("--smoke", action="store_true", help="16 posts, 128 tokens, batch 4, grad-accum 3")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="stop the process after this epoch (the last checkpoint stays for the next run)")
    parser.add_argument("--keep-last", action="store_true", help="keep the resume checkpoint after the test scoring")
    args = parser.parse_args()
    args.epochs = args.epochs or (1 if args.smoke else 3)
    if args.mode == "post_liwc" and args.config != HUB_CONFIG:
        parser.error(f"post_liwc needs --config {HUB_CONFIG}")
    if args.mode in ("post_history", "post_history_chunks") and args.config not in HISTORY_CONFIGS:
        parser.error(f"post_history needs one of {HISTORY_CONFIGS}")
    if args.smoke:
        args.max_length, args.batch_size, args.grad_accum = 128, 4, 3
    return args


def main() -> None:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when the output is redirected to a file
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[SETUP] device={device} precision={'bf16' if bf16 else 'fp32'} mode={args.mode} seed={args.seed}")

    data = prepare_data(args)
    print(f"[DATA] Hub commit {data.hub_commit[:8]}; train/val/test = {len(data.train)}/{len(data.validation)}/{len(data.test)}; "
          f"train delta rate {data.train.labels.mean():.3f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(args, data, device, bf16)
    trainer.resume()
    if not trainer.fit():
        return
    if device.type == "cuda":
        print(f"[VRAM] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    result = trainer.score_test()
    if not args.keep_last:
        trainer.last_path.unlink()
    print("=== TEST (best validation epoch) ===\n" + json.dumps(result["test"], indent=1), "\nCI95 (OP-level bootstrap):", result["test_ci95_by_op"])


if __name__ == "__main__":
    main()
