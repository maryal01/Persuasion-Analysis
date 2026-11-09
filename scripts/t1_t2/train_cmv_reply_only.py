"""T1 ablation: fine-tune ModernBERT-base on CMV replies only (no original post).

Research question: does including the original post help ModernBERT pick the delta reply? The
published T1 model (nlp-final-project11/nlp-project-modernbert-cmv) reads tok(original, reply).
This script trains a fresh answerdotai/ModernBERT-base on tok(reply) and changes nothing else
it can avoid changing: same pinned t1-t2 splits, labels, loss, optimizer defaults, learning rate,
effective batch, epochs, max length, eval/save interval and checkpoint-selection metric as the
published run (scripts/train_delta.py at commit bf53e36, transformers 5.20.1).

Stages (run in order):

  train     Fine-tune on train, evaluate on validation every --eval-steps, keep the best checkpoint
            by --select-metric, and save it to <out-dir>/final. Never touches the test split.
  evaluate  Score the test split once with the selected model, report pairwise accuracy / AUROC /
            row accuracy / F1 with 95% post-cluster bootstrap CIs, and the paired difference to the
            published original+reply model (cached scores from cmv_baseline.py).

Usage:
  python scripts/t1_t2/train_cmv_reply_only.py train --smoke         # tiny subset, 4 optimizer steps
  python scripts/t1_t2/train_cmv_reply_only.py evaluate --smoke
  caffeinate -i python scripts/t1_t2/train_cmv_reply_only.py train   # full run (hours on an M5)
  python scripts/t1_t2/train_cmv_reply_only.py evaluate

Memory on Apple MPS: fp32 at 4,096 tokens does not fit a batch of 4 without gradient checkpointing.
--checkpointing auto (default) turns it on only for batches padded beyond 1,024 tokens; it changes
memory and speed, not the computed gradients. Checkpoints go to git-ignored outputs/; nothing is
pushed. The Hub token is read from the environment by huggingface_hub and never printed.
"""

from __future__ import annotations

import os

# fail fast with an MPS out-of-memory error instead of swapping (set before importing torch)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_delta as td  # noqa: E402  (team metrics, loss, resume helper and hyperparameters)
import cmv_baseline as cb  # noqa: E402  (pinned revisions, pairing, bootstrap and scoring helpers)

# Published run (train_delta.py @ bf53e36, Gilbreth A100, transformers 5.20.1). The values below are
# read from train_delta.py where they still match; select_metric differs from today's code.
BASELINE = {
    "base_model": td.MODEL_NAME,
    "dataset_revision": cb.DATASET_REVISION,  # t1-t2 head; unchanged since before the published run
    "max_length": td.MAX_LENGTH,
    "per_device_batch": td.BATCH_SIZE,
    "grad_accum": td.GRAD_ACCUM,
    "epochs": td.EPOCHS,
    "learning_rate": td.LR,
    "eval_save_steps": td.SAVE_STEPS,
    "select_metric": "f1",  # published run; train_delta.py now uses "auroc" (changed after that run)
    "seed": 42,  # TrainingArguments default
    "precision": "fp32",
    "logging_steps": 25,
}
CHECKPOINT_THRESHOLD = 1024  # padded tokens; above this a batch of 4 needs gradient checkpointing on MPS
SMOKE = {"train_pairs": 28, "train_longest_pairs": 4, "eval_pairs": 8, "eval_longest_pairs": 2,
         "max_steps": 4, "grad_accum": 2, "eval_save_steps": 2}


# --------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------------

def make_reply_only_tokenize(tok, max_length: int) -> Callable[[dict], dict]:
    """The only intended change from train_delta.py: tok(reply) instead of tok(original, reply)."""
    def tokenize(batch: dict) -> dict:
        return tok(batch["reply"], truncation="longest_first", max_length=max_length)
    return tokenize


def check_splits(splits: dict[str, tuple[Sequence[int], Sequence[str], Sequence[int]]], delta_label: int) -> dict:
    """splits: name -> (pair_ids, post_ids, labels). Every pair must be complete within its split
    (one delta + one no-delta row, same post), and no post or pair may appear in two splits."""
    summary, posts, pairs = {}, {}, {}
    for name, (pair_ids, post_ids, labels) in splits.items():
        built = cb.build_pairs(pair_ids, post_ids, labels, delta_label=delta_label)  # raises if malformed
        posts[name], pairs[name] = set(post_ids), {p.pair_id for p in built}
        summary[name] = {"rows": len(labels), "pairs": len(built), "posts": len(posts[name])}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if posts[a] & posts[b] or pairs[a] & pairs[b]:
                raise ValueError(f"leakage between {a} and {b}: {len(posts[a] & posts[b])} posts, "
                                 f"{len(pairs[a] & pairs[b])} pairs shared")
    return summary


def select_smoke_rows(pair_ids: Sequence[int], lengths: Sequence[int], n_random: int, n_longest: int,
                      seed: int = 0) -> list[int]:
    """Rows of whole pairs: the n_longest pairs by longest reply (to exercise peak memory) plus
    n_random other pairs."""
    rows_by_pair: dict[int, list[int]] = {}
    for row, pid in enumerate(pair_ids):
        rows_by_pair.setdefault(int(pid), []).append(row)
    longest = sorted(rows_by_pair, key=lambda p: -max(lengths[r] for r in rows_by_pair[p]))[:n_longest]
    rest = sorted(set(rows_by_pair) - set(longest))
    chosen = longest + [int(p) for p in np.random.default_rng(seed).choice(rest, size=min(n_random, len(rest)), replace=False)]
    return sorted(r for p in chosen for r in rows_by_pair[p])


def set_checkpointing(model, enabled: bool) -> int:
    """Toggle gradient checkpointing on every GradientCheckpointingLayer (they check the flag on each
    call); returns how many layers were set. The model must have had gradient_checkpointing_enable()."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    n = 0
    for module in model.modules():
        if isinstance(module, GradientCheckpointingLayer):
            module.gradient_checkpointing = enabled
            n += 1
    return n


def deviations_from_baseline(settings: dict) -> list[str]:
    return [f"{k}: {settings[k]} (published run: {v})" for k, v in BASELINE.items() if k in settings and settings[k] != v]


def cluster_bootstrap_indices(post_ids: Sequence[str], n_boot: int, seed: int):
    """Yield row-index arrays for a post-cluster bootstrap (whole posts resampled with replacement)."""
    post_ids = np.asarray(post_ids)
    unique, inverse = np.unique(post_ids, return_inverse=True)
    rows_by_post = [np.where(inverse == i)[0] for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), size=len(unique))
        yield np.concatenate([rows_by_post[i] for i in picks])


def row_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Row accuracy, delta-class precision/recall/F1 (argmax = score > 0) and AUROC, as train_delta."""
    preds = (scores > 0).astype(int)
    tp, fp, fn = int(((preds == 1) & (labels == 1)).sum()), int(((preds == 1) & (labels == 0)).sum()), int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": float((preds == labels).mean()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "auroc": td.binary_auroc(scores, labels)}


class PeakMemoryMonitor:
    """Samples torch.mps.driver_allocated_memory() in a background thread (MPS has no peak counter)."""

    def __init__(self, interval: float = 0.05):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def __enter__(self):
        import torch
        if torch.backends.mps.is_available():
            def poll():
                while not self._stop.is_set():
                    self.peak = max(self.peak, torch.mps.driver_allocated_memory())
                    time.sleep(self.interval)
            threading.Thread(target=poll, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

def load_splits(revision: str):
    from datasets import load_dataset
    return load_dataset(cb.DATASET_REPO, cb.DATASET_CONFIG, revision=revision)


def apply_smoke(ds, tok, seed: int):
    from datasets import DatasetDict
    out = {}
    for split in ("train", "validation", "test"):
        lengths = [len(x) for x in tok(list(ds[split]["reply"]), add_special_tokens=False)["input_ids"]]
        n_rand, n_long = ((SMOKE["train_pairs"], SMOKE["train_longest_pairs"]) if split == "train"
                          else (SMOKE["eval_pairs"], SMOKE["eval_longest_pairs"]))
        out[split] = ds[split].select(select_smoke_rows(list(ds[split]["pair_id"]), lengths, n_rand, n_long, seed))
    return DatasetDict(out)


def hardware_info() -> dict:
    info = {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            line = line.strip()
            for key in ("Model Name", "Chip", "Memory", "Total Number of Cores"):
                if line.startswith(key + ":"):
                    info[key] = line.split(":", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


# --------------------------------------------------------------------------------------------
# Stage 1: train
# --------------------------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    out_dir = Path(args.out_dir) / ("smoke" if args.smoke else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "base_model": td.MODEL_NAME, "dataset_revision": args.dataset_revision, "max_length": args.max_length,
        "per_device_batch": args.per_device_batch,
        "grad_accum": SMOKE["grad_accum"] if args.smoke else args.grad_accum,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "eval_save_steps": SMOKE["eval_save_steps"] if args.smoke else args.eval_steps,
        "select_metric": args.select_metric, "seed": args.seed, "precision": "fp32", "logging_steps": 25,
        "checkpointing": args.checkpointing, "smoke": args.smoke,
        "max_steps": SMOKE["max_steps"] if args.smoke else -1,
    }
    config_path = out_dir / "run_config.json"
    resume = None if args.smoke else td.last_complete_checkpoint(str(out_dir))
    if resume:
        previous = json.loads(config_path.read_text())["settings"]
        if previous != settings:
            raise SystemExit(f"[STOP] {out_dir} holds checkpoints from different settings; use another --out-dir")
        print(f"[RESUME] continuing from {resume}")

    ds = load_splits(args.dataset_revision)
    label_names = ds["train"].features["label"].names
    delta_id = label_names.index("delta")
    split_summary = check_splits({s: (list(ds[s]["pair_id"]), list(ds[s]["post_id"]), list(ds[s]["label"]))
                                  for s in ("train", "validation", "test")}, delta_label=delta_id)
    print("[DATA] " + ", ".join(f"{s}: {v['rows']} rows / {v['pairs']} pairs / {v['posts']} posts"
                                for s, v in split_summary.items()) + "; no post or pair shared across splits")

    tok = AutoTokenizer.from_pretrained(td.MODEL_NAME)
    if args.smoke:
        ds = apply_smoke(ds, tok, args.seed)
        print("[SMOKE] " + ", ".join(f"{s}={len(ds[s])} rows" for s in ds) + " (whole pairs, incl. the longest replies)")
    ds = ds.rename_column("label", "labels")
    counts = np.bincount(list(ds["train"]["labels"]), minlength=len(label_names))
    class_weights = torch.tensor(counts.sum() / (len(label_names) * np.maximum(counts, 1)), dtype=torch.float32)
    print("[CLASSES] " + ", ".join(f"{n}={c} (weight {w:.2f})" for n, c, w in zip(label_names, counts, class_weights.tolist())))
    ds = ds.map(make_reply_only_tokenize(tok, args.max_length), batched=True, remove_columns=["original", "reply"])

    # Seed before creating the model so the new classifier head is reproducible. (The published run
    # created its model before Trainer seeded, so its head initialisation was not seeded.)
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        td.MODEL_NAME, num_labels=len(label_names), id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)}, dtype=torch.float32)
    threshold = {"auto": CHECKPOINT_THRESHOLD, "always": 0, "never": None}[args.checkpointing]
    if threshold is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[MODEL] {td.MODEL_NAME}, attention {model.config._attn_implementation}, fp32, "
          f"checkpointing {args.checkpointing}" + (f" (> {threshold} tokens)" if args.checkpointing == "auto" else ""))

    class ReplyOnlyTrainer(Trainer):
        checkpointed_batches = 0
        batches = 0

        def training_step(self, model, inputs, num_items_in_batch=None):
            if threshold is not None:
                use = inputs["input_ids"].shape[1] > threshold
                set_checkpointing(model, use)
                ReplyOnlyTrainer.checkpointed_batches += int(use)
            ReplyOnlyTrainer.batches += 1
            return super().training_step(model, inputs, num_items_in_batch)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=settings["per_device_batch"],
        per_device_eval_batch_size=settings["per_device_batch"],
        gradient_accumulation_steps=settings["grad_accum"],
        num_train_epochs=settings["epochs"],
        max_steps=settings["max_steps"],
        learning_rate=settings["learning_rate"],
        eval_strategy="steps", eval_steps=settings["eval_save_steps"],
        save_strategy="steps", save_steps=settings["eval_save_steps"],
        save_total_limit=None,  # keep every evaluated checkpoint (published run kept 2 + best; disk only)
        logging_steps=25 if not args.smoke else 1,
        load_best_model_at_end=True,
        metric_for_best_model=args.select_metric,
        seed=args.seed,
        report_to="none",
    )
    trainer = ReplyOnlyTrainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
                               data_collator=DataCollatorWithPadding(tokenizer=tok), compute_metrics=td.compute_metrics,
                               compute_loss_func=td.weighted_cross_entropy(class_weights))

    run_config = {
        "settings": settings, "deviations_from_published_run": deviations_from_baseline(settings),
        "fixed_differences": [
            "input: tok(reply) instead of tok(original, reply) (the ablation)",
            "hardware: Apple MPS instead of one A100-80GB (CUDA)",
            f"gradient checkpointing: {args.checkpointing} (memory only; same gradients up to floating-point order)",
            "classifier head initialised after set_seed(seed) (published run: unseeded)",
            "save_total_limit None: all evaluated checkpoints kept (published: 2 + best; disk only)",
            "attention: sdpa (published run: not recorded; sdpa unless flash-attn was installed)",
        ],
        "optimizer": {"optim": str(targs.optim), "weight_decay": targs.weight_decay, "adam_betas": [targs.adam_beta1, targs.adam_beta2],
                      "adam_epsilon": targs.adam_epsilon, "max_grad_norm": targs.max_grad_norm,
                      "lr_scheduler": str(targs.lr_scheduler_type), "warmup_steps": targs.warmup_steps},
        "class_weights": class_weights.tolist(), "splits": split_summary,
        "device": str(targs.device), "hardware": hardware_info(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "mps_watermarks": {k: os.environ.get(k) for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")},
    }
    config_path.write_text(json.dumps(run_config, indent=2))
    for line in run_config["deviations_from_published_run"]:
        print(f"[DEVIATION] {line}")

    start = time.time()
    with PeakMemoryMonitor() as mem:
        trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - start

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    evals = [{k: v for k, v in e.items() if k.startswith("eval_") or k in ("step", "epoch")}
             for e in trainer.state.log_history if "eval_loss" in e]
    summary = {
        "train_seconds_this_session": round(elapsed, 1), "resumed_from": resume,
        "global_steps": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric, "select_metric": args.select_metric,
        "peak_mps_driver_memory_gib": round(mem.peak / 2**30, 2),
        "batches": ReplyOnlyTrainer.batches, "checkpointed_batches": ReplyOnlyTrainer.checkpointed_batches,
        "validation_history": evals,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== VALIDATION HISTORY (model selection; the test split is not used here) ===")
    print(f"  {'step':>6} {'loss':>7} {'acc':>7} {'f1':>7} {'auroc':>7}")
    for e in evals:
        print(f"  {e['step']:>6} {e['eval_loss']:7.4f} {e['eval_accuracy']:7.4f} {e['eval_f1']:7.4f} {e['eval_auroc']:7.4f}")
    print(f"  selected: {summary['best_checkpoint']} by {args.select_metric} = {summary['best_metric']}")
    print(f"  {summary['global_steps']} optimizer steps in {elapsed / 60:.1f} min; peak MPS driver memory "
          f"{summary['peak_mps_driver_memory_gib']} GiB; checkpointing on for {summary['checkpointed_batches']}/{summary['batches']} batches")
    print(f"  saved selected model -> {final_dir}")
    return 0


# --------------------------------------------------------------------------------------------
# Stage 2: evaluate (test split, once)
# --------------------------------------------------------------------------------------------

def run_evaluate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    out_dir = Path(args.out_dir) / ("smoke" if args.smoke else "full")
    final_dir = out_dir / "final"
    metrics_path = out_dir / "test_metrics.json"
    if metrics_path.exists() and not args.allow_reevaluate:
        raise SystemExit(f"[STOP] {metrics_path} exists: the test split was already scored for this model. "
                         "Use --allow-reevaluate only to reproduce it, not to tune.")
    run_config = json.loads((out_dir / "run_config.json").read_text())
    settings = run_config["settings"]

    ds = load_splits(settings["dataset_revision"])
    tok = AutoTokenizer.from_pretrained(str(final_dir))
    if args.smoke:
        ds = apply_smoke(ds, AutoTokenizer.from_pretrained(td.MODEL_NAME), settings["seed"])
    test = ds["test"]
    pair_ids, post_ids, labels = list(test["pair_id"]), list(test["post_id"]), np.array(list(test["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(str(final_dir), dtype=torch.float32)
    delta_id = {k.lower(): int(v) for k, v in model.config.label2id.items()}["delta"]
    pairs = cb.build_pairs(pair_ids, post_ids, labels, delta_label=delta_id)
    tokenized = test.map(make_reply_only_tokenize(tok, settings["max_length"]), batched=True,
                         remove_columns=["original", "reply"]).rename_column("label", "labels")
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out_dir / "eval_tmp"), report_to="none",
                                                          per_device_eval_batch_size=settings["per_device_batch"]),
                      data_collator=DataCollatorWithPadding(tokenizer=tok), compute_metrics=td.compute_metrics)
    start = time.time()
    pred = trainer.predict(tokenized)
    elapsed = time.time() - start
    logits = np.asarray(pred.predictions)
    scores = logits[:, delta_id] - logits[:, 1 - delta_id]
    is_delta = (labels == delta_id).astype(int)

    team = {k[5:]: v for k, v in pred.metrics.items() if k.startswith("test_") and isinstance(v, float)}
    team["pairwise_acc"] = td.pairwise_accuracy(scores, is_delta, pair_ids)

    # Paired comparison with the published original+reply model on the same rows
    base_path = Path(args.baseline_dir) / "predictions.csv"
    with open(base_path) as fh:
        base_rows = {int(r["row"]): r for r in csv.DictReader(fh)}
    # smoke subsets keep dataset row order, so map back to full-split row indices by pair and label
    base_by_key = {(int(r["pair_id"]), int(r["label"])): float(r["score"]) for r in base_rows.values()}
    base_scores = np.array([base_by_key[(int(p), int(l))] for p, l in zip(pair_ids, labels)])

    d_rows = np.array([p.delta_row for p in pairs])
    n_rows = np.array([p.nodelta_row for p in pairs])
    pair_posts = np.array([p.post_id for p in pairs])
    new_win = cb.win_score(scores[d_rows], scores[n_rows])
    base_win = cb.win_score(base_scores[d_rows], base_scores[n_rows])
    boot = cb.cluster_bootstrap_means(pair_posts, {"new": new_win, "base": base_win, "diff": new_win - base_win},
                                       args.n_boot, args.seed)
    row_reps = {k: [] for k in ("accuracy", "f1", "auroc", "auroc_base", "auroc_diff")}
    for idx in cluster_bootstrap_indices(post_ids, args.n_boot_rows, args.seed):
        m = row_metrics(scores[idx], is_delta[idx])
        mb = td.binary_auroc(base_scores[idx], is_delta[idx])
        for k in ("accuracy", "f1", "auroc"):
            row_reps[k].append(m[k])
        row_reps["auroc_base"].append(mb)
        row_reps["auroc_diff"].append(m["auroc"] - mb)
    ci = lambda v: cb.percentile_ci(np.array(v)[~np.isnan(np.array(v))])
    own = row_metrics(scores, is_delta)
    both = {"both correct": int(((new_win == 1) & (base_win == 1)).sum()), "reply-only only": int(((new_win == 1) & (base_win == 0)).sum()),
            "original+reply only": int(((new_win == 0) & (base_win == 1)).sum()), "both wrong": int(((new_win == 0) & (base_win == 0)).sum())}
    result = {
        "smoke": args.smoke, "test_pairs": len(pairs), "test_rows": len(labels), "test_posts": len(set(post_ids)),
        "team_metrics": team,
        "pairwise_acc": {"value": float(new_win.mean()), "ci": cb.percentile_ci(boot["new"])},
        "auroc": {"value": own["auroc"], "ci": ci(row_reps["auroc"])},
        "row_accuracy": {"value": own["accuracy"], "ci": ci(row_reps["accuracy"])},
        "f1_delta": {"value": own["f1"], "ci": ci(row_reps["f1"]), "precision": own["precision"], "recall": own["recall"]},
        "share_predicted_delta": float((scores > 0).mean()),
        "baseline": {"model": cb.MODEL_REPO, "pairwise_acc": float(base_win.mean()), "pairwise_ci": cb.percentile_ci(boot["base"]),
                     "auroc": td.binary_auroc(base_scores, is_delta), "auroc_ci": ci(row_reps["auroc_base"])},
        "paired_difference_reply_only_minus_baseline": {
            "pairwise_acc": float(new_win.mean() - base_win.mean()), "pairwise_ci": cb.percentile_ci(boot["diff"]),
            "auroc": own["auroc"] - td.binary_auroc(base_scores, is_delta), "auroc_ci": ci(row_reps["auroc_diff"]),
            "pair_agreement": both},
        "n_boot_pairs": args.n_boot, "n_boot_rows": args.n_boot_rows, "seed": args.seed,
        "inference_seconds": round(elapsed, 1),
    }
    if abs(result["pairwise_acc"]["value"] - team["pairwise_acc"]) > 1e-12:
        raise AssertionError("per-pair recount disagrees with train_delta.pairwise_accuracy")
    metrics_path.write_text(json.dumps(result, indent=2))
    with open(out_dir / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "post_id", "label", "logit_nodelta", "logit_delta", "score", "baseline_score"])
        for i in range(len(labels)):
            w.writerow([pair_ids[i], post_ids[i], int(labels[i]), f"{logits[i, 1 - delta_id]:.6f}", f"{logits[i, delta_id]:.6f}",
                        f"{scores[i]:.6f}", f"{base_scores[i]:.6f}"])

    p, b, d = result["pairwise_acc"], result["baseline"], result["paired_difference_reply_only_minus_baseline"]
    print(f"\n=== TEST ({'SMOKE subset' if args.smoke else 'official held-out split'}: {len(pairs)} pairs, {len(set(post_ids))} posts) ===")
    print(f"  reply-only pairwise acc : {100 * p['value']:.2f}%  95% CI {cb.percentile_ci(boot['new'])}")
    print(f"  reply-only AUROC        : {result['auroc']['value']:.4f}  95% CI {result['auroc']['ci']}")
    print(f"  row accuracy / F1(delta): {result['row_accuracy']['value']:.4f} / {result['f1_delta']['value']:.4f}")
    print(f"  original+reply (cached) : pairwise {100 * b['pairwise_acc']:.2f}%, AUROC {b['auroc']:.4f}")
    print(f"  difference (reply-only − original+reply): pairwise {100 * d['pairwise_acc']:+.2f} pts, 95% CI "
          f"[{100 * d['pairwise_ci'][0]:+.2f}, {100 * d['pairwise_ci'][1]:+.2f}]; AUROC {d['auroc']:+.4f}")
    print(f"  pair agreement: {both}")
    print(f"  wrote {metrics_path} and {out_dir / 'test_predictions.csv'}")
    return 0


# --------------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train", help="fine-tune reply-only ModernBERT; validation-based selection")
    t.add_argument("--out-dir", default="outputs/cmv_reply_only", help="git-ignored; smoke/ and full/ inside (default: %(default)s)")
    t.add_argument("--smoke", action="store_true", help="tiny subset (whole pairs, incl. the longest), 4 optimizer steps")
    t.add_argument("--dataset-revision", default=cb.DATASET_REVISION, help="pinned t1-t2 commit (default: %(default)s)")
    t.add_argument("--max-length", type=int, default=BASELINE["max_length"])
    t.add_argument("--per-device-batch", type=int, default=BASELINE["per_device_batch"])
    t.add_argument("--grad-accum", type=int, default=BASELINE["grad_accum"])
    t.add_argument("--epochs", type=float, default=BASELINE["epochs"])
    t.add_argument("--learning-rate", type=float, default=BASELINE["learning_rate"])
    t.add_argument("--eval-steps", type=int, default=BASELINE["eval_save_steps"], help="eval + checkpoint interval")
    t.add_argument("--select-metric", default=BASELINE["select_metric"], choices=["f1", "auroc"],
                   help="validation metric for the kept checkpoint (published run: f1; current train_delta.py: auroc)")
    t.add_argument("--seed", type=int, default=BASELINE["seed"])
    t.add_argument("--checkpointing", default="auto", choices=["auto", "always", "never"],
                   help=f"gradient checkpointing: auto = only batches padded beyond {CHECKPOINT_THRESHOLD} tokens")

    e = sub.add_parser("evaluate", help="score the test split once and compare with the published model")
    e.add_argument("--out-dir", default="outputs/cmv_reply_only")
    e.add_argument("--smoke", action="store_true", help="evaluate the smoke model on the smoke test subset")
    e.add_argument("--baseline-dir", default="outputs/cmv_length", help="cached original+reply scores (read only)")
    e.add_argument("--n-boot", type=int, default=10000, help="post-cluster resamples for pairwise metrics")
    e.add_argument("--n-boot-rows", type=int, default=2000, help="post-cluster resamples for AUROC / accuracy / F1")
    e.add_argument("--seed", type=int, default=577, help="bootstrap seed")
    e.add_argument("--allow-reevaluate", action="store_true", help="re-score an already evaluated model (reproduction only)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_train(args) if args.stage == "train" else run_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
