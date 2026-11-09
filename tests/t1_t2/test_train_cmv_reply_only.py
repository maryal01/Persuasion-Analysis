"""Synthetic unit tests for scripts/t1_t2/train_cmv_reply_only.py (no model download, no data).

Usage:
  python -m unittest tests/t1_t2/test_train_cmv_reply_only.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "t1_t2"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("train_cmv_reply_only", SCRIPTS / "train_cmv_reply_only.py")
ro = importlib.util.module_from_spec(spec)
sys.modules["train_cmv_reply_only"] = ro
spec.loader.exec_module(ro)


class RecordingTokenizer:
    """Stands in for the HF tokenizer and records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"input_ids": [[1, 2, 3]] * len(args[0])}


class TokenizeTest(unittest.TestCase):
    def test_reply_only_input(self):
        tok = RecordingTokenizer()
        fn = ro.make_reply_only_tokenize(tok, 4096)
        batch = {"original": ["POST A", "POST B"], "reply": ["reply a", "reply b"], "pair_id": [1, 1]}
        fn(batch)
        (args, kwargs), = tok.calls
        self.assertEqual(args, (["reply a", "reply b"],))  # one text argument: the replies, no post
        self.assertEqual(kwargs, {"truncation": "longest_first", "max_length": 4096})


class SplitCheckTest(unittest.TestCase):
    GOOD = {"train": ([1, 1, 2, 2], ["a", "a", "b", "b"], [1, 0, 0, 1]),
            "validation": ([3, 3], ["c", "c"], [0, 1]),
            "test": ([4, 4], ["d", "d"], [1, 0])}

    def test_good_splits(self):
        s = ro.check_splits(self.GOOD, delta_label=1)
        self.assertEqual(s["train"], {"rows": 4, "pairs": 2, "posts": 2})

    def test_post_leakage_detected(self):
        bad = dict(self.GOOD, test=([4, 4], ["a", "a"], [1, 0]))
        with self.assertRaises(ValueError):
            ro.check_splits(bad, delta_label=1)

    def test_pair_leakage_detected(self):
        bad = dict(self.GOOD, test=([1, 1], ["z", "z"], [1, 0]))
        with self.assertRaises(ValueError):
            ro.check_splits(bad, delta_label=1)

    def test_broken_pair_detected(self):
        bad = dict(self.GOOD, validation=([3, 3], ["c", "c"], [1, 1]))
        with self.assertRaises(ValueError):
            ro.check_splits(bad, delta_label=1)


class SmokeSelectionTest(unittest.TestCase):
    def test_whole_pairs_and_longest_included(self):
        pair_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        lengths = [5, 6, 900, 7, 8, 9, 10, 11, 12, 5000]
        rows = ro.select_smoke_rows(pair_ids, lengths, n_random=1, n_longest=2, seed=0)
        chosen = [pair_ids[r] for r in rows]
        self.assertEqual(len(rows), 6)
        self.assertTrue({1, 4} <= set(chosen))  # the two pairs with the longest replies
        for p in set(chosen):
            self.assertEqual(chosen.count(p), 2)  # whole pairs only
        self.assertEqual(rows, sorted(rows))  # dataset order kept


class CheckpointToggleTest(unittest.TestCase):
    def test_toggle_sets_every_layer(self):
        import torch
        from transformers.modeling_layers import GradientCheckpointingLayer

        class Layer(GradientCheckpointingLayer):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.lin(x)

        model = torch.nn.Sequential(Layer(), torch.nn.ReLU(), Layer())
        self.assertEqual(ro.set_checkpointing(model, True), 2)
        self.assertTrue(all(m.gradient_checkpointing for m in model if isinstance(m, Layer)))
        ro.set_checkpointing(model, False)
        self.assertFalse(any(m.gradient_checkpointing for m in model if isinstance(m, Layer)))


class ConfigTest(unittest.TestCase):
    def test_defaults_match_published_run(self):
        args = ro.parse_args(["train"])
        settings = {"max_length": args.max_length, "per_device_batch": args.per_device_batch,
                    "grad_accum": args.grad_accum, "epochs": args.epochs, "learning_rate": args.learning_rate,
                    "eval_save_steps": args.eval_steps, "select_metric": args.select_metric, "seed": args.seed,
                    "dataset_revision": args.dataset_revision}
        self.assertEqual(ro.deviations_from_baseline(settings), [])
        self.assertEqual(ro.BASELINE["per_device_batch"] * ro.BASELINE["grad_accum"], 32)
        self.assertEqual((ro.BASELINE["max_length"], ro.BASELINE["learning_rate"], ro.BASELINE["epochs"]), (4096, 2e-5, 3))
        self.assertEqual(ro.BASELINE["base_model"], "answerdotai/ModernBERT-base")

    def test_deviation_reported(self):
        self.assertEqual(ro.deviations_from_baseline({"select_metric": "auroc"}),
                         ["select_metric: auroc (published run: f1)"])


class StatsTest(unittest.TestCase):
    def test_bootstrap_indices_keep_posts_whole(self):
        posts = ["a", "a", "b", "c", "c", "c"]
        for idx in ro.cluster_bootstrap_indices(posts, n_boot=50, seed=0):
            counts = {p: sum(posts[i] == p for i in idx) for p in "abc"}
            self.assertEqual(counts["a"] % 2, 0)
            self.assertEqual(counts["c"] % 3, 0)

    def test_row_metrics_match_team_compute_metrics(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(200, 2))
        labels = rng.integers(0, 2, size=200)
        team = ro.td.compute_metrics((logits, labels))
        ours = ro.row_metrics(logits[:, 1] - logits[:, 0], labels)
        for k in ("accuracy", "precision", "recall", "f1", "auroc"):
            self.assertAlmostEqual(team[k], ours[k], places=12)

    def test_paired_difference_zero_for_identical_models(self):
        win = np.array([1, 0, 1, 1, 0.5, 0])
        boot = ro.cb.cluster_bootstrap_means(list("aabbcc"), {"diff": win - win}, n_boot=100, seed=0)
        self.assertTrue(np.allclose(boot["diff"], 0))


if __name__ == "__main__":
    unittest.main()
