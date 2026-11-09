"""Synthetic unit tests for scripts/t1_t2/evaluate_cmv_length.py (no model, no data download).

Usage:
  python -m unittest tests/t1_t2/test_evaluate_cmv_length.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "t1_t2" / "evaluate_cmv_length.py"
spec = importlib.util.spec_from_file_location("evaluate_cmv_length", SCRIPT)
ev = importlib.util.module_from_spec(spec)
sys.modules["evaluate_cmv_length"] = ev
spec.loader.exec_module(ev)


class BuildPairsTest(unittest.TestCase):
    def test_pairs_matched_by_pair_id_not_row_order(self):
        # rows deliberately interleaved: pair 7 = rows 0 and 3, pair 2 = rows 1 and 2
        pairs = ev.build_pairs(pair_ids=[7, 2, 2, 7], post_ids=["a", "b", "b", "a"], labels=[0, 1, 0, 1], delta_label=1)
        self.assertEqual(pairs, [ev.Pair(2, "b", 1, 2), ev.Pair(7, "a", 3, 0)])

    def test_delta_label_taken_from_argument(self):
        pairs = ev.build_pairs([1, 1], ["a", "a"], [0, 1], delta_label=0)
        self.assertEqual(pairs[0].delta_row, 0)

    def test_rejects_malformed_pairs(self):
        with self.assertRaises(ValueError):  # two delta rows
            ev.build_pairs([1, 1], ["a", "a"], [1, 1], delta_label=1)
        with self.assertRaises(ValueError):  # singleton
            ev.build_pairs([1, 1, 2], ["a", "a", "b"], [1, 0, 1], delta_label=1)
        with self.assertRaises(ValueError):  # rows from different posts
            ev.build_pairs([1, 1], ["a", "b"], [1, 0], delta_label=1)


class PairQuantitiesTest(unittest.TestCase):
    def test_win_score_counts_ties_half(self):
        np.testing.assert_array_equal(ev.win_score([2, 1, 3], [1, 2, 3]), [1.0, 0.0, 0.5])

    def test_length_ratio(self):
        np.testing.assert_allclose(ev.length_ratio([80, 100, 0, 5], [100, 80, 0, 0]), [0.8, 0.8, 1.0, 0.0])

    def test_threshold_is_inclusive_at_boundary(self):
        ratio = ev.length_ratio([4, 9, 19], [5, 10, 20])  # exactly 0.80, 0.90, 0.95
        self.assertTrue(ev.SUBSETS["ratio >= 0.80"](ratio).all())
        self.assertEqual(ev.SUBSETS["ratio >= 0.90"](ratio).tolist(), [False, True, True])
        self.assertEqual(ev.SUBSETS["ratio >= 0.95"](ratio).tolist(), [False, False, True])
        self.assertFalse(ev.SUBSETS["ratio < 0.80"](ratio).any())

    def test_matches_team_pairwise_accuracy(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=40)
        scores[10] = scores[11]  # one tied pair
        labels = np.tile([1, 0], 20)
        pair_ids = np.repeat(np.arange(20), 2)
        team = ev.pairwise_accuracy(scores, labels, pair_ids)
        ours = ev.win_score(scores[0::2], scores[1::2]).mean()
        self.assertAlmostEqual(team, ours)

    def test_text_features(self):
        f = ev.text_features("&gt; you said X\nsee https://a.b and [c](www.d.e)\n> again")
        self.assertEqual(f["n_quote_lines"], 2)
        self.assertEqual(f["n_links"], 3)  # https://, www., ](
        self.assertEqual(f["reply_words"], 10)


class IntervalsTest(unittest.TestCase):
    def test_wilson_known_value(self):
        lo, hi = ev.wilson_interval(0.5, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_cluster_bootstrap_degenerate_cases(self):
        boot = ev.cluster_bootstrap_means(["a", "a", "b"], {"x": np.ones(3)}, n_boot=200, seed=0)
        self.assertTrue(np.allclose(boot["x"], 1.0))
        boot = ev.cluster_bootstrap_means(["a"] * 4, {"x": np.array([1, 0, 1, 0])}, n_boot=200, seed=0)
        self.assertTrue(np.allclose(boot["x"], 0.5))  # one cluster: no between-cluster variance

    def test_cluster_bootstrap_keeps_clusters_together(self):
        # post a: 10 correct pairs, post b: 10 wrong pairs -> replicate means are k/2 for k in {0,1,2}
        boot = ev.cluster_bootstrap_means(["a"] * 10 + ["b"] * 10, {"x": np.r_[np.ones(10), np.zeros(10)]},
                                          n_boot=500, seed=0)
        self.assertTrue(set(np.round(boot["x"], 6)) <= {0.0, 0.5, 1.0})

    def test_cluster_ci_wider_than_iid_when_clustered(self):
        rng = np.random.default_rng(0)
        post_effect = rng.random(50) < 0.6
        clusters = np.repeat(np.arange(50).astype(str), 10)
        x = np.repeat(post_effect, 10).astype(float)  # fully correlated within post
        boot = ev.cluster_bootstrap_means(clusters, {"x": x}, n_boot=2000, seed=0)
        lo, hi = ev.percentile_ci(boot["x"])
        wlo, whi = ev.wilson_interval(x.mean(), len(x))
        self.assertGreater(hi - lo, 1.5 * (whi - wlo))

    def test_bootstrap_difference_equals_difference_of_means(self):
        clusters = list("aabbc")
        m, b = np.array([1, 0, 1, 1, 0.5]), np.array([0, 0, 1, 0, 1])
        boot = ev.cluster_bootstrap_means(clusters, {"m": m, "b": b, "d": m - b}, n_boot=300, seed=3)
        np.testing.assert_allclose(boot["d"], boot["m"] - boot["b"])


class StatsTest(unittest.TestCase):
    def test_spearman_monotone(self):
        self.assertAlmostEqual(ev.spearman(np.arange(10), np.arange(10) ** 3), 1.0)
        self.assertAlmostEqual(ev.spearman(np.arange(10), -np.arange(10)), -1.0)

    def test_logistic_recovers_slope(self):
        rng = np.random.default_rng(0)
        x = rng.random(20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-(-1 + 2 * x)))).astype(float)
        b0, b1 = ev.weighted_logistic(x, y)
        self.assertAlmostEqual(b0, -1, delta=0.1)
        self.assertAlmostEqual(b1, 2, delta=0.15)

    def test_logistic_weights_equal_duplication(self):
        x, y = np.array([0.1, 0.5, 0.9, 0.3]), np.array([0, 1, 1, 0.5])
        w = np.array([2, 1, 3, 1])
        np.testing.assert_allclose(ev.weighted_logistic(x, y, w), ev.weighted_logistic(np.repeat(x, w), np.repeat(y, w)),
                                   atol=1e-8)


if __name__ == "__main__":
    unittest.main()
