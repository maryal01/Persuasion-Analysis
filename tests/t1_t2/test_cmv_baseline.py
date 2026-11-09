"""Synthetic unit tests for scripts/t1_t2/cmv_baseline.py (no model, no data download).

Usage:
  python -m unittest tests/t1_t2/test_cmv_baseline.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "t1_t2" / "cmv_baseline.py"
spec = importlib.util.spec_from_file_location("cmv_baseline", SCRIPT)
cb = importlib.util.module_from_spec(spec)
sys.modules["cmv_baseline"] = cb
spec.loader.exec_module(cb)


class BuildPairsTest(unittest.TestCase):
    def test_pairs_matched_by_pair_id_not_row_order(self):
        # rows deliberately interleaved: pair 7 = rows 0 and 3, pair 2 = rows 1 and 2
        pairs = cb.build_pairs(pair_ids=[7, 2, 2, 7], post_ids=["a", "b", "b", "a"], labels=[0, 1, 0, 1], delta_label=1)
        self.assertEqual(pairs, [cb.Pair(2, "b", 1, 2), cb.Pair(7, "a", 3, 0)])

    def test_delta_label_taken_from_argument(self):
        pairs = cb.build_pairs([1, 1], ["a", "a"], [0, 1], delta_label=0)
        self.assertEqual(pairs[0].delta_row, 0)

    def test_rejects_malformed_pairs(self):
        with self.assertRaises(ValueError):  # two delta rows
            cb.build_pairs([1, 1], ["a", "a"], [1, 1], delta_label=1)
        with self.assertRaises(ValueError):  # singleton
            cb.build_pairs([1, 1, 2], ["a", "a", "b"], [1, 0, 1], delta_label=1)
        with self.assertRaises(ValueError):  # rows from different posts
            cb.build_pairs([1, 1], ["a", "b"], [1, 0], delta_label=1)


class PairQuantitiesTest(unittest.TestCase):
    def test_win_score_counts_ties_half(self):
        np.testing.assert_array_equal(cb.win_score([2, 1, 3], [1, 2, 3]), [1.0, 0.0, 0.5])

    def test_length_ratio(self):
        np.testing.assert_allclose(cb.length_ratio([80, 100, 0, 5], [100, 80, 0, 0]), [0.8, 0.8, 1.0, 0.0])

    def test_matches_team_pairwise_accuracy(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=40)
        scores[10] = scores[11]  # one tied pair
        labels = np.tile([1, 0], 20)
        pair_ids = np.repeat(np.arange(20), 2)
        team = cb.pairwise_accuracy(scores, labels, pair_ids)
        ours = cb.win_score(scores[0::2], scores[1::2]).mean()
        self.assertAlmostEqual(team, ours)

    def test_text_features(self):
        f = cb.text_features("&gt; you said X\nsee https://a.b and [c](www.d.e)\n> again")
        self.assertEqual(f["n_quote_lines"], 2)
        self.assertEqual(f["n_links"], 3)  # https://, www., ](
        self.assertEqual(f["reply_words"], 10)

    def test_prediction_fields_unchanged(self):
        # cached predictions.csv files are read by column name; keep the schema stable
        self.assertEqual(cb.PREDICTION_FIELDS[:7], ["row", "pair_id", "post_id", "label", "logit_nodelta", "logit_delta", "score"])


class BootstrapTest(unittest.TestCase):
    def test_cluster_bootstrap_degenerate_cases(self):
        boot = cb.cluster_bootstrap_means(["a", "a", "b"], {"x": np.ones(3)}, n_boot=200, seed=0)
        self.assertTrue(np.allclose(boot["x"], 1.0))
        boot = cb.cluster_bootstrap_means(["a"] * 4, {"x": np.array([1, 0, 1, 0])}, n_boot=200, seed=0)
        self.assertTrue(np.allclose(boot["x"], 0.5))  # one cluster: no between-cluster variance

    def test_cluster_bootstrap_keeps_clusters_together(self):
        # post a: 10 correct pairs, post b: 10 wrong pairs -> replicate means are k/2 for k in {0,1,2}
        boot = cb.cluster_bootstrap_means(["a"] * 10 + ["b"] * 10, {"x": np.r_[np.ones(10), np.zeros(10)]},
                                          n_boot=500, seed=0)
        self.assertTrue(set(np.round(boot["x"], 6)) <= {0.0, 0.5, 1.0})

    def test_bootstrap_difference_equals_difference_of_means(self):
        clusters = list("aabbc")
        m, b = np.array([1, 0, 1, 1, 0.5]), np.array([0, 0, 1, 0, 1])
        boot = cb.cluster_bootstrap_means(clusters, {"m": m, "b": b, "d": m - b}, n_boot=300, seed=3)
        np.testing.assert_allclose(boot["d"], boot["m"] - boot["b"])

    def test_percentile_ci(self):
        lo, hi = cb.percentile_ci(np.arange(1001) / 1000)
        self.assertAlmostEqual(lo, 0.025)
        self.assertAlmostEqual(hi, 0.975)


class StatsTest(unittest.TestCase):
    def test_spearman_monotone(self):
        self.assertAlmostEqual(cb.spearman(np.arange(10), np.arange(10) ** 3), 1.0)
        self.assertAlmostEqual(cb.spearman(np.arange(10), -np.arange(10)), -1.0)


if __name__ == "__main__":
    unittest.main()
