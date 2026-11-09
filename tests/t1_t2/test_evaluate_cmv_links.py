"""Synthetic unit tests for scripts/t1_t2/evaluate_cmv_links.py (no model, no data download).

Usage:
  python -m unittest tests/t1_t2/test_evaluate_cmv_links.py -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "t1_t2"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("evaluate_cmv_links", SCRIPTS / "evaluate_cmv_links.py")
lk = importlib.util.module_from_spec(spec)
sys.modules["evaluate_cmv_links"] = lk
spec.loader.exec_module(lk)


def urls(text):
    return [(s.kind, text[s.start:s.end]) for s in lk.find_url_spans(text)]


class UrlDetectionTest(unittest.TestCase):
    def test_bare_urls_and_trailing_punctuation(self):
        text = "See https://example.com/a?b=1&amp;c=2. Also (www.foo.org/x), and http://bar.net!"
        self.assertEqual(urls(text), [("bare", "https://example.com/a?b=1&amp;c=2"),
                                      ("bare", "www.foo.org/x"), ("bare", "http://bar.net")])

    def test_balanced_parentheses_inside_url(self):
        text = "per https://en.wikipedia.org/wiki/Mercury_(planet). Done"
        self.assertEqual(urls(text), [("bare", "https://en.wikipedia.org/wiki/Mercury_(planet)")])

    def test_markdown_link_is_one_span(self):
        text = "As [this study](https://doi.org/10.1/x_(y)) shows, and [ref](http://a.b \"title\")."
        self.assertEqual(urls(text), [("markdown", "[this study](https://doi.org/10.1/x_(y))"),
                                      ("markdown", "[ref](http://a.b \"title\")")])

    def test_not_detected(self):
        for text in ("see /r/changemyview", "[wiki](/r/changemyview/wiki)", "example.com is a site",
                     "just http:// alone", "no links here"):
            self.assertEqual(urls(text), [], text)

    def test_case_insensitive_scheme(self):
        self.assertEqual(urls("HTTPS://X.ORG/a"), [("bare", "HTTPS://X.ORG/a")])


class EditTest(unittest.TestCase):
    TEXT = "Read [the paper](https://x.org/p) and http://y.com/q, then [https://z.io](https://z.io) ok."

    def test_original_unchanged(self):
        self.assertEqual(lk.edit_urls(self.TEXT, "original"), self.TEXT)

    def test_placeholder(self):
        self.assertEqual(lk.edit_urls(self.TEXT, "placeholder"),
                         "Read [the paper](URL) and URL, then [URL](URL) ok.")

    def test_remove(self):
        self.assertEqual(lk.edit_urls(self.TEXT, "remove"), "Read the paper and , then  ok.")

    def test_edits_leave_no_url_and_only_change_spans(self):
        spans = lk.find_url_spans(self.TEXT)
        for v in lk.VARIANTS:
            edited = lk.edit_urls(self.TEXT, v)
            self.assertEqual(lk.find_url_spans(edited), [])
            self.assertTrue(lk.changes_within_spans(self.TEXT, edited, spans))

    def test_span_check_catches_outside_changes(self):
        spans = lk.find_url_spans(self.TEXT)
        tampered = lk.edit_urls(self.TEXT, "remove").replace("Read", "Skim")
        self.assertFalse(lk.changes_within_spans(self.TEXT, tampered, spans))

    def test_both_checks_reject_outside_changes(self):
        spans = lk.find_url_spans(self.TEXT)
        for v in lk.VARIANTS:
            edited = lk.edit_urls(self.TEXT, v)
            self.assertTrue(lk._gaps_preserved(self.TEXT, edited, spans))
            for tampered in (edited.replace("Read", "Skim"), edited.replace(" ok.", " OK."), edited.replace("then", "than"),
                             edited + "!", "X" + edited):
                self.assertFalse(lk._gaps_preserved(self.TEXT, tampered, spans), (v, tampered))
                self.assertFalse(lk._diff_within_spans(self.TEXT, tampered, spans), (v, tampered))

    def test_gap_check_rejects_new_text_inside_span(self):
        text = "a [b](http://c.d) e"
        spans = lk.find_url_spans(text)
        self.assertTrue(lk._gaps_preserved(text, "a b e", spans))
        self.assertFalse(lk._gaps_preserved(text, "a NEW e", spans))  # not drawn from the span

    def test_gap_check_adjacent_spans(self):
        text = "x http://a.b[c](http://d.e) y"
        spans = lk.find_url_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertTrue(lk._gaps_preserved(text, lk.edit_urls(text, "remove"), spans))
        self.assertTrue(lk._gaps_preserved(text, lk.edit_urls(text, "placeholder"), spans))

    def test_link_at_end_of_text(self):
        # the shape of the case that tripped the diff check: markdown link as the last characters
        text = "Intro text here. [A source with words](https://example.org/some/long/path_(x)?q=1)."
        spans = lk.find_url_spans(text)
        for v in lk.VARIANTS:
            self.assertTrue(lk.changes_within_spans(text, lk.edit_urls(text, v), spans))

    def test_no_url_text_is_unchanged(self):
        self.assertEqual(lk.edit_urls("plain text", "remove"), "plain text")

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            lk.edit_urls(self.TEXT, "mask")


class CategoryAndStatsTest(unittest.TestCase):
    def test_link_category(self):
        self.assertEqual(lk.link_category(False, False), "neither")
        self.assertEqual(lk.link_category(True, True), "both")
        self.assertEqual(lk.link_category(True, False), "delta only")
        self.assertEqual(lk.link_category(False, True), "no-delta only")

    def test_group_mean_replicates_point_and_nan(self):
        clusters = ["a", "a", "b", "c"]
        values = np.array([1.0, 3.0, 5.0, 100.0])
        member = np.array([True, True, True, False])
        reps = lk.group_mean_replicates(clusters, values, member, n_boot=500, seed=0)
        finite = reps[~np.isnan(reps)]
        self.assertTrue(np.all((finite >= 1 - 1e-9) & (finite <= 5 + 1e-9)))  # never includes the non-member value
        self.assertTrue(np.isnan(reps).any())  # replicates drawing only cluster c have no members
        lo, hi = lk.nan_ci(reps)
        self.assertLessEqual(lo, 3.0)
        self.assertGreaterEqual(hi, 3.0)

    def test_paired_change_bootstrap_zero_when_no_change(self):
        clusters = list("aabbcc")
        before = np.array([1, 0, 1, 1, 0, 0.5])
        boot = lk.ecl.cluster_bootstrap_means(clusters, {"chg": before - before}, n_boot=200, seed=0)
        self.assertTrue(np.allclose(boot["chg"], 0))

    def test_pair_flip_logic(self):
        # pair correct before (delta 1.0 > nodelta 0.5); removing delta's URL lowers it to 0.2 -> flip
        before = lk.ecl.win_score([1.0], [0.5])
        after = lk.ecl.win_score([0.2], [0.5])
        self.assertEqual((before[0], after[0]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
