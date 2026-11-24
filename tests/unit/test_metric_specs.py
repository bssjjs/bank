"""指标出口登记表：名字、中文名、方向、小数位只登记一次（讲义 C-L1a ②）。"""

from __future__ import annotations

import unittest

from bank_training import metric_specs as MS
from bank_training import metrics as M
from bank_training import report as R


class RegistryTest(unittest.TestCase):
    def test_every_metric_has_a_spec(self) -> None:
        MS.assert_complete(M.METRIC_NAMES)
        self.assertEqual(tuple(spec.name for spec in MS.METRIC_SPECS), M.METRIC_NAMES)

    def test_missing_or_extra_metric_is_reported(self) -> None:
        with self.assertRaises(ValueError) as caught:
            MS.assert_complete((*M.METRIC_NAMES, "quota_extra"))
        self.assertIn("quota_extra", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            MS.assert_complete(tuple(n for n in M.METRIC_NAMES if n != "ece"))
        self.assertIn("ece", str(caught.exception))


class DirectionTest(unittest.TestCase):
    def test_lower_is_better_metrics(self) -> None:
        for name in ("fp_at_quota", "fn_at_quota", "brier", "log_loss", "ece"):
            with self.subTest(name=name):
                self.assertEqual(MS.direction(name), "越低越好")

    def test_higher_is_better_metrics(self) -> None:
        for name in ("precision_at_quota", "recall_at_quota", "ap", "roc_auc", "tp_at_quota"):
            with self.subTest(name=name):
                self.assertEqual(MS.direction(name), "越高越好")

    def test_unknown_metric_names_the_known_ones(self) -> None:
        with self.assertRaises(KeyError) as caught:
            MS.spec("f1")  # 名字少了 _at_quota
        self.assertIn("没有登记", str(caught.exception))
        self.assertIn("f1_at_quota", str(caught.exception))


class FormatTest(unittest.TestCase):
    def test_counts_are_integers_ratios_have_four_digits(self) -> None:
        self.assertEqual(MS.format_value("tp_at_quota", 264.0), "264")
        self.assertEqual(MS.format_value("precision_at_quota", 0.741573), "0.7416")
        self.assertEqual(MS.format_value("brier", 0.2852345), "0.2852")

    def test_report_rows_come_from_the_registry(self) -> None:
        expected = tuple((s.name, s.label, f".{s.digits}f") for s in MS.METRIC_SPECS)
        self.assertEqual(R.METRIC_ROWS, expected)
        self.assertEqual(len(R.METRIC_ROWS), len(M.METRIC_NAMES))


if __name__ == "__main__":
    unittest.main()
