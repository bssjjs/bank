"""PSI 的口径测试：分箱边界取自基准、缺失单独一箱、类别新值合并。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd
from bank import drift


class PsiNumericTest(unittest.TestCase):
    def test_same_distribution_is_zero(self) -> None:
        ref = pd.Series(np.linspace(0, 100, 500), name="x")
        result = drift.psi_numeric(ref, ref.copy())
        self.assertLess(result.psi, 1e-9)
        self.assertEqual(result.level, "稳定")

    def test_shifted_distribution_is_severe(self) -> None:
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 2000), name="x")
        cur = pd.Series(rng.normal(3, 1, 2000), name="x")
        self.assertGreater(drift.psi_numeric(ref, cur).psi, 0.25)

    def test_missing_gets_its_own_bin(self) -> None:
        ref = pd.Series(np.linspace(0, 10, 200), name="x")
        cur = ref.copy()
        cur.iloc[:100] = np.nan
        result = drift.psi_numeric(ref, cur)
        self.assertTrue(result.detail["missing_bin"])
        self.assertTrue(math.isfinite(result.psi))
        self.assertGreater(result.psi, 0.25)

    def test_constant_reference_does_not_crash(self) -> None:
        ref = pd.Series([5.0] * 50, name="x")
        result = drift.psi_numeric(ref, ref.copy())
        self.assertEqual(len(result.detail["edges"]), 2)

    def test_all_missing_reference_raises(self) -> None:
        ref = pd.Series([np.nan] * 10, name="x")
        with self.assertRaises(ValueError):
            drift.psi_numeric(ref, ref.copy())


class PsiCategoricalTest(unittest.TestCase):
    def test_same_distribution_is_zero(self) -> None:
        ref = pd.Series(["甲"] * 60 + ["乙"] * 40, name="c")
        self.assertLess(drift.psi_categorical(ref, ref.copy()).psi, 1e-9)

    def test_unseen_category_is_reported(self) -> None:
        ref = pd.Series(["甲"] * 90 + ["乙"] * 10, name="c")
        cur = pd.Series(["甲"] * 50 + ["丙"] * 50, name="c")
        result = drift.psi_categorical(ref, cur)
        self.assertGreater(result.psi, 0.25)
        self.assertAlmostEqual(result.detail["unseen_share"], 0.5, places=6)


class LevelAndReportTest(unittest.TestCase):
    def test_level_thresholds(self) -> None:
        self.assertEqual(drift.level_of(0.05), "稳定")
        self.assertEqual(drift.level_of(0.15), "关注")
        self.assertEqual(drift.level_of(0.9), "严重")

    def test_report_sorts_desc_and_skips_missing(self) -> None:
        ref = pd.DataFrame({"a": np.arange(100.0), "b": ["x"] * 100, "c": np.arange(100.0)})
        cur = pd.DataFrame({"a": np.arange(100.0) + 50, "b": ["y"] * 100})
        rows = drift.report(ref, cur, ["a", "b", "c"], numeric_columns={"a"})
        self.assertEqual([row.feature for row in rows], ["b", "a"])
        self.assertTrue(all(rows[0].psi >= rows[1].psi for _ in [0]))

    def test_score_psi_is_zero_for_same_scores(self) -> None:
        scores = np.linspace(0, 1, 300)
        self.assertLess(drift.score_psi(scores, scores.copy()).psi, 1e-9)

    def test_label_rate_shift_in_percent(self) -> None:
        self.assertAlmostEqual(drift.label_rate_shift(0.1, 0.2), 100.0, places=6)
        self.assertTrue(math.isinf(drift.label_rate_shift(0.0, 0.2)))


if __name__ == "__main__":
    unittest.main()
