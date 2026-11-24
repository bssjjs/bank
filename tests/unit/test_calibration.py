"""校准表与切片表的单测：数字要能手算核对，切片名额按比例分配。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from bank import schema as S
from bank_training import calibration as C
from bank_training import slices as SL


def _frame(values: dict[str, list[str]], n: int) -> pd.DataFrame:
    return pd.DataFrame(values)


class ReliabilityTest(unittest.TestCase):
    def test_table_counts_sum_to_n(self) -> None:
        y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype="int8")
        s = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.65, 0.4, 0.55, 0.05])
        table = C.reliability(y, s, bins=5)
        self.assertEqual(len(table), 5)
        self.assertEqual(int(sum(row["count"] for row in table)), 10)
        for row in table:
            self.assertAlmostEqual(row["gap"], abs(row["mean_score"] - row["observed_rate"]))

    def test_perfectly_calibrated_has_zero_ece(self) -> None:
        # 4 条分数都是 0.5，正好一半正例 → 平均分 = 实际正例率
        y = np.array([1, 0, 1, 0], dtype="int8")
        s = np.array([0.5, 0.5, 0.5, 0.5])
        summary = C.summary(y, s)
        self.assertAlmostEqual(summary["ece"], 0.0, places=6)
        self.assertAlmostEqual(summary["mean_score"], 0.5, places=6)
        self.assertAlmostEqual(summary["base_rate"], 0.5, places=6)

    def test_summary_reports_worst_bin(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int8")
        s = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        summary = C.summary(y, s)
        self.assertEqual(summary["n"], 8.0)
        self.assertGreater(summary["worst_gap"], 0.5)
        self.assertIn("[", summary["worst_bin"])

    def test_render_skips_empty_bins(self) -> None:
        y = np.array([1, 0], dtype="int8")
        s = np.array([0.95, 0.05])
        table = C.reliability(y, s, bins=4)
        lines = C.render(C.summary(y, s), table)
        body = [line for line in lines if line.startswith("| [")]
        self.assertEqual(len(body), 2)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            C.summary(np.array([1, 0]), np.array([0.5]))


class SliceTest(unittest.TestCase):
    def test_age_band_boundaries(self) -> None:
        frame = pd.DataFrame({S.AGE: [17, 18, 29, 30, 39, 40, 59, 60, 61]})
        bands = SL.age_band(frame).tolist()
        self.assertEqual(
            bands, ["18-29", "18-29", "18-29", "30-39", "30-39", "40-49", "50-59", "60+", "60+"]
        )

    def _synthetic(self, n_per_group: int = 120) -> pd.DataFrame:
        rows: dict[str, list[object]] = {
            S.AGE: [],
            S.OCCUPATION: [],
            S.EDU: [],
            S.CONTACT: [],
            S.MARITAL: [],
            S.TARGET: [],
        }
        for index in range(n_per_group * 3):
            rows[S.AGE].append(30 + index % 40)
            rows[S.OCCUPATION].append(["甲组", "乙组", "丙组"][index // n_per_group])
            rows[S.EDU].append("中学")
            rows[S.CONTACT].append("手机")
            rows[S.MARITAL].append("已婚")
            rows[S.TARGET].append(1 if index % 2 else 0)
        return pd.DataFrame(rows)

    def test_quota_is_proportional(self) -> None:
        frame = self._synthetic(n_per_group=120)
        score = np.linspace(0.99, 0.01, len(frame))
        rows = SL.slice_table(frame, score=score, quota_n=36, by=S.OCCUPATION)
        self.assertEqual(len(rows), 3)  # 甲组 / 乙组 / 第三组
        for row in rows:
            self.assertFalse(row["skipped"])
            self.assertEqual(row["n"], 120)
            self.assertEqual(row["quota"], 12)  # 120 / 360 × 36
            self.assertLessEqual(int(row["positives"]), 120)
            self.assertLessEqual(float(row["precision"]), 1.0)

    def test_small_slice_is_skipped(self) -> None:
        frame = self._synthetic(n_per_group=120)
        frame.loc[0, S.OCCUPATION] = "小组"  # 1 人
        score = np.linspace(0.99, 0.01, len(frame))
        rows = SL.slice_table(frame, score=score, quota_n=36, by=S.OCCUPATION)
        skipped = [row for row in rows if row["skipped"]]
        self.assertEqual([row["value"] for row in skipped], ["小组"])

    def test_default_tables_cover_all_dimensions(self) -> None:
        frame = self._synthetic(n_per_group=120)
        score = np.linspace(0.99, 0.01, len(frame))
        tables = SL.default_tables(frame, score=score, quota_n=36)
        self.assertEqual(set(tables), set(SL.DEFAULT_SLICES))
        rendered = SL.render(tables)
        self.assertTrue(any(line.startswith("### 按") for line in rendered))

    def test_missing_slice_column_raises(self) -> None:
        frame = pd.DataFrame({S.TARGET: [1, 0]})
        with self.assertRaises(KeyError):
            SL.slice_table(frame, score=np.array([0.5, 0.5]), quota_n=1, by=S.OCCUPATION)


if __name__ == "__main__":
    unittest.main()
