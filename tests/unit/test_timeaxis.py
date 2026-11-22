"""时间轴重建的单测：M2 的时间切分全部依赖它，规则必须钉死。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from bank import schema as S
from bank import timeaxis as T


class BuildTimeAxisTest(unittest.TestCase):
    def test_basic_month_progression(self) -> None:
        out = T.build_time_axis(np.array([5, 5, 6, 7]), np.array([5, 6, 1, 3]))
        self.assertEqual(
            list(out[T.DATE].dt.strftime("%Y-%m-%d")),
            ["2008-05-05", "2008-05-06", "2008-06-01", "2008-07-03"],
        )
        self.assertEqual(list(out[T.YEAR]), [2008, 2008, 2008, 2008])
        self.assertEqual(list(out[T.MONTH_REBUILT]), [5, 5, 6, 7])

    def test_year_rollover_december_to_january(self) -> None:
        # 首月必须等于 start_month，所以这里显式指定 12 月起始。
        out = T.build_time_axis(np.array([12, 12, 1]), np.array([30, 31, 2]), start_month=12)
        self.assertEqual(
            list(out[T.DATE].dt.strftime("%Y-%m-%d")),
            ["2008-12-30", "2008-12-31", "2009-01-02"],
        )
        self.assertEqual(list(out[T.YEAR]), [2008, 2008, 2009])

    def test_missing_month_gap_is_forward_jump(self) -> None:
        # 2008-09 整月缺失：08 → 10 是前进两个月，不是回退。
        out = T.build_time_axis(np.array([8, 10]), np.array([30, 1]), start_month=8)
        self.assertEqual(list(out[T.DATE].dt.strftime("%Y-%m-%d")), ["2008-08-30", "2008-10-01"])

    def test_first_month_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            T.build_time_axis(np.array([6, 7]), np.array([1, 1]))

    def test_day_backtrack_in_same_month_raises(self) -> None:
        with self.assertRaises(ValueError):
            T.build_time_axis(np.array([5, 5]), np.array([10, 9]))

    def test_month_backtrack_other_than_rollover_raises(self) -> None:
        # 6 → 5 如果按 (cur-prev)%12 算会变成前进 11 个月，日期仍然单调，错误被掩盖。
        with self.assertRaises(ValueError):
            T.build_time_axis(np.array([6, 5]), np.array([1, 1]))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            T.build_time_axis(np.array([5, 6]), np.array([1]))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            T.build_time_axis(np.array([]), np.array([]))

    def test_custom_start_year(self) -> None:
        out = T.build_time_axis(np.array([5]), np.array([5]), start_year=2015)
        self.assertEqual(out[T.DATE].iloc[0].strftime("%Y-%m-%d"), "2015-05-05")

    def test_month_key(self) -> None:
        dates = pd.to_datetime(["2008-12-31", "2009-01-02"])
        self.assertEqual(list(T.month_key(pd.Series(dates))), ["2008-12", "2009-01"])


class RealDataTimeAxisTest(unittest.TestCase):
    """拿真实库跑一遍：数据卡的结论必须能被断言复现，而不只是写在文档里。"""

    @classmethod
    def setUpClass(cls) -> None:
        from bank import profile

        cls.df = profile.with_time_axis(profile.load_ordered())

    def test_shape_and_label(self) -> None:
        self.assertEqual(len(self.df), 45_211)
        self.assertEqual(int((self.df[S.TARGET_RAW] == S.POSITIVE).sum()), 5_289)

    def test_date_range_and_monotonic(self) -> None:
        dates = self.df[T.DATE]
        self.assertEqual(str(dates.min().date()), "2008-05-05")
        self.assertEqual(str(dates.max().date()), "2010-11-17")
        self.assertEqual(int(dates.nunique()), 561)
        self.assertTrue(bool(dates.is_monotonic_increasing))

    def test_month_blocks_and_missing_month(self) -> None:
        dates = self.df[T.DATE]
        months = sorted(T.month_key(dates).unique())
        self.assertEqual(len(months), 30)
        self.assertNotIn("2008-09", months)
        self.assertEqual(months[0], "2008-05")
        self.assertEqual(months[-1], "2010-11")

    def test_rowid_sequence_preserved(self) -> None:
        # 行号必须从 1 连续到底：它是时间序的载体，断了时间轴就不可信。
        seq = self.df[T.SEQUENCE]
        self.assertEqual(int(seq.iloc[0]), 1)
        self.assertEqual(int(seq.iloc[-1]), 45_211)
        self.assertTrue(bool(seq.is_monotonic_increasing))


if __name__ == "__main__":
    unittest.main()
