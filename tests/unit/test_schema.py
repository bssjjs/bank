"""口径规则单测：python -m unittest discover -s tests -v

用标准库 unittest，不额外引入 pytest 依赖。
重点覆盖边界值——口径类 bug 几乎都出在边界上（30 到底进哪档、-1 算不算联系过）。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from bank import dataset as D
from bank import schema as S

from tests.support import synthetic_raw_frame


class TestPdays(unittest.TestCase):
    def test_contacted_flag(self):
        got = S.pdays_contacted(pd.Series([-1, 1, 30, 871])).tolist()
        self.assertEqual(got, [0, 1, 1, 1])
        self.assertEqual(S.pdays_contacted(pd.Series([-1])).iloc[0], 0)

    def test_log_zero_only_for_never(self):
        got = S.pdays_log(pd.Series([-1, 1, 871])).tolist()
        self.assertEqual(got[0], 0.0)
        self.assertAlmostEqual(got[1], np.log1p(1), places=12)
        self.assertAlmostEqual(got[2], np.log1p(871), places=12)
        # 最小的真实天数也必须严格大于 0，否则 0 不再是唯一哨兵
        self.assertGreater(got[1], 0.0)

    def test_bucket_boundaries(self):
        days = pd.Series([-1, 1, 30, 31, 90, 91, 180, 181, 365, 366, 871])
        got = S.pdays_bucket(days).tolist()
        # 0=未联系；1=(0,30]; 2=(30,90]; 3=(90,180]; 4=(180,365]; 5=365+
        self.assertEqual(got, [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_bucket_never_vs_touched_disjoint(self):
        days = pd.Series([-1, 1])
        self.assertEqual(S.pdays_bucket(days).tolist()[0], 0)
        self.assertEqual(int(S.pdays_contacted(days).iloc[0]), 0)
        self.assertNotEqual(S.pdays_bucket(days).tolist()[1], 0)


class TestOrdinals(unittest.TestCase):
    def test_edu_ordinal_and_unknown(self):
        edu = pd.Series(["小学", "中学", "高等教育", S.UNKNOWN])
        self.assertEqual(S.edu_ordinal(edu).tolist()[:3], [1.0, 2.0, 3.0])
        self.assertTrue(np.isnan(S.edu_ordinal(edu).iloc[3]))
        self.assertEqual(S.edu_unknown(edu).tolist(), [0, 0, 0, 1])

    def test_month_ordinal_covers_all_twelve(self):
        months = pd.Series(list(S.MONTH_ORDER))
        got = S.month_to_num(months).tolist()
        self.assertEqual(got, list(range(1, 13)))

    def test_month_unknown_raises(self):
        with self.assertRaises(ValueError):
            S.month_to_num(pd.Series(["十三月"]))

    def test_cyclical_wraps_around(self):
        # 十二月与一月在周期编码下应当接近，而不是相距 11 个单位
        s, c = S.cyclical(pd.Series([1, 12]), 12)
        dist = np.hypot(s.iloc[0] - s.iloc[1], c.iloc[0] - c.iloc[1])
        self.assertLess(dist, 0.6)
        # 一月至六月应当相距最远
        s2, c2 = S.cyclical(pd.Series([1, 6]), 12)
        far = np.hypot(s2.iloc[0] - s2.iloc[1], c2.iloc[0] - c2.iloc[1])
        self.assertGreater(far, 1.9)


class TestUnknown(unittest.TestCase):
    def test_info_completeness(self):
        df = pd.DataFrame(
            {
                "职业": ["未知", "未知", "未知", "蓝领工人"],
                "教育程度": ["未知", "未知", "中学", "中学"],
                "联系方式": ["未知", "手机", "手机", "手机"],
            }
        )
        self.assertEqual(S.unknown_flags(df).tolist(), [3, 2, 1, 0])
        self.assertEqual(S.info_completeness(df).tolist(), [0, 1, 2, 3])


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = synthetic_raw_frame(rows=200)
        cls.df = D.build_dataset(cls.raw)

    def test_rows_and_target_preserved(self):
        self.assertEqual(len(self.df), len(self.raw))
        self.assertEqual(
            int(self.df[S.TARGET].sum()),
            int((self.raw[S.TARGET_RAW] == S.POSITIVE).sum()),
        )

    def test_no_excluded_col_in_features(self):
        for version in ("A", "B"):
            cols = D.model_features(version)
            for bad in S.EXCLUDE_COLS:
                self.assertNotIn(bad, cols)
            self.assertEqual(len(cols), len(set(cols)), f"{version} 版存在重复列")

    def test_b_version_drops_month(self):
        a, b = set(D.model_features("A")), set(D.model_features("B"))
        self.assertEqual(a - b, set(S.TIME_LEAKY))

    def test_validate_passes_on_synthetic_data(self):
        facts = D.validate(self.df, self.raw)
        self.assertEqual(facts["n_rows"], len(self.raw))
        self.assertAlmostEqual(facts["pos_rate"], 0.20, delta=0.05)

    def test_validate_catches_tampering(self):
        broken = self.df.copy()
        broken[S.PDAYS_LOG] = 0.0  # 把所有 _log 抹成 0，制造 0 碰撞
        with self.assertRaises(AssertionError):
            D.validate(broken, self.raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
