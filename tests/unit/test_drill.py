"""F7 演练：注入函数必须是确定性的，五项检查必须真的会响。"""

from __future__ import annotations

import unittest

import pandas as pd
from bank import schema as S
from bank_training import drill
from bank_training.data import load_frozen


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            S.ACCOUNT: [1000.0, 2000.0, 3000.0, 4000.0],
            S.OCCUPATION: ["管理", "蓝领", "学生", "退休"],
            S.MARITAL: ["已婚", "单身", "已婚", "离婚"],
            "label": [1, 0, 0, 0],
        }
    )


class InjectionTest(unittest.TestCase):
    def test_shift_numeric_is_affine_and_does_not_mutate(self) -> None:
        frame = _frame()
        shifted = drill.shift_numeric(frame, S.ACCOUNT, factor=2.0, offset=100.0)
        self.assertEqual(list(shifted[S.ACCOUNT]), [2100.0, 4100.0, 6100.0, 8100.0])
        self.assertEqual(list(frame[S.ACCOUNT]), [1000.0, 2000.0, 3000.0, 4000.0])

    def test_shift_category_is_deterministic(self) -> None:
        frame = _frame()
        first = drill.shift_category(frame, S.OCCUPATION, share=0.5, seed=7)
        second = drill.shift_category(frame, S.OCCUPATION, share=0.5, seed=7)
        self.assertEqual(list(first[S.OCCUPATION]), list(second[S.OCCUPATION]))
        self.assertTrue((first[S.OCCUPATION] == S.UNKNOWN).any())
        self.assertEqual(list(frame[S.OCCUPATION]), ["管理", "蓝领", "学生", "退休"])

    def test_shift_category_rejects_bad_share(self) -> None:
        with self.assertRaises(ValueError):
            drill.shift_category(_frame(), S.OCCUPATION, share=1.5)

    def test_flip_negatives_only_touches_negatives(self) -> None:
        frame = _frame()
        flipped = drill.flip_negatives(frame, share=1.0)
        self.assertEqual(int(flipped["label"].sum()), 4)  # 三条负例全翻
        self.assertEqual(int(frame["label"].sum()), 1)  # 原表没动

    def test_flip_negatives_zero_share_keeps_labels(self) -> None:
        frame = _frame()
        self.assertEqual(int(drill.flip_negatives(frame, share=0.0)["label"].sum()), 1)


class DrillOnRealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.valid, _manifest = load_frozen()

    def test_all_five_checks_pass(self) -> None:
        checks = drill.run(self.train, self.valid)
        self.assertEqual(len(checks), 5)
        failures = [check.name for check in checks if not check.passed]
        self.assertEqual(failures, [], f"演练没按预期响应：{failures}")

    def test_numeric_injection_is_caught_and_adaptive_binning_hides_it(self) -> None:
        shifted = drill.shift_numeric(self.valid, S.ACCOUNT)
        from bank import drift as D

        caught = D.psi_numeric(self.train[S.ACCOUNT], shifted[S.ACCOUNT])
        hidden = drill.adaptive_psi_numeric(self.train[S.ACCOUNT], shifted[S.ACCOUNT])
        self.assertGreaterEqual(caught.psi, D.WATCH)
        self.assertLess(hidden, D.STABLE)


if __name__ == "__main__":
    unittest.main()
