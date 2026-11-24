"""模型 vs 规则名额取舍表的单测：只测「算得对不对」，不测真数据上的数字。"""

from __future__ import annotations

import json
import unittest

import numpy as np
from bank_training import compare as C


class QuotaRowsTest(unittest.TestCase):
    """名额换算与指标：人数、抓到的阳性、精确率、召回率。"""

    def test_quota_sizes_follow_fraction(self) -> None:
        y = np.zeros(100, dtype="int8")
        y[:20] = 1
        score = np.linspace(1.0, 0.0, 100)
        rows = C.quota_rows(y, {"m": score}, fractions=(0.10, 0.20))
        self.assertEqual([row["quota_n"] for row in rows], [10, 20])
        self.assertEqual([row["fraction"] for row in rows], [0.10, 0.20])

    def test_perfect_ranking_gets_all_positives_in_quota(self) -> None:
        y = np.zeros(40, dtype="int8")
        y[:4] = 1
        score = np.concatenate([np.linspace(1.0, 0.5, 20), np.linspace(0.4, 0.0, 20)])
        rows = C.quota_rows(y, {"p": score}, fractions=(0.10,))
        self.assertEqual(rows[0]["tp"], 4)
        self.assertAlmostEqual(rows[0]["precision"], 1.0)
        self.assertAlmostEqual(rows[0]["recall"], 1.0)

    def test_precision_and_recall_agree_with_tp(self) -> None:
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype="int8")
        score = np.array([0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05])
        rows = C.quota_rows(y, {"m": score}, fractions=(0.40,))
        row = rows[0]
        self.assertEqual(row["quota_n"], 4)
        self.assertEqual(row["tp"], 1)
        self.assertAlmostEqual(row["precision"], 0.25)
        self.assertAlmostEqual(row["recall"], 0.5)

    def test_quota_is_at_least_one_and_at_most_rows(self) -> None:
        y = np.array([1, 0, 0], dtype="int8")
        score = np.array([0.9, 0.2, 0.1])
        rows = C.quota_rows(y, {"m": score}, fractions=(0.01, 1.0))
        self.assertEqual([row["quota_n"] for row in rows], [1, 3])


class GainsTest(unittest.TestCase):
    """模型比规则多抓几个人、精确率差多少。"""

    def _rows(self) -> list[dict[str, object]]:
        return [
            {
                "fraction": 0.10,
                "quota_n": 100,
                "method": C.MODEL_NAME,
                "tp": 60,
                "precision": 0.60,
                "recall": 0.5,
            },
            {
                "fraction": 0.10,
                "quota_n": 100,
                "method": C.RULE_NAME,
                "tp": 55,
                "precision": 0.55,
                "recall": 0.46,
            },
            {
                "fraction": 0.20,
                "quota_n": 200,
                "method": C.MODEL_NAME,
                "tp": 90,
                "precision": 0.45,
                "recall": 0.75,
            },
            {
                "fraction": 0.20,
                "quota_n": 200,
                "method": C.RULE_NAME,
                "tp": 95,
                "precision": 0.475,
                "recall": 0.79,
            },
        ]

    def test_differences_are_model_minus_rule(self) -> None:
        pairs = C.gains(self._rows(), fractions=(0.10, 0.20))
        self.assertEqual(pairs[0]["extra_positives"], 5)
        self.assertAlmostEqual(pairs[0]["precision_gap"], 0.05)
        self.assertEqual(pairs[1]["extra_positives"], -5)
        self.assertAlmostEqual(pairs[1]["precision_gap"], -0.025)

    def test_quota_n_comes_from_model_row(self) -> None:
        pairs = C.gains(self._rows(), fractions=(0.20,))
        self.assertEqual(pairs[0]["quota_n"], 200)


class RenderTest(unittest.TestCase):
    """Markdown 至少要有两张表和一段结论，比例写成百分比。"""

    def test_render_has_two_tables_and_conclusion(self) -> None:
        rows = C.quota_rows(
            np.array([1, 1, 0, 0], dtype="int8"),
            {
                "逻辑回归": np.array([0.9, 0.4, 0.3, 0.1]),
                "规则基线": np.array([0.8, 0.2, 0.7, 0.1]),
            },
            fractions=(0.5,),
        )
        text = C.render(rows)
        self.assertIn("## 对比表", text)
        self.assertIn("## 名额取舍", text)
        self.assertIn("## 结论", text)
        self.assertIn("50%", text)
        self.assertEqual(text.count("| 逻辑回归 |"), 1)

    def test_render_is_stable(self) -> None:
        rows = C.quota_rows(
            np.array([1, 0], dtype="int8"),
            {"逻辑回归": np.array([0.9, 0.1]), "规则基线": np.array([0.8, 0.2])},
            fractions=(0.5,),
        )
        self.assertEqual(C.render(rows), C.render(rows))


class MainTest(unittest.TestCase):
    """真跑一遍入口（不写 mlflow）：要出两个文件、契约号对得上。"""

    def test_main_writes_files_and_reports_contract(self) -> None:
        self.assertEqual(C.main(["--no-mlflow"]), 0)
        self.assertTrue(C.MD_PATH.exists())
        self.assertTrue(C.JSON_PATH.exists())
        payload = json.loads(C.JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract"], C.CONTRACT)
        self.assertEqual(len(payload["table"]), 2 * len(C.FRACTIONS))
        self.assertEqual(len(payload["gains"]), len(C.FRACTIONS))
        self.assertIn("## 名额取舍", C.MD_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
