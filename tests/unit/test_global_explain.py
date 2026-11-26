"""全局解释的单元测试：两套重要度对齐到原始列、方向正确、结论可渲染。

全局解释只用线性贡献（系数 × 标准化值，与单条解释同源）与置换重要度，
不引入 shap 等新依赖。测试用小号冻结集切片，不读全量数据。
"""

from __future__ import annotations

import unittest

from bank.catalog import load_catalog
from bank_training.data import load_frozen, split_xy
from bank_training.global_explain import _group_by_raw, global_rows, render
from bank_training.pipeline import build_model


class GroupByRawTest(unittest.TestCase):
    def test_one_hot_groups_back_to_parent(self) -> None:
        import numpy as np

        names = ["年龄", "职业=蓝领工人", "职业=学生", "住房贷款"]
        per_row = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        raw, grouped = _group_by_raw(names, per_row)
        self.assertEqual(raw, ["年龄", "职业", "住房贷款"])
        # 职业的两项按行相加：第一行 2+3=5，第二行 6+7=13
        np.testing.assert_allclose(grouped, np.array([[1.0, 5.0, 4.0], [5.0, 13.0, 8.0]]))


class GlobalExplainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train, valid, _ = load_frozen()
        cls.train = train.head(400).reset_index(drop=True)
        cls.valid = valid.head(100).reset_index(drop=True)

    def test_rows_cover_all_raw_features_and_are_sorted(self) -> None:
        spec = load_catalog()
        x_train, y_train = split_xy(self.train, "B")
        x_valid, y_valid = split_xy(self.valid, "B")
        model = build_model("lr", spec, list(x_train.columns)).fit(x_train, y_train)
        rows = global_rows(model, x_valid, y_valid, quota_n=10)
        self.assertEqual(len(rows), len(x_valid.columns))
        # 按置换重要度降序：前一个不小于后一个
        perm = [r.perm_importance for r in rows]
        self.assertEqual(perm, sorted(perm, reverse=True))

    def test_direction_is_sign(self) -> None:
        spec = load_catalog()
        x_train, y_train = split_xy(self.train, "B")
        x_valid, y_valid = split_xy(self.valid, "B")
        model = build_model("lr", spec, list(x_train.columns)).fit(x_train, y_train)
        rows = global_rows(model, x_valid, y_valid, quota_n=10)
        for row in rows:
            self.assertIn(row.linear_direction, ("+", "−"))

    def test_render_has_header_and_rows(self) -> None:
        from bank_training.global_explain import GlobalRow

        rows = [GlobalRow("账户余额", 1.0, "+", 0.05), GlobalRow("职业", 0.5, "−", 0.02)]
        text = render(rows, quota_n=10, n=100)
        self.assertIn("# 全局解释", text)
        self.assertIn("账户余额", text)
        self.assertIn("置换重要度", text)


if __name__ == "__main__":
    unittest.main()
