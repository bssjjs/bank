"""C-L1b：解释必须和模型对得上——贡献之和 + 截距 = logit。"""

from __future__ import annotations

import unittest

import numpy as np
from bank import catalog as C
from bank import explain as E
from bank import schema as S
from bank_training import algorithms
from bank_training import data as D
from bank_training import model_config as MC


def _fitted():
    spec = C.load_catalog()
    config = MC.load_model_config()
    train, valid, _ = D.load_frozen()
    features = list(D.split_xy(train, feature_set="B")[0].columns)
    pipeline = algorithms.build(
        config.model.kind,
        spec,
        features,
        c=config.model.c,
        max_iter=config.model.max_iter,
    )
    x_train, y_train = D.split_xy(train, feature_set="B")
    pipeline.fit(x_train, y_train)
    x_valid, _ = D.split_xy(valid, feature_set="B")
    return pipeline, x_valid.head(20)


class ContributionIdentityTest(unittest.TestCase):
    """解释是"为什么给这个分"，那就必须能加回这个分。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline, cls.rows = _fitted()
        cls.spec = C.load_catalog()

    def test_contributions_plus_intercept_equal_decision(self) -> None:
        for index in range(len(self.rows)):
            row = self.rows.iloc[[index]]
            intercept, items = E.contributions(self.pipeline, row)
            total = intercept + sum(item.contribution for item in items)
            decision = float(np.asarray(self.pipeline.decision_function(row)).reshape(-1)[0])
            self.assertAlmostEqual(total, decision, places=10)

    def test_sigmoid_of_decision_is_the_predicted_probability(self) -> None:
        for index in range(len(self.rows)):
            row = self.rows.iloc[[index]]
            decision = float(np.asarray(self.pipeline.decision_function(row)).reshape(-1)[0])
            probability = float(np.asarray(self.pipeline.predict_proba(row)).reshape(-1)[1])
            self.assertAlmostEqual(1.0 / (1.0 + np.exp(-decision)), probability, places=10)

    def test_contributions_cover_every_transformed_column(self) -> None:
        row = self.rows.iloc[[0]]
        _, items = E.contributions(self.pipeline, row)
        pre = self.pipeline.steps[0][1]
        self.assertEqual(len(items), len(E.column_names(pre)))
        self.assertEqual(len({item.feature for item in items}), len(items))

    def test_contributions_are_sorted_by_absolute_value(self) -> None:
        _, items = E.contributions(self.pipeline, self.rows.iloc[[0]])
        magnitudes = [abs(item.contribution) for item in items]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_explain_one_is_the_rounded_head_of_contributions(self) -> None:
        row = self.rows.iloc[[0]]
        _, items = E.contributions(self.pipeline, row)
        shown = E.explain_one(self.pipeline, row, top_k=5)
        self.assertEqual(
            [entry["feature"] for entry in shown], [item.feature for item in items[:5]]
        )
        for entry, item in zip(shown, items[:5], strict=True):
            self.assertAlmostEqual(float(entry["contribution"]), round(item.contribution, 6))

    def test_non_linear_model_is_rejected(self) -> None:
        spec = C.load_catalog()
        features = list(self.rows.columns)
        pipeline = algorithms.build("rf", spec, features, c=1.0, max_iter=5000)
        pipeline.fit(self.rows, np.zeros(len(self.rows), dtype="int8"))
        with self.assertRaisesRegex(RuntimeError, "解释只支持线性模型"):
            E.contributions(pipeline, self.rows.iloc[[0]])

    def test_method_name_matches_the_implementation(self) -> None:
        self.assertEqual(E.EXPLAIN_METHOD, "linear_coef_x_scaled")
        self.assertTrue(hasattr(S, "TARGET"))


if __name__ == "__main__":
    unittest.main()
