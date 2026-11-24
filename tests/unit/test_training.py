"""M3 训练侧的单测：指标口径、规则基线、Pipeline 可复现、单条解释、训练参数。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import load_catalog
from bank.explain import explain_one
from bank_training import metrics as M
from bank_training import report as R
from bank_training.baselines import rule_score
from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate, threshold_metrics, two_views
from bank_training.model_config import ModelConfigError, load_model_config
from bank_training.pipeline import build_model


def _toy(n: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, valid, _ = load_frozen()
    return train.head(n).reset_index(drop=True), valid.head(max(n // 4, 10)).reset_index(drop=True)


class MetricsTest(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        got = evaluate(y, score, quota_n=7)
        self.assertEqual(got["quota_n"], 7.0)
        self.assertAlmostEqual(got["precision_at_quota"], 1.0)
        self.assertAlmostEqual(got["recall_at_quota"], 1.0)
        self.assertAlmostEqual(got["ap"], 1.0)

    def test_quota_bounds_and_ties(self) -> None:
        y = np.array([1, 0, 1, 0, 1])
        score = np.zeros(5)  # 全同分：按稳定排序取前 2 行
        got = evaluate(y, score, quota_n=2)
        self.assertAlmostEqual(got["precision_at_quota"], 0.5)
        with self.assertRaises(ValueError):
            evaluate(y, score, quota_n=0)
        with self.assertRaises(ValueError):
            evaluate(y, score, quota_n=6)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(np.array([0, 1]), np.array([0.3]), quota_n=1)

    def test_threshold_metrics(self) -> None:
        y = np.array([0, 1, 1, 0])
        score = np.array([0.2, 0.9, 0.6, 0.4])
        got = threshold_metrics(y, score, 0.5)
        self.assertEqual(got["selected"], 2.0)
        self.assertAlmostEqual(got["precision"], 1.0)
        self.assertAlmostEqual(got["recall"], 1.0)


class RuleBaselineTest(unittest.TestCase):
    def test_range_and_order(self) -> None:
        _, valid = _toy()
        score = rule_score(valid)
        self.assertGreaterEqual(score.min(), 0.0)
        self.assertLessEqual(score.max(), 7.0)
        best = valid[(valid[S.POUTCOME] == "成功") & (valid[S.CONTACT] == "手机")]
        if not best.empty:
            self.assertEqual(rule_score(best).iloc[0], 7.0)


class PipelineTest(unittest.TestCase):
    def test_fit_predict_and_reproducible(self) -> None:
        train, valid = _toy(500)
        spec = load_catalog()
        x_train, y_train = split_xy(train, "B")
        x_valid, _ = split_xy(valid, "B")
        columns = list(x_train.columns)
        first = build_model("lr", spec, columns).fit(x_train, y_train)
        second = build_model("lr", spec, columns).fit(x_train, y_train)
        p1 = first.predict_proba(x_valid)[:, 1]
        p2 = second.predict_proba(x_valid)[:, 1]
        self.assertEqual(p1.shape, (len(x_valid),))
        self.assertTrue(((p1 >= 0) & (p1 <= 1)).all())
        np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-12)

    def test_unknown_value_does_not_crash(self) -> None:
        train, valid = _toy(500)
        x_train, y_train = split_xy(train, "B")
        model = build_model("lr", load_catalog(), list(x_train.columns)).fit(x_train, y_train)
        row = valid.head(1).copy()
        row[S.CONTACT] = "卫星电话"  # 训练集里没有的类别
        proba = model.predict_proba(row[x_train.columns])[:, 1]
        self.assertTrue(0.0 <= float(proba[0]) <= 1.0)


class ExplainTest(unittest.TestCase):
    def test_top_k_sorted_by_abs_contribution(self) -> None:
        train, valid = _toy(500)
        x_train, y_train = split_xy(train, "B")
        model = build_model("lr", load_catalog(), list(x_train.columns)).fit(x_train, y_train)
        contributions = explain_one(model, valid.head(1)[list(x_train.columns)], top_k=5)
        self.assertLessEqual(len(contributions), 5)
        values = [abs(float(item["contribution"])) for item in contributions]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertTrue(all(str(item["feature"]) for item in contributions))


class ModelConfigTest(unittest.TestCase):
    def test_real_config(self) -> None:
        cfg = load_model_config()
        self.assertEqual(cfg.model.kind, "lr")
        self.assertGreater(cfg.model.c, 0)

    def test_unknown_key(self) -> None:
        text = "schema_version: 1\nmodel:\n  kind: lr\n  c: 1.0\n  max_iter: 1000\n  lambda: 3\n"
        with self.assertRaises(ModelConfigError):
            load_model_config(_write(text))

    def test_bad_values(self) -> None:
        with self.assertRaises(ModelConfigError):
            load_model_config(
                _write("schema_version: 1\nmodel:\n  kind: svm\n  c: 1.0\n  max_iter: 1000\n")
            )
        with self.assertRaises(ModelConfigError):
            load_model_config(
                _write("schema_version: 1\nmodel:\n  kind: lr\n  c: 0\n  max_iter: 1000\n")
            )
        with self.assertRaises(ModelConfigError):
            load_model_config(
                _write("schema_version: 1\nmodel:\n  kind: lr\n  c: 1.0\n  max_iter: 5\n")
            )


class MetricsConsistencyTest(unittest.TestCase):
    """10 行的手算例子：13 个指标逐项对上手算值（讲义 C-L2a 的验收要求）。"""

    def setUp(self) -> None:
        self.y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])
        self.score = np.array([0.5, 0.4, 0.9, 0.3, 0.8, 0.2, 0.7, 0.65, 0.1, 0.6])
        self.got = evaluate(self.y, self.score, quota_n=4)

    def test_confusion_matches_hand_count(self) -> None:
        # 前 4 名是下标 2/4/6/7，全是正例 → tp=4、fp=0、fn=1、tn=5
        self.assertEqual(self.got["tp_at_quota"], 4.0)
        self.assertEqual(self.got["fp_at_quota"], 0.0)
        self.assertEqual(self.got["fn_at_quota"], 1.0)
        self.assertEqual(self.got["tp_at_quota"] + self.got["fp_at_quota"], 4.0)
        self.assertEqual(self.got["tp_at_quota"] + self.got["fn_at_quota"], 5.0)

    def test_quota_metrics_match_hand_value(self) -> None:
        self.assertAlmostEqual(self.got["precision_at_quota"], 1.0, places=6)
        self.assertAlmostEqual(self.got["recall_at_quota"], 0.8, places=6)
        self.assertAlmostEqual(self.got["f1_at_quota"], 2 * 1.0 * 0.8 / 1.8, places=6)
        self.assertAlmostEqual(self.got["lift_at_quota"], 2.0, places=4)
        self.assertAlmostEqual(self.got["accuracy_at_quota"], 0.9, places=6)
        self.assertAlmostEqual(self.got["base_rate"], 0.5, places=6)

    def test_ranking_and_calibration_match_hand_value(self) -> None:
        self.assertAlmostEqual(self.got["roc_auc"], 1.0, places=6)
        self.assertAlmostEqual(self.got["ap"], 1.0, places=6)
        self.assertAlmostEqual(self.got["brier"], 0.09725, places=6)
        self.assertAlmostEqual(self.got["log_loss"], 0.351593, places=5)
        self.assertAlmostEqual(self.got["ece"], 0.285, places=6)

    def test_two_views_differ_in_selected_count(self) -> None:
        views = two_views(self.y, self.score, quota_n=4, threshold=0.5)
        self.assertEqual(views["top_k"]["selected"], 4.0)
        self.assertEqual(views["fixed_threshold"]["selected"], 6.0)
        self.assertAlmostEqual(views["fixed_threshold"]["precision"], 5 / 6, places=6)
        self.assertAlmostEqual(views["fixed_threshold"]["recall"], 1.0, places=6)

    def test_empty_and_short_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(np.array([], dtype="int8"), np.array([]), quota_n=1)


class BootstrapTest(unittest.TestCase):
    def test_same_seed_same_interval_and_order(self) -> None:
        _, valid = _toy()
        y = valid[S.TARGET].to_numpy()
        score = np.linspace(0, 1, len(y))
        first = M.bootstrap_ci(y, score, quota_n=10, samples=20, seed=7)
        again = M.bootstrap_ci(y, score, quota_n=10, samples=20, seed=7)
        self.assertEqual(first, again)
        for low, high in first.values():
            self.assertLessEqual(low, high)

    def test_different_seed_changes_interval(self) -> None:
        _, valid = _toy()
        y = valid[S.TARGET].to_numpy()
        score = np.linspace(0, 1, len(y))
        a = M.bootstrap_ci(y, score, quota_n=10, samples=20, seed=1)
        b = M.bootstrap_ci(y, score, quota_n=10, samples=20, seed=2)
        self.assertNotEqual(a["ap"], b["ap"])


class ReportTest(unittest.TestCase):
    def test_render_has_all_sections(self) -> None:
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])
        score = np.array([0.5, 0.4, 0.9, 0.3, 0.8, 0.2, 0.7, 0.65, 0.1, 0.6])
        metrics = evaluate(y, score, quota_n=4)
        built = R.Report(
            title="测试报告",
            n=10,
            positives=5,
            quota_n=4,
            base_rate=0.5,
            metrics=metrics,
            views=two_views(y, score, 4),
            comparisons=(R.Comparison("规则基线", metrics),),
            notes=("这是一条说明",),
        )
        text = R.render(built)
        self.assertIn("# 测试报告", text)
        self.assertIn("## 13 个指标", text)
        self.assertIn("名额内精确率", text)
        self.assertIn("## 两种名额口径", text)
        self.assertIn("## 对照", text)
        self.assertIn("这是一条说明", text)
        self.assertIn("—", text)  # 没有区间时那一列是占位符

    def test_comparison_rejects_missing_metric(self) -> None:
        with self.assertRaises(KeyError):
            R.comparison("坏的对照", {"precision_at_quota": 1.0})


def _write(text: str) -> Path:
    """写到仓库的 .tmp 下，不落到工作区根目录（tempfile 在本机沙箱里会被重定向到 cwd）。"""
    import uuid

    directory = S.ROOT / ".tmp" / "model-config"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    unittest.main()
