"""算法注册表的契约：一个文件一种算法，且每种算法都过得去同一套检查。

这份测试是给"以后再加一种算法"准备的：新算法只要在
`packages/training/src/bank_training/algorithms/` 下放一个文件并用 `@algorithm("名字")`
登记，就会自动被这里的三组断言覆盖——

1. 注册表本身：名字合规、不许重名、`MODEL_KINDS` 与注册表一致、文件和登记一一对应；
2. 能训练能预测：在冻结集的头部小样本上训练两次，两次分数必须逐位相同（不许有隐藏随机性）；
3. 与冻结参照一致：`python -m bank_training.parity --check` 必须通过（指标容差 1e-6、逐条
   分数指纹相同、名额内人数完全相等）。
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from bank import catalog
from bank import schema as S
from bank_training import algorithms, data, parity, pipeline
from bank_training import metrics as M

ALGORITHM_DIR = Path(pipeline.__file__).parent / "algorithms"
SAMPLE_ROWS = 800
# 先把登记表加载出来：`algorithm()` 只能看见已经 import 过的模块，
# 不先加载就直接登记测试用的假名字，会把真名字占掉（这里踩过一次）。
algorithms.available()


def _frozen_head() -> tuple[object, object, object]:
    train, valid, manifest = data.load_frozen()
    return train.head(SAMPLE_ROWS), valid, manifest


class RegistryTest(unittest.TestCase):
    def test_known_algorithms(self) -> None:
        self.assertEqual(algorithms.available(), ("lr", "rf"))

    def test_pipeline_kinds_match_registry(self) -> None:
        self.assertEqual(pipeline.MODEL_KINDS, algorithms.available())

    def test_unknown_name_lists_available(self) -> None:
        spec = catalog.load_catalog()
        with self.assertRaises(ValueError) as caught:
            algorithms.build("gbm", spec, ["年龄"])
        message = str(caught.exception)
        self.assertIn("gbm", message)
        self.assertIn("lr, rf", message)

    def test_duplicate_registration_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            algorithms.algorithm("lr")(lambda *a, **k: None)
        self.assertIn("重复登记", str(caught.exception))

    def test_bad_names_rejected(self) -> None:
        for name in ("", "Lr", "有中文", "a-b", "9x"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    algorithms.algorithm(name)(lambda *a, **k: None)

    def test_one_file_per_algorithm(self) -> None:
        files = sorted(p.stem for p in ALGORITHM_DIR.glob("*.py") if p.name != "__init__.py")
        self.assertEqual(files, ["forest", "linear"])
        registered: set[str] = set()
        for stem in files:
            module = importlib.import_module(f"bank_training.algorithms.{stem}")
            self.assertTrue(module.__doc__, f"{stem}.py 缺文档字符串：要说清这是什么算法")
            self.assertTrue(callable(getattr(module, "build", None)), f"{stem}.py 没有 build")
            registered.add(module.build.__name__ if hasattr(module.build, "__name__") else stem)
        self.assertEqual(len(algorithms.available()), len(files))


class FitsAndPredictsTest(unittest.TestCase):
    def test_every_algorithm_is_deterministic(self) -> None:
        spec = catalog.load_catalog()
        frame, _, _ = _frozen_head()
        features = list(data.split_xy(frame, "B")[0].columns)
        x, y = data.split_xy(frame, "B")
        for name in algorithms.available():
            with self.subTest(algorithm=name):
                first = algorithms.build(name, spec, features).fit(x, y)
                second = algorithms.build(name, spec, features).fit(x, y)
                scores_first = first.predict_proba(x)[:, 1]
                scores_second = second.predict_proba(x)[:, 1]
                np.testing.assert_allclose(scores_first, scores_second, atol=0.0, rtol=0.0)
                self.assertEqual(scores_first.shape, (len(x),))
                self.assertFalse(np.isnan(scores_first).any())
                self.assertTrue(((scores_first >= 0) & (scores_first <= 1)).all())


class ParityReferenceTest(unittest.TestCase):
    def test_reference_file_covers_every_algorithm(self) -> None:
        self.assertTrue(parity.REF_PATH.is_file(), "缺 baselines/v1/algorithm_parity.json")
        import json

        payload = json.loads(parity.REF_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract"], parity.CONTRACT)
        self.assertEqual(sorted(payload["algorithms"]), sorted(algorithms.available()))
        self.assertEqual(payload["feature_set"], "B")
        self.assertEqual(payload["seed"], S.SEED)
        for name, row in payload["algorithms"].items():
            with self.subTest(algorithm=name):
                self.assertEqual(len(row["scores_sha256"]), 64)
                self.assertEqual(row["quota_n"], row["n_selected"])
                self.assertGreater(row["quota_n"], 0)
                for field in ("ap", "roc_auc", "precision_at_quota", "recall_at_quota"):
                    self.assertGreaterEqual(float(row[field]), 0.0)
                    self.assertLessEqual(float(row[field]), 1.0)

    def test_check_passes_against_reference(self) -> None:
        self.assertEqual(parity.check(), 0)


class ParityToleranceTest(unittest.TestCase):
    """容差与人数：差一点点放过，差出指标或人数就报出来（用造出来的报告测，不重训）。"""

    def _report(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "n": 10,
            "positives": 5,
            "quota_n": 3,
            "n_selected": 3,
            "ap": 0.5,
            "roc_auc": 0.5,
            "precision_at_quota": 0.5,
            "recall_at_quota": 0.5,
            "scores_sha256": "a" * 64,
        }
        return {"algorithms": {"lr": dict(row)}}

    def test_tiny_drift_within_tolerance(self) -> None:
        got = self._report()
        got["algorithms"]["lr"]["ap"] = 0.5 + 5e-7
        self.assertEqual(parity._differences(got, self._report(), parity.TOLERANCE), [])

    def test_metric_drift_reported(self) -> None:
        got = self._report()
        got["algorithms"]["lr"]["ap"] = 0.51
        problems = parity._differences(got, self._report(), parity.TOLERANCE)
        self.assertTrue(any("lr.ap" in line for line in problems))

    def test_count_drift_reported(self) -> None:
        got = self._report()
        got["algorithms"]["lr"]["n_selected"] = 4
        problems = parity._differences(got, self._report(), parity.TOLERANCE)
        self.assertTrue(any("n_selected" in line for line in problems))

    def test_extra_algorithm_reported(self) -> None:
        got = self._report()
        got["algorithms"]["gbm"] = dict(got["algorithms"]["lr"])
        problems = parity._differences(got, self._report(), parity.TOLERANCE)
        self.assertTrue(any("多出算法 gbm" in line for line in problems))

    def test_metrics_module_sees_same_quota(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        self.assertEqual(M.evaluate(y, scores, 3)["quota_n"], 3)


if __name__ == "__main__":
    unittest.main()
