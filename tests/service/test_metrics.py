"""E-L2 服务指标：`GET /metrics` 的计数要能对得上真实请求，且不泄露特征值。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from typing import Any

from bank import bundle as B
from bank import schema as S
from bank_api.app import create_app
from bank_training.data import load_frozen
from fastapi.testclient import TestClient

from tests.support import ensure_served_bundle


def _payload(bundle: B.Bundle, row: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in bundle.doc.required_columns:
        value = row[name]
        data[name] = value.item() if hasattr(value, "item") else value
    return data


class ServiceMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_served_bundle()
        cls.bundle = B.load_bundle(B.SERVING_DIR)
        cls.valid = load_frozen()[1]
        cls.root = S.ROOT / ".tmp" / "test-metrics" / uuid.uuid4().hex
        cls.root.mkdir(parents=True)
        cls.client = TestClient(create_app(log_path=cls.root / "predictions.jsonl"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _metrics(self) -> dict[str, Any]:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200, response.text)
        body: dict[str, Any] = response.json()
        return body

    def test_contract_and_model_block(self) -> None:
        body = self._metrics()
        self.assertEqual(body["contract"], "service_metrics.v1")
        self.assertTrue(body["model"]["available"])
        self.assertEqual(body["model"]["version"], self.bundle.version)
        self.assertIn("hot_reload", body["reload"])
        for key in ("uptime_s", "requests", "by_route", "by_status", "predictions", "by_level"):
            self.assertIn(key, body)

    def test_counts_requests_and_latency(self) -> None:
        before = self._metrics()
        self.client.get("/health")
        after = self._metrics()
        self.assertGreaterEqual(after["requests"], before["requests"] + 2)  # /health + /metrics
        self.assertGreater(after["latency_ms_max"], 0.0)
        self.assertIn("/health 200", after["by_route"])

    def test_predictions_and_levels_counted(self) -> None:
        before = self._metrics()
        sample = _payload(self.bundle, self.valid.iloc[0])
        body = self.client.post("/v1/predict", json=sample).json()
        after = self._metrics()
        self.assertEqual(after["predictions"], before["predictions"] + 1)
        self.assertEqual(
            after["by_level"].get(body["level"], 0),
            before["by_level"].get(body["level"], 0) + 1,
        )
        self.assertIn("/v1/predict 200", after["by_route"])

    def test_batch_counts_every_item(self) -> None:
        before = self._metrics()
        items = [_payload(self.bundle, self.valid.iloc[index]) for index in range(3)]
        self.assertEqual(self.client.post("/v1/predict/batch", json=items).status_code, 200)
        after = self._metrics()
        self.assertEqual(after["predictions"], before["predictions"] + 3)

    def test_422_counted_but_not_a_prediction(self) -> None:
        before = self._metrics()
        sample = _payload(self.bundle, self.valid.iloc[0])
        sample["年龄"] = 999
        self.assertEqual(self.client.post("/v1/predict", json=sample).status_code, 422)
        after = self._metrics()
        self.assertEqual(after["predictions"], before["predictions"])
        self.assertGreaterEqual(after["by_status"].get("422", 0), 1)

    def test_metrics_do_not_leak_feature_values(self) -> None:
        self.client.post("/v1/predict", json=_payload(self.bundle, self.valid.iloc[0]))
        text = json.dumps(self._metrics(), ensure_ascii=False)
        self.assertNotIn("退休", text)
        self.assertNotIn("81204", text)


if __name__ == "__main__":
    unittest.main()
