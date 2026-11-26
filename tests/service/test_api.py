"""M5 服务的行为测试：探针、契约、422 字段级报错、预测记录。

用 TestClient 跑在进程内，不占端口；模型包用 `models/serving` 里真实的 1.0.0。

模型包不在 Git 里（10 KB 也是二进制），所以**干净检出时这里会先导出一次**：
质量门不能依赖"我本机跑过 export"这种前提，否则换台机器或换个 tag 回放就红。
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any

from bank import bundle as B
from bank import schema as S
from bank_api.app import create_app
from bank_training.data import load_frozen
from fastapi.testclient import TestClient

from tests.support import ensure_served_bundle

VERSION = "1.0.0"


def _new_root() -> Path:
    root = S.ROOT / ".tmp" / "test-api" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def _payload(bundle: B.Bundle, row: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in bundle.doc.required_columns:
        value = row[name]
        data[name] = value.item() if hasattr(value, "item") else value
    return data


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_served_bundle(VERSION)
        cls.bundle = B.load_bundle(B.SERVING_DIR)
        cls.valid = load_frozen()[1]
        cls.root = _new_root()
        cls.log_file = cls.root / "predictions.jsonl"
        cls.app = create_app(log_path=cls.log_file)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _sample(self, index: int = 0) -> dict[str, Any]:
        return _payload(self.bundle, self.valid.iloc[index])

    def _log_lines(self) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []
        text = self.log_file.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def test_health_is_200(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ready_reports_model_version(self) -> None:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], self.bundle.version)

    def test_model_info_drives_the_form(self) -> None:
        info = self.client.get("/v1/model").json()
        self.assertEqual(info["version"], self.bundle.version)
        self.assertEqual(len(info["inputs"]), len(self.bundle.doc.required_columns))
        self.assertEqual(info["decision"]["quota_n"], 356)
        for field in info["inputs"]:
            self.assertTrue({"name", "type", "display"} <= set(field))
        numeric = {field["name"]: field for field in info["inputs"]}
        self.assertEqual(numeric["年龄"]["min"], 18.0)
        self.assertEqual(numeric["年龄"]["max"], 95.0)

    def test_predict_returns_score_level_factors_advice(self) -> None:
        response = self.client.post("/v1/predict", json=self._sample())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreaterEqual(body["score"], 0.0)
        self.assertLessEqual(body["score"], 1.0)
        self.assertIn(body["level"], B.LEVELS)
        self.assertEqual(len(body["factors"]), 5)
        self.assertTrue(body["advice"])
        self.assertTrue(body["request_id"].startswith("bank-"))
        self.assertEqual(body["model_version"], self.bundle.version)
        self.assertEqual(body["contribution_unit"], "log_odds")
        self.assertGreaterEqual(body["base_probability"], 0.0)
        self.assertLessEqual(body["base_probability"], 1.0)
        for factor in body["factors"]:
            self.assertIn("feature", factor)
            self.assertIn("contribution", factor)

    def test_out_of_range_is_422_on_that_field(self) -> None:
        payload = self._sample()
        payload["年龄"] = 999
        response = self.client.post("/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        locs = [item["loc"] for item in response.json()["detail"]]
        self.assertEqual(locs, [["body", "年龄"]])

    def test_unknown_category_is_422_on_that_field(self) -> None:
        payload = self._sample()
        payload["职业"] = "宇航员"
        response = self.client.post("/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual([item["loc"] for item in response.json()["detail"]], [["body", "职业"]])

    def test_missing_field_is_422_on_that_field(self) -> None:
        payload = self._sample()
        payload.pop("教育程度")
        response = self.client.post("/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            [item["loc"] for item in response.json()["detail"]], [["body", "教育程度"]]
        )

    def test_nan_is_422_not_a_crash(self) -> None:
        """JSON 里混进 NaN 时在门口就挡住，不放它走到 sklearn 里炸。"""
        payload = self._sample()
        body = json.dumps(payload, ensure_ascii=False)
        # 直接 json.dumps(float("nan")) 会被标准库拒绝，所以先序列化再把那个数换成 NaN：
        # 这模拟的是「别人手写的请求体里带了 NaN 这个 token」。
        body = body.replace(f'"年龄": {payload["年龄"]}', '"年龄": NaN')
        response = self.client.post(
            "/v1/predict", content=body, headers={"content-type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual([item["loc"] for item in response.json()["detail"]], [["body", "年龄"]])

    def test_batch_returns_one_item_per_row(self) -> None:
        items = [self._sample(index) for index in range(3)]
        response = self.client.post("/v1/predict/batch", json=items)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(body["items"]), 3)
        self.assertEqual(len({item["request_id"] for item in body["items"]}), 1)
        self.assertEqual(body["items"][0]["request_id"], body["request_id"])

    def test_batch_over_limit_is_422(self) -> None:
        items = [self._sample() for _ in range(101)]
        response = self.client.post("/v1/predict/batch", json=items)
        self.assertEqual(response.status_code, 422)
        self.assertIn("最多 100 条", response.json()["detail"][0]["msg"])

    def test_empty_batch_is_422(self) -> None:
        response = self.client.post("/v1/predict/batch", json=[])
        self.assertEqual(response.status_code, 422)

    def test_every_prediction_writes_one_log_line(self) -> None:
        before = len(self._log_lines())
        self.client.post("/v1/predict", json=self._sample())
        self.client.post("/v1/predict", json=self._sample(1), params={"control": "true"})
        self.client.post("/v1/predict/batch", json=[self._sample(2), self._sample(3)])
        lines = self._log_lines()
        self.assertEqual(len(lines) - before, 4)
        for line in lines[before:]:
            self.assertEqual(
                set(line),
                {
                    "ts",
                    "contract",
                    "request_id",
                    "model_version",
                    "score",
                    "level",
                    "control",
                    "features_hash",
                    "quota_frac",
                    "quota_n",
                },
            )
            self.assertEqual(line["contract"], "prediction_record.v2")
            self.assertEqual(len(line["features_hash"]), 64)
            self.assertNotIn("年龄", line)  # 日志里不写客户特征值，只留哈希
            self.assertIsInstance(line["control"], bool)
        self.assertTrue(lines[before + 1]["control"])
        # 不显式指定时按 request_id 哈希自动判定，同一批条目共用一个标记
        batch = [line for line in lines[before + 2 :]]
        self.assertEqual(len({line["control"] for line in batch}), 1)


class NotReadyTest(unittest.TestCase):
    def test_missing_bundle_is_503_but_health_still_200(self) -> None:
        root = _new_root()
        try:
            app = create_app(bundle_path=root / "empty", log_path=root / "log.jsonl")
            client = TestClient(app)
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/v1/model").status_code, 503)
            response = client.post("/v1/predict", json={})
            self.assertEqual(response.status_code, 503)
            self.assertIn("模型包未就绪", response.json()["detail"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
