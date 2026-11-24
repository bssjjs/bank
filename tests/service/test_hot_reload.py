"""E-L2 热加载：只改 `current.json` 就换模型，不重启进程。

不依赖真实 `models/serving`：在 `.tmp` 里复制两份版本目录（1.0.0 / 1.1.0），
把 `current.json` 指过去再指回来，观察 `/ready`、`/v1/predict` 与 `/metrics` 的变化。
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any

from bank import bundle as B
from bank import repro
from bank import schema as S
from bank_api.app import create_app
from bank_training.data import load_frozen
from fastapi.testclient import TestClient

from tests.support import ensure_served_bundle

VERSION_FROM = "1.0.0"
VERSION_TO = "1.1.0"


def _new_root() -> Path:
    root = S.ROOT / ".tmp" / "hot-reload" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def _clone(src: Path, root: Path, version: str) -> Path:
    """复制一个版本目录并改版本号；校验和按同样口径重算，否则加载会被拒。"""
    target = root / version
    shutil.copytree(src, target)
    doc = json.loads((target / B.BUNDLE_JSON).read_text(encoding="utf-8"))
    doc["version"] = version
    payload = {key: value for key, value in doc.items() if key != "checksum"}
    doc["checksum"] = {"algorithm": "sha256", "value": repro.sha256_json(payload)}
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    (target / B.BUNDLE_JSON).write_text(text, encoding="utf-8")
    return target


def _payload(bundle: B.Bundle, row: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in bundle.doc.required_columns:
        value = row[name]
        data[name] = value.item() if hasattr(value, "item") else value
    return data


class HotReloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_served_bundle(VERSION_FROM)
        cls.source = B.SERVING_DIR / B.read_current()
        cls.root = _new_root()
        _clone(cls.source, cls.root, VERSION_FROM)
        _clone(cls.source, cls.root, VERSION_TO)
        B.write_current(VERSION_FROM, cls.root)
        cls.valid = load_frozen()[1]
        cls.bundle = B.load_bundle(cls.root / VERSION_FROM)
        cls.log_file = cls.root / "predictions.jsonl"
        cls.app = create_app(serving_dir=cls.root, log_path=cls.log_file)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _ready_version(self) -> str:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200, response.text)
        return str(response.json()["model_version"])

    def _metrics(self) -> dict[str, Any]:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200, response.text)
        body: dict[str, Any] = response.json()
        return body

    # 用例按名字排序执行，状态是链条式推进的（每一步依赖上一步的结果）

    def test_1_serves_pointer_version(self) -> None:
        self.assertEqual(self._ready_version(), VERSION_FROM)
        sample = _payload(self.bundle, self.valid.iloc[0])
        body = self.client.post("/v1/predict", json=sample).json()
        self.assertEqual(body["model_version"], VERSION_FROM)

    def test_2_pointer_change_swaps_model_without_restart(self) -> None:
        before = self._metrics()["reload"]["loads"]
        B.write_current(VERSION_TO, self.root)
        self.assertEqual(self._ready_version(), VERSION_TO)
        sample = _payload(self.bundle, self.valid.iloc[0])
        body = self.client.post("/v1/predict", json=sample).json()
        self.assertEqual(body["model_version"], VERSION_TO)
        metrics = self._metrics()
        self.assertGreater(metrics["reload"]["loads"], before)
        self.assertEqual(metrics["model"]["version"], VERSION_TO)
        self.assertTrue(metrics["reload"]["hot_reload"])

    def test_3_broken_pointer_keeps_old_model(self) -> None:
        failures_before = self._metrics()["reload"]["reload_failures"]
        (self.root / B.CURRENT_FILE).write_text("{这不是 JSON", encoding="utf-8")
        # 指针坏了：旧模型继续服务，探针不倒
        self.assertEqual(self._ready_version(), VERSION_TO)
        metrics = self._metrics()
        self.assertGreater(metrics["reload"]["reload_failures"], failures_before)
        self.assertTrue(metrics["reload"]["last_reload_error"])
        self.assertEqual(metrics["model"]["version"], VERSION_TO)
        # 修好后自动恢复
        B.write_current(VERSION_FROM, self.root)
        self.assertEqual(self._ready_version(), VERSION_FROM)
        self.assertIsNone(self._metrics()["reload"]["last_reload_error"])

    def test_4_fixed_bundle_disables_hot_reload(self) -> None:
        app = create_app(
            bundle_path=self.root / VERSION_TO,
            log_path=self.root / "fixed.jsonl",
        )
        metrics = TestClient(app).get("/metrics").json()
        self.assertFalse(metrics["reload"]["hot_reload"])
        self.assertEqual(metrics["model"]["version"], VERSION_TO)


if __name__ == "__main__":
    unittest.main()
