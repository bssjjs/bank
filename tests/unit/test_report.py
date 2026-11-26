"""评估报告的两份产物：同一份 `Report` 写成 Markdown 与 JSON，数字必须同源。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import schema as S
from bank_training.report import Report, write_json


def _report(metrics: dict[str, float] | None = None) -> Report:
    return Report(
        title="评估报告：测试",
        n=100,
        positives=30,
        quota_n=10,
        base_rate=0.3,
        metrics={"ap": 0.5123, "precision_at_quota": 0.7} if metrics is None else metrics,
        intervals={"ap": (0.45, 0.57)},
        views={"quota": {"n": 10.0}},
        calibration={"ece": 0.12},
        reliability=({"bin": 0.0, "count": 10.0},),
        slices={"职业": [{"name": "蓝领工人", "n": 20.0}]},
    )


class WriteJsonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 沙箱只允许写工作区；不用 TemporaryDirectory（它建出来的目录在
        # Windows 沙箱里会被拒绝写入），改用手工命名的子目录。
        cls.tmp_root = S.ROOT / ".tmp" / "test_report"
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _workdir(self) -> Path:
        path = self.tmp_root / f"case-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def test_round_trip_keeps_the_same_numbers(self) -> None:
        path = write_json(self._workdir() / "evaluation_report.json", _report())
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        payload = json.loads(text)
        self.assertEqual(payload["metrics"], {"ap": 0.5123, "precision_at_quota": 0.7})
        self.assertEqual(payload["quota_n"], 10)
        self.assertEqual(payload["intervals"]["ap"], [0.45, 0.57])
        self.assertEqual(payload["slices"]["职业"][0]["name"], "蓝领工人")

    def test_refuses_nan_instead_of_writing_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            write_json(self._workdir() / "bad.json", _report(metrics={"ap": float("nan")}))


if __name__ == "__main__":
    unittest.main()
