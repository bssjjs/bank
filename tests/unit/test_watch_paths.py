"""追加 F6 剩余路径的测试：渠道构造、投递成功、服务告警分支、CLI、错误路径。"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from bank import alerts as A
from bank import schema as S
from bank_training import watch as W


def _root() -> Path:
    path = S.ROOT / ".tmp" / "watch" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _stable(root: Path) -> tuple[Path, Path]:
    drift = root / "drift.json"
    feedback = root / "feedback.json"
    drift.write_text(
        json.dumps(
            {
                "features": [{"feature": "婚姻状态", "psi": 0.01}],
                "label": {"relative_change_pct": 1.0},
            }
        ),
        encoding="utf-8",
    )
    feedback.write_text(json.dumps({"coverage": 0.9}), encoding="utf-8")
    return drift, feedback


class ChannelBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_channels_from_config(self) -> None:
        config = W.WatchConfig.model_validate(
            {
                "schema_version": 1,
                "thresholds": {
                    "psi_watch": 0.1,
                    "psi_alert": 0.25,
                    "label_shift_pct": 50,
                    "coverage_min": 0.7,
                },
                "channels": [
                    {"kind": "console"},
                    {"kind": "file", "path": "data/processed/alert-test.jsonl"},
                    {"kind": "webhook", "url": "https://example.invalid/hook"},
                ],
                "schedule": {"interval_seconds": 60},
            }
        )
        channels = W.channels_from(config, root=self.root)
        self.assertEqual([channel.name for channel in channels], ["console", "file", "webhook"])
        self.assertEqual(channels[1].path, self.root / "data" / "processed" / "alert-test.jsonl")

    def test_console_channel_prints_each_alert(self) -> None:
        buffer = io.StringIO()
        alert = A.Alert("warn", "drift", "psi_x", "x 漂了")
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(A.ConsoleChannel().send([alert]), [])
        self.assertIn("[warn] drift/psi_x: x 漂了", buffer.getvalue())

    def test_webhook_success_returns_nothing(self) -> None:
        class _Response:
            status = 200

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        alert = A.Alert("alert", "drift", "psi_x", "x")
        with mock.patch("bank.alerts.urllib.request.urlopen", return_value=_Response()):
            channel = A.WebhookChannel("https://example.invalid/hook")
            degraded = channel.send([alert])
        self.assertEqual(degraded, [])


class ServiceBranchTest(unittest.TestCase):
    def test_model_unavailable_is_alert(self) -> None:
        found = W.service_alerts("http://svc", fetch=lambda url: {"model": {"available": False}})
        self.assertEqual([(a.code, a.level) for a in found], [("model_unavailable", "alert")])

    def test_reload_failures_is_warning(self) -> None:
        payload = {
            "model": {"available": True},
            "reload": {"reload_failures": 2, "last_reload_error": "坏指针"},
        }
        found = W.service_alerts("http://svc", fetch=lambda url: payload)
        self.assertEqual([(a.code, a.level) for a in found], [("reload_failures", "warn")])
        self.assertIn("坏指针", found[0].message)

    def test_fetch_json_rejects_non_object(self) -> None:
        class _Response:
            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return b"[1, 2]"

        with mock.patch("bank_training.watch.urllib.request.urlopen", return_value=_Response()):
            with self.assertRaises(ValueError):
                W._fetch_json("http://svc/metrics")


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.drift, self.feedback = _stable(self.root)
        self.history = self.root / "history.jsonl"
        self.out = self.root / "watch.json"

    def _args(self) -> list[str]:
        return [
            "--no-send",
            "--drift",
            str(self.drift),
            "--feedback",
            str(self.feedback),
            "--history",
            str(self.history),
            "--out",
            str(self.out),
        ]

    def test_stable_run_returns_zero(self) -> None:
        self.assertEqual(W.main(self._args()), 0)
        self.assertTrue(self.out.exists())

    def test_alert_run_returns_one(self) -> None:
        payload = {"features": [{"feature": "职业", "psi": 0.9}]}
        self.drift.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(W.main(self._args()), 1)

    def test_unknown_argument_returns_two(self) -> None:
        self.assertEqual(W.main(["--nope"]), 2)

    def test_missing_drift_returns_one(self) -> None:
        self.drift.unlink()
        self.assertEqual(W.main(self._args()), 1)


class ErrorPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_schema_version_two_rejected(self) -> None:
        path = self.root / "watch.yaml"
        text = (S.ROOT / "configs" / "watch.yaml").read_text(encoding="utf-8")
        path.write_text(text.replace("schema_version: 1", "schema_version: 2"), encoding="utf-8")
        with self.assertRaises(W.WatchError) as caught:
            W.load_watch_config(path)
        self.assertIn("版本 2", str(caught.exception))

    def test_missing_file_message(self) -> None:
        with self.assertRaises(W.WatchError) as caught:
            W._read_json(self.root / "nope.json", "漂移报告")
        self.assertIn("找不到漂移报告", str(caught.exception))

    def test_bad_json_message(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{oops", encoding="utf-8")
        with self.assertRaises(W.WatchError) as caught:
            W._read_json(path, "回流汇总")
        self.assertIn("不是合法 JSON", str(caught.exception))

    def test_feature_row_missing_psi(self) -> None:
        thresholds = W.Thresholds(
            psi_watch=0.1, psi_alert=0.25, label_shift_pct=50.0, coverage_min=0.7
        )
        with self.assertRaises(W.WatchError):
            W.psi_alerts({"features": [{"feature": "职业"}]}, thresholds)

    def test_bad_history_line(self) -> None:
        history = self.root / "history.jsonl"
        history.write_text("{broken\n", encoding="utf-8")
        with self.assertRaises(W.WatchError) as caught:
            W.previous_breaches(history)
        self.assertIn("最后一行", str(caught.exception))

    def test_empty_history_is_empty(self) -> None:
        self.assertEqual(W.previous_breaches(self.root / "none.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
