"""F6 考卷：配置校验、PSI 分级、渠道投递、连续两轮才建议重训。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import alerts as A
from bank import schema as S
from bank_training import watch as W

YAML = """schema_version: 1
thresholds:
  psi_watch: 0.1
  psi_alert: 0.25
  label_shift_pct: 50
  coverage_min: 0.7
channels:
  - kind: console
schedule:
  interval_seconds: 3600
"""


def _root() -> Path:
    path = S.ROOT / ".tmp" / "watch" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _variant(root: Path, old: str, new: str) -> Path:
    path = root / uuid.uuid4().hex[:8] / "watch.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = YAML if old == "<all>" else YAML.replace(old, new, 1)
    if old == "<all>":
        text = new
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _drift(features: dict[str, float], *, label_shift: float = 0.0) -> dict[str, object]:
    return {
        "contract": "drift_report.v1",
        "features": [{"feature": name, "psi": psi} for name, psi in features.items()],
        "label": {"reference_rate": 0.13, "current_rate": 0.50, "relative_change_pct": label_shift},
    }


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_real_config_loads(self) -> None:
        config = W.load_watch_config()
        self.assertGreater(config.thresholds.psi_alert, config.thresholds.psi_watch)
        self.assertIn("console", [item.kind for item in config.channels])

    def test_unknown_key_rejected(self) -> None:
        path = _variant(self.root, "coverage_min: 0.7", "coverage_min: 0.7\n  coverage_mni: 0.5")
        with self.assertRaises(W.WatchError) as caught:
            W.load_watch_config(path)
        self.assertIn("coverage_mni", str(caught.exception))

    def test_channel_kind_unknown(self) -> None:
        path = _variant(self.root, "- kind: console", "- kind: telegram")
        with self.assertRaises(W.WatchError) as caught:
            W.load_watch_config(path)
        self.assertIn("未知渠道", str(caught.exception))

    def test_channel_file_needs_path(self) -> None:
        path = _variant(self.root, "- kind: console", "- kind: file")
        with self.assertRaises(W.WatchError) as caught:
            W.load_watch_config(path)
        self.assertIn("必须写 path", str(caught.exception))

    def test_alert_line_must_be_above_watch_line(self) -> None:
        path = _variant(self.root, "psi_alert: 0.25", "psi_alert: 0.05")
        with self.assertRaises(W.WatchError) as caught:
            W.load_watch_config(path)
        self.assertIn("必须大于", str(caught.exception))


class SeverityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = W.Thresholds(
            psi_watch=0.1, psi_alert=0.25, label_shift_pct=50.0, coverage_min=0.7
        )

    def test_psi_levels_from_list(self) -> None:
        found = W.psi_alerts(_drift({"职业": 0.3, "年龄": 0.15, "婚姻状态": 0.05}), self.thresholds)
        levels = {alert.code: alert.level for alert in found}
        self.assertEqual(levels, {"psi_职业": "alert", "psi_年龄": "warn"})

    def test_psi_accepts_mapping_form(self) -> None:
        found = W.psi_alerts({"features": {"职业": {"psi": 0.9}}}, self.thresholds)
        self.assertEqual([(a.code, a.level) for a in found], [("psi_职业", "alert")])

    def test_label_shift_alerts(self) -> None:
        found = W.label_alerts(_drift({}, label_shift=277.66), self.thresholds)
        self.assertEqual([a.code for a in found], ["label_shift"])
        self.assertEqual(W.label_alerts(_drift({}, label_shift=10.0), self.thresholds), [])

    def test_coverage_low_is_warn(self) -> None:
        found = W.feedback_alerts({"coverage": 0.5}, self.thresholds)
        self.assertEqual([(a.code, a.level) for a in found], [("coverage_low", "warn")])
        self.assertEqual(W.feedback_alerts({"coverage": 0.9}, self.thresholds), [])


class ChannelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_bad_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            A.Alert("fatal", "drift", "x", "y")

    def test_file_channel_appends_jsonl(self) -> None:
        path = self.root / "alerts.jsonl"
        channel = A.FileChannel(path)
        channel.send([A.Alert("warn", "drift", "psi_x", "x 漂了", 0.3)])
        channel.send([A.Alert("alert", "drift", "psi_y", "y 漂了", 0.9)])
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["level"], "warn")
        self.assertEqual(json.loads(lines[1])["value"], 0.9)

    def test_webhook_failure_becomes_warning(self) -> None:
        channel = A.WebhookChannel("http://127.0.0.1:9/none", timeout=0.5)
        degraded = channel.send([A.Alert("alert", "drift", "psi_x", "x")])
        self.assertEqual([(a.code, a.level) for a in degraded], [("webhook_failed", "warn")])


class RunOnceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _root()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.config = W.load_watch_config()
        self.drift = self.root / "drift.json"
        self.feedback = self.root / "feedback.json"
        self.history = self.root / "history.jsonl"
        self.out = self.root / "watch.json"
        self.drift.write_text(
            json.dumps(_drift({"职业": 0.3, "婚姻状态": 0.05}, label_shift=5.0)), encoding="utf-8"
        )
        self.feedback.write_text(json.dumps({"coverage": 0.9}), encoding="utf-8")

    def _run(self) -> tuple[list[A.Alert], int]:
        return W.run_once(
            config=self.config,
            drift_path=self.drift,
            feedback_path=self.feedback,
            history_path=self.history,
            out_path=self.out,
            send=False,
        )

    def test_alert_exit_code_and_report(self) -> None:
        found, code = self._run()
        self.assertEqual(code, 1)
        self.assertEqual([a.code for a in found], ["psi_职业"])
        payload = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract"], W.CONTRACT)
        self.assertEqual(payload["breaches"], ["职业"])

    def test_two_runs_trigger_retrain(self) -> None:
        self._run()
        found, _ = self._run()
        self.assertIn("psi_two_runs", [a.code for a in found])
        self.assertEqual(len(self.history.read_text(encoding="utf-8").splitlines()), 2)

    def test_missing_feedback_is_a_warning(self) -> None:
        self.feedback.unlink()
        found, _ = self._run()
        self.assertIn("missing_feedback", [a.code for a in found])

    def test_no_alerts_when_stable(self) -> None:
        self.drift.write_text(
            json.dumps(_drift({"婚姻状态": 0.01}, label_shift=1.0)), encoding="utf-8"
        )
        found, code = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(found, [])

    def test_service_unreachable_is_warning(self) -> None:
        found = W.service_alerts("http://127.0.0.1:9", fetch=None)
        self.assertEqual([(a.code, a.level) for a in found], [("service_unreachable", "warn")])


if __name__ == "__main__":
    unittest.main()
