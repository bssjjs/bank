"""追踪层测试：协议、注册表、内存后端与配置文件（不需要起 MLflow）。"""

from __future__ import annotations

import unittest
from pathlib import Path

from bank.config import ConfigError
from bank_training.tracking import TrackingError, available, get_tracker
from bank_training.tracking.base import (
    TRACKERS,
    TrackerOptions,
    as_text,
    load_tracking_config,
    resolve_tracking_uri,
    tracker,
)
from bank_training.tracking.memory import InMemoryTracker


class RegistryTest(unittest.TestCase):
    def test_builtin_backends(self) -> None:
        self.assertEqual(available(), ("memory", "mlflow"))

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(TrackingError) as caught:
            get_tracker("wandb")
        self.assertIn("memory", str(caught.exception))

    def test_duplicate_registration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):

            @tracker("memory")
            def _again(_options: TrackerOptions) -> InMemoryTracker:
                return InMemoryTracker()

    def test_get_tracker_returns_an_implementation(self) -> None:
        self.assertIsInstance(get_tracker("memory"), InMemoryTracker)
        self.assertEqual(get_tracker("memory").name, "memory")


class MemoryTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = InMemoryTracker()

    def test_params_become_strings(self) -> None:
        with self.tracker.start_run("r") as run_id:
            self.tracker.log_params({"c": 1.0, "flag": True, "quota": 356, "set": {1, 2}})
        record = self.tracker.runs[run_id]
        self.assertEqual(record.params["c"], "1.0")
        self.assertEqual(record.params["flag"], "true")
        self.assertEqual(record.params["quota"], "356")
        self.assertTrue(record.params["set"].startswith("["))

    def test_tags_and_metrics(self) -> None:
        with self.tracker.start_run("r") as run_id:
            self.tracker.set_tags({"stage": "drill"})
            self.tracker.log_metric("ap", 0.5)
            self.tracker.log_metrics({"roc_auc": 0.6})
        record = self.tracker.runs[run_id]
        self.assertEqual(record.tags, {"stage": "drill"})
        self.assertEqual(record.metrics, {"ap": 0.5, "roc_auc": 0.6})

    def test_artifact_records_whether_file_exists(self) -> None:
        target = Path(__file__)
        with self.tracker.start_run("r") as run_id:
            self.tracker.log_artifact(target, artifact_path="code")
            self.tracker.log_artifact(Path("no-such-file.json"))
        artifacts = self.tracker.runs[run_id].artifacts
        self.assertTrue(artifacts[0]["exists"])
        self.assertEqual(artifacts[0]["artifact_path"], "code")
        self.assertFalse(artifacts[1]["exists"])

    def test_log_artifact_into_finished_run(self) -> None:
        with self.tracker.start_run("r") as run_id:
            pass
        self.tracker.log_artifact(Path(__file__), run_id=run_id)
        self.assertEqual(len(self.tracker.runs[run_id].artifacts), 1)

    def test_logging_outside_a_run_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.tracker.log_metrics({"ap": 0.5})

    def test_unknown_run_id_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.tracker.log_artifact(Path(__file__), run_id="nope")

    def test_resuming_a_run_reuses_its_record(self) -> None:
        with self.tracker.start_run("r") as run_id:
            self.tracker.log_metrics({"ap": 0.5})
        with self.tracker.start_run("ignored", run_id=run_id):
            self.tracker.log_metrics({"roc_auc": 0.6})
        self.assertEqual(self.tracker.runs[run_id].metrics, {"ap": 0.5, "roc_auc": 0.6})


class ConfigTest(unittest.TestCase):
    def test_real_config(self) -> None:
        cfg = load_tracking_config()
        self.assertEqual(cfg.backend, "mlflow")
        self.assertEqual(cfg.experiment, "bank-marketing")

    def test_relative_sqlite_uri_is_resolved_against_repo(self) -> None:
        resolved = resolve_tracking_uri("sqlite:///mlflow.db")
        self.assertTrue(resolved.startswith("sqlite:///"))
        self.assertIn("mlflow.db", resolved)
        self.assertEqual(resolve_tracking_uri("http://127.0.0.1:5000"), "http://127.0.0.1:5000")

    def _write(self, body: str) -> Path:
        directory = Path(__file__).resolve().parents[2] / ".tmp" / "tracking" / "conf"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{id(body)}.yaml"
        path.write_text(body, encoding="utf-8", newline="\n")
        return path

    def test_unknown_key_is_rejected(self) -> None:
        path = self._write(
            "schema_version: 1\ntracking:\n  backend: mlflow\n  experiment: e\n"
            "  tracking_uri: u\n  extra: 1\n"
        )
        with self.assertRaises(ConfigError) as caught:
            load_tracking_config(path)
        self.assertIn("extra", str(caught.exception))

    def test_empty_backend_is_rejected(self) -> None:
        path = self._write(
            'schema_version: 1\ntracking:\n  backend: ""\n  experiment: e\n  tracking_uri: u\n'
        )
        with self.assertRaises(ConfigError):
            load_tracking_config(path)

    def test_unknown_schema_version_is_rejected(self) -> None:
        path = self._write(
            "schema_version: 2\ntracking:\n  backend: mlflow\n  experiment: e\n  tracking_uri: u\n"
        )
        with self.assertRaises(ConfigError) as caught:
            load_tracking_config(path)
        self.assertIn("只支持 1", str(caught.exception))

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(ConfigError):
            load_tracking_config(Path("no-such-tracking.yaml"))


class AsTextTest(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(as_text(True), "true")
        self.assertEqual(as_text(False), "false")
        self.assertEqual(as_text("x"), "x")
        self.assertEqual(as_text(3), "3")
        self.assertEqual(as_text({"b": 1, "a": 2}), '{"a": 2, "b": 1}')
        self.assertEqual(as_text(None), "null")


class MemoryBackendNotInConfigTest(unittest.TestCase):
    def test_memory_can_still_be_chosen_explicitly(self) -> None:
        self.assertIn("memory", TRACKERS)
        self.assertEqual(get_tracker("memory", TrackerOptions()).name, "memory")


if __name__ == "__main__":
    unittest.main()
