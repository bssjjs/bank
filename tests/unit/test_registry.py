"""注册表的单测：状态机、门禁先行、审计先行、回退。

每个用例在 `.tmp/registry/<uuid>` 下造一个假的 serving 目录（只放最小 `bundle.json`），
不碰真实的 `models/serving`；`tempfile.TemporaryDirectory` 在 Windows 沙箱里清理会失败，不用它。
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import registry as REG
from bank import schema as S


def _new_root() -> Path:
    root = S.ROOT / ".tmp" / "registry" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _new_root()
        self.addCleanup(shutil.rmtree, self.root, True)

    def _fake_bundle(self, version: str) -> None:
        target = self.root / version
        target.mkdir(parents=True, exist_ok=True)
        (target / "bundle.json").write_text(
            json.dumps(
                {
                    "contract": "model_bundle.v1",
                    "version": version,
                    "checksum": {"algorithm": "sha256", "value": "a" * 64},
                }
            ),
            encoding="utf-8",
        )

    def _register(self, version: str, *, passed: bool = True) -> None:
        self._fake_bundle(version)
        checks = () if passed else ("AP 0.40 < 0.50",)
        REG.register(
            version,
            run_id="run-" + version,
            metrics={"ap": 0.65, "precision_at_quota": 0.74},
            gate=REG.GateRecord(passed=passed, checks=checks),
            root=self.root,
        )

    def test_register_creates_candidate_and_event(self) -> None:
        self._register("1.0.0")
        self.assertEqual(REG.status("1.0.0", self.root), "candidate")
        self.assertTrue(REG.registry_path(self.root).exists())
        events = REG.events(self.root)
        self.assertEqual([event["event"] for event in events], ["register"])
        self.assertEqual(events[0]["version"], "1.0.0")

    def test_duplicate_version_is_rejected(self) -> None:
        self._register("1.0.0")
        with self.assertRaises(REG.RegistryError):
            self._register("1.0.0")

    def test_register_without_bundle_is_rejected(self) -> None:
        with self.assertRaises(REG.RegistryError):
            REG.register("9.9.9", run_id="x", metrics={}, gate=REG.GateRecord(True), root=self.root)

    def test_promote_auto_stages_then_publishes(self) -> None:
        self._register("1.0.0")
        REG.promote("1.0.0", actor="测试", reason="首版上线", root=self.root)
        self.assertEqual(REG.status("1.0.0", self.root), "production")
        self.assertEqual(REG.production(self.root), "1.0.0")
        self.assertEqual(
            [event["event"] for event in REG.events(self.root)],
            ["register", "stage", "promote"],
        )
        pointer = json.loads((self.root / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(pointer["version"], "1.0.0")

    def test_gate_failure_blocks_promotion_and_leaves_audit_alone(self) -> None:
        self._register("1.1.0", passed=False)
        before = len(REG.events(self.root))
        with self.assertRaises(REG.RegistryError):
            REG.promote("1.1.0", actor="测试", reason="想上线", root=self.root)
        self.assertEqual(REG.status("1.1.0", self.root), "candidate")
        self.assertEqual(len(REG.events(self.root)), before)
        self.assertIsNone(REG.production(self.root))

    def test_illegal_transition_is_rejected(self) -> None:
        self._register("1.0.0")
        with self.assertRaises(REG.RegistryError):
            REG.retire("1.0.0", actor="测试", reason="未上线就退役", root=self.root)

    def test_rollback_restores_previous_version(self) -> None:
        self._register("1.0.0")
        REG.promote("1.0.0", actor="测试", reason="首版", root=self.root)
        self._register("1.1.0")
        REG.promote("1.1.0", actor="测试", reason="新版", root=self.root)
        self.assertEqual(REG.production(self.root), "1.1.0")
        self.assertEqual(REG.status("1.0.0", self.root), "staging")

        REG.rollback("1.0.0", actor="测试", reason="新版指标掉了", root=self.root)
        self.assertEqual(REG.production(self.root), "1.0.0")
        self.assertEqual(REG.status("1.1.0", self.root), "staging")
        self.assertEqual(REG.events(self.root)[-1]["event"], "rollback")

    def test_audit_is_append_only(self) -> None:
        self._register("1.0.0")
        first = REG.audit_path(self.root).read_text(encoding="utf-8")
        REG.promote("1.0.0", actor="测试", reason="上线", root=self.root)
        second = REG.audit_path(self.root).read_text(encoding="utf-8")
        self.assertTrue(second.startswith(first))
        self.assertGreater(len(second.splitlines()), len(first.splitlines()))

    def test_show_lists_versions_and_alias(self) -> None:
        self._register("1.0.0")
        REG.promote("1.0.0", actor="测试", reason="上线", root=self.root)
        text = REG.show(self.root)
        self.assertIn("1.0.0", text)
        self.assertIn("production", text)
        self.assertIn("0.6500", text)

    def test_load_rejects_unknown_contract(self) -> None:
        REG.registry_path(self.root).write_text(
            json.dumps({"contract": "something.v9", "versions": {}, "aliases": {}}),
            encoding="utf-8",
        )
        with self.assertRaises(REG.RegistryError):
            REG.load(self.root)


if __name__ == "__main__":
    unittest.main()
