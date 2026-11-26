"""对账点 1 的验收测试：一致时退出 0，账单一改就退出 1，且不回写冻结数据。

用合成库跑，不读 `data/bank-marketing.db`；否则干净检出（任务书要求不交数据）
会因为缺库而红。
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from bank import freeze, reconcile, repro

from tests.support import scratch_dir, synthetic_config, write_synthetic_db


class SyntheticReconcileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = scratch_dir("reconcile")
        cls.db = write_synthetic_db(cls.root / "synthetic.db", rows=20_000, per_day=22)
        cls.frozen_dir = cls.root / "frozen"
        cls.cfg = synthetic_config(cls.db, cls.frozen_dir)
        cls.built = freeze.build(cls.cfg)
        freeze.write(cls.built)

    def _copied_bill(self) -> Path:
        """把账单抄一份到临时目录，改坏了也不影响本测试的冻结目录。"""
        target = scratch_dir("reconcile-bill")
        for name in ("manifest.json", "FROZEN.json"):
            shutil.copy2(self.frozen_dir / name, target / name)
        return target

    def test_synthetic_frozen_dir_reconciles(self) -> None:
        self.assertEqual(reconcile.reconcile(self.frozen_dir, self.cfg), 0)

    def test_reconcile_does_not_touch_frozen(self) -> None:
        before = repro.sha256_file(self.frozen_dir / "FROZEN.json")
        reconcile.reconcile(self.frozen_dir, self.cfg)
        self.assertEqual(before, repro.sha256_file(self.frozen_dir / "FROZEN.json"))

    def test_items_cover_the_required_checkpoints(self) -> None:
        manifest, frozen, _ = reconcile.load_bill(self.frozen_dir)
        items = reconcile.expected_items(manifest)
        names = {item.name for item in items}
        for required in (
            "raw.rows",
            "raw.positives",
            "window.rows",
            "train.rows",
            "train.positives",
            "train.content_sha256",
            "valid.rows",
            "valid.positives",
            "valid.content_sha256",
            "capacity.quota_n",
        ):
            self.assertIn(required, names)
        self.assertGreaterEqual(len(items), 15)
        self.assertEqual(frozen["contract"], "frozen.v1")

    def test_train_rows_tampered(self) -> None:
        target = self._copied_bill()
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["tables"]["train"]["rows"] += 1
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target, self.cfg), 1)

    def test_content_fingerprint_tampered(self) -> None:
        target = self._copied_bill()
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["tables"]["valid"]["content_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target, self.cfg), 1)

    def test_source_fingerprint_tampered(self) -> None:
        target = self._copied_bill()
        path = target / "FROZEN.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["source"]["sha256"] = "f" * 64
        path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target, self.cfg), 1)

    def test_missing_bill(self) -> None:
        self.assertEqual(reconcile.reconcile(scratch_dir("reconcile-empty"), self.cfg), 1)


if __name__ == "__main__":
    unittest.main()
