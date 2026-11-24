"""对账点 1 的验收测试：一致时退出 0，账单一改就退出 1，且不回写冻结数据。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import reconcile, repro
from bank import schema as S

FROZEN_DIR = S.ROOT / "data" / "frozen"


def _scratch() -> Path:
    """仓库内建的临时目录：Windows 沙箱里 tempfile 清理会撞 ACL。"""
    path = S.ROOT / ".tmp" / "reconcile" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copied_bill() -> Path:
    """把真账单抄一份到临时目录，改坏了也不影响 data/frozen。"""
    target = _scratch()
    for name in ("manifest.json", "FROZEN.json"):
        shutil.copy2(FROZEN_DIR / name, target / name)
    return target


class RealDataTest(unittest.TestCase):
    def test_real_frozen_dir_reconciles(self) -> None:
        self.assertEqual(reconcile.reconcile(), 0)

    def test_reconcile_does_not_touch_frozen(self) -> None:
        before = repro.sha256_file(FROZEN_DIR / "FROZEN.json")
        reconcile.reconcile()
        self.assertEqual(before, repro.sha256_file(FROZEN_DIR / "FROZEN.json"))

    def test_items_cover_the_required_checkpoints(self) -> None:
        manifest, frozen, _ = reconcile.load_bill()
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


class TamperedBillTest(unittest.TestCase):
    """对账必须能失败：账单一改，退出码就得是 1。"""

    def test_train_rows_tampered(self) -> None:
        target = _copied_bill()
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["tables"]["train"]["rows"] += 1
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target), 1)

    def test_content_fingerprint_tampered(self) -> None:
        target = _copied_bill()
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["tables"]["valid"]["content_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target), 1)

    def test_source_fingerprint_tampered(self) -> None:
        target = _copied_bill()
        path = target / "FROZEN.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["source"]["sha256"] = "f" * 64
        path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(reconcile.reconcile(target), 1)

    def test_missing_bill(self) -> None:
        self.assertEqual(reconcile.reconcile(_scratch()), 1)


if __name__ == "__main__":
    unittest.main()
