"""架构测试：用断言守住"依赖方向"和"质量门存在"，而不是靠人记得。"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKGS = ROOT / "packages"
CORE = PKGS / "core" / "src" / "bank"
TRAINING = PKGS / "training" / "src" / "bank_training"
API = PKGS / "api" / "src" / "bank_api"


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


class LayoutTest(unittest.TestCase):
    def test_three_packages_exist(self) -> None:
        for pkg in ("core", "training", "api"):
            self.assertTrue((PKGS / pkg / "pyproject.toml").exists(), f"缺 packages/{pkg}")

    def test_workspace_declares_members(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for member in ("packages/core", "packages/training", "packages/api"):
            self.assertIn(member, text)

    def test_entry_files_exist(self) -> None:
        for path in (
            "configs/base.yaml",
            "configs/features.yaml",
            "scripts/ci.py",
            ".importlinter",
        ):
            self.assertTrue((ROOT / path).exists(), f"缺 {path}")


class DependencyDirectionTest(unittest.TestCase):
    def test_core_does_not_import_training_or_api(self) -> None:
        offenders = []
        for path in CORE.rglob("*.py"):
            bad = imported_roots(path) & {"bank_training", "bank_api"}
            if bad:
                offenders.append(f"{path.relative_to(ROOT)} -> {sorted(bad)}")
        self.assertEqual(offenders, [], "core 必须谁都不依赖")

    def test_training_and_api_do_not_import_each_other(self) -> None:
        for side, other in ((TRAINING, "bank_api"), (API, "bank_training")):
            for path in side.rglob("*.py"):
                self.assertNotIn(
                    other, imported_roots(path), f"{path.relative_to(ROOT)} 反向依赖了 {other}"
                )

    def test_no_sys_path_hacks(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in list(PKGS.rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
            if "sys.path" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [], "不要用 sys.path 改导入路径，靠包安装解决")


class QualityGateTest(unittest.TestCase):
    def test_ci_has_five_gates(self) -> None:
        text = (ROOT / "scripts" / "ci.py").read_text(encoding="utf-8")
        for name in ('ruff", "check', 'ruff", "format', "mypy", "importlinter", "pytest"):
            self.assertIn(name, text)

    def test_ci_uses_current_interpreter(self) -> None:
        text = (ROOT / "scripts" / "ci.py").read_text(encoding="utf-8")
        self.assertIn("sys.executable", text)

    def test_ci_prints_single_conclusion(self) -> None:
        text = (ROOT / "scripts" / "ci.py").read_text(encoding="utf-8")
        self.assertIn("质量门：全绿", text)
        self.assertIn("质量门：未通过", text)

    def test_importlinter_has_two_contracts(self) -> None:
        text = (ROOT / ".importlinter").read_text(encoding="utf-8")
        self.assertIn("type = forbidden", text)
        self.assertIn("type = independence", text)


if __name__ == "__main__":
    unittest.main()
