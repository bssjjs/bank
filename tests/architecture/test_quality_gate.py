"""守门测试：质量门本身不许被悄悄调低。

这里守的是"评判标准"，不是业务行为：
- 五道门还在不在、跑的是不是同一批命令；
- 覆盖率门槛是不是 85（`pyproject.toml` 里那一个数字）；
- ruff / mypy 的严格项有没有被关掉；
- 有没有偷偷写 `# type: ignore`、`# noqa`、`# pragma: no cover` 让检查闭嘴；
- import-linter 的两条契约还在不在，名字对不对；
- 有没有人建 `.github/`（课程规则：CI 只在本机这条命令上）。

改门槛必须同时改这个文件——那就会在 review 里被看见。这是有意的。
"""

from __future__ import annotations

import importlib.util
import io
import tokenize
import tomllib
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GATES = ("ruff check", "ruff format", "mypy", "import-linter", "pytest")


def load_ci() -> ModuleType:
    """把 `scripts/ci.py` 当模块加载：它是脚本，不是包。"""
    path = ROOT / "scripts" / "ci.py"
    spec = importlib.util.spec_from_file_location("ci_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"加载不了 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def tool_section(*keys: str) -> Any:
    node: Any = pyproject()
    for key in keys:
        node = node[key]
    return node


class GatesTest(unittest.TestCase):
    def test_five_gates_in_order(self) -> None:
        ci = load_ci()
        names = tuple(gate.name for gate in ci.gates())
        self.assertEqual(names, EXPECTED_GATES)

    def test_pytest_gate_has_coverage(self) -> None:
        ci = load_ci()
        pytest_gate = next(gate for gate in ci.gates() if gate.name == "pytest")
        self.assertIn("--cov=packages", pytest_gate.cmd)
        self.assertTrue(any(arg.startswith("--cov-report") for arg in pytest_gate.cmd))

    def test_gates_run_through_current_interpreter(self) -> None:
        ci = load_ci()
        for gate in ci.gates():
            self.assertEqual(gate.cmd[0], ci.PY)

    def test_no_gate_silently_ignores_failure(self) -> None:
        # 门命令里出现 `|| true` 之类就等于没门
        ci = load_ci()
        for gate in ci.gates():
            joined = " ".join(gate.cmd)
            self.assertNotIn("||", joined)
            self.assertNotIn("--exit-zero", joined)


class ThresholdTest(unittest.TestCase):
    def test_coverage_threshold_is_85(self) -> None:
        report = tool_section("tool", "coverage", "report")
        assert isinstance(report, dict)
        self.assertEqual(report["fail_under"], 85)

    def test_coverage_source_is_packages(self) -> None:
        run = tool_section("tool", "coverage", "run")
        assert isinstance(run, dict)
        self.assertEqual(run["source"], ["packages"])

    def test_ruff_keeps_core_rules(self) -> None:
        lint = tool_section("tool", "ruff", "lint")
        assert isinstance(lint, dict)
        select = set(lint["select"])
        self.assertTrue({"E", "F", "W", "I", "ISC", "RUF"} <= select, select)
        # 只允许"中文全角标点"这一条有理由的忽略；E/F/W 一条都不许关
        self.assertEqual(set(lint["ignore"]), {"RUF001", "RUF002", "RUF003"})

    def test_mypy_keeps_strict_options(self) -> None:
        mypy = tool_section("tool", "mypy")
        assert isinstance(mypy, dict)
        for option in (
            "check_untyped_defs",
            "no_implicit_optional",
            "warn_redundant_casts",
            "warn_unused_ignores",
        ):
            self.assertIs(mypy[option], True, f"mypy 选项 {option} 被关掉了")
        self.assertEqual(mypy["files"], ["packages"])

    def test_mypy_overrides_only_for_untyped_dependencies(self) -> None:
        overrides = tool_section("tool", "mypy", "overrides")
        assert isinstance(overrides, list)
        modules = sorted(m for entry in overrides for m in entry["module"])
        self.assertEqual(modules, ["joblib.*", "sklearn.*"])

    def test_pytest_has_no_pythonpath_hack(self) -> None:
        options = tool_section("tool", "pytest", "ini_options")
        assert isinstance(options, dict)
        addopts = options["addopts"]
        self.assertNotIn("pythonpath", addopts)
        self.assertIn("-p no:cacheprovider", addopts)
        self.assertEqual(options["testpaths"], ["tests"])


class ContractsTest(unittest.TestCase):
    def test_contract_names_are_the_two_expected(self) -> None:
        text = (ROOT / ".importlinter").read_text(encoding="utf-8")
        self.assertIn("[importlinter:contract:core-independent]", text)
        self.assertIn("[importlinter:contract:training-api-independent]", text)
        self.assertIn("type = forbidden", text)
        self.assertIn("type = independence", text)

    def test_root_packages_cover_three_import_names(self) -> None:
        text = (ROOT / ".importlinter").read_text(encoding="utf-8")
        for name in ("bank_training", "bank_api"):
            self.assertIn(name, text)


class HygieneTest(unittest.TestCase):
    def test_no_suppression_comments_in_source(self) -> None:
        # 只看真正的注释（tokenize），文档字符串里提到这些词不算犯规
        offenders: list[str] = []
        sources = [
            *(ROOT / "packages").rglob("*.py"),
            *(ROOT / "scripts").rglob("*.py"),
            *(ROOT / "tests").rglob("*.py"),
        ]
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type != tokenize.COMMENT:
                    continue
                body = token.string
                if (
                    "type: ignore" in body
                    or body.startswith("# noqa")
                    or "pragma: no cover" in body
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{token.start[0]}: {body.strip()}")
        self.assertEqual(offenders, [], "检查闭嘴的注释会让守门失效，请改代码而不是改检查")

    def test_no_github_directory(self) -> None:
        self.assertFalse((ROOT / ".github").exists(), "课程规则：质量门只在本机这条命令上")

    def test_agents_md_exists(self) -> None:
        self.assertTrue((ROOT / "AGENTS.md").is_file())

    def test_claude_md_just_includes_agents(self) -> None:
        lines = [
            line.strip()
            for line in (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(lines[0], "@AGENTS.md")


if __name__ == "__main__":
    unittest.main()
