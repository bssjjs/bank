"""一条命令跑完质量门：`python scripts/ci.py`。

五道门（任何一道红，整条命令就红）：
1. `ruff check`　——静态风格与疑似 bug；
2. `ruff format --check`　——格式必须与 formatter 的结论一致，评审不再讨论空格；
3. `mypy`　——类型；
4. `lint-imports`　——`core` 谁都不依赖、`training` 与 `api` 互不依赖；
5. `pytest`　——单测 + 架构测试 + 契约测试。

约定：工具一律用当前解释器的 `-m`（`sys.executable`），避免"命令行里那个 ruff 是另一个版本"。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class Gate(NamedTuple):
    name: str
    cmd: list[str]
    why: str


GATES = (
    Gate("ruff check", [PY, "-m", "ruff", "check", "."], "风格与疑似 bug"),
    Gate("ruff format", [PY, "-m", "ruff", "format", "--check", "."], "格式一致"),
    Gate("mypy", [PY, "-m", "mypy"], "类型"),
    Gate("import-linter", [PY, "-m", "importlinter.cli", "lint", "--no-logo"], "依赖方向"),
    Gate("pytest", [PY, "-m", "pytest", "tests"], "行为与架构断言"),
)


def _fmt_cmd(cmd: list[str]) -> str:
    tail = [c.replace(str(ROOT), ".") for c in cmd[1:]]
    return "python " + " ".join(tail)


def run(gate: Gate) -> tuple[bool, str]:
    result = subprocess.run(gate.cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = (result.stdout + result.stderr).strip()
    tail = "\n".join(output.splitlines()[-20:]) if output else "(无输出)"
    return result.returncode == 0, tail


def main() -> int:
    print(f"质量门：{len(GATES)} 道，工作目录 {ROOT}")
    failed: list[str] = []
    for gate in GATES:
        ok, tail = run(gate)
        print(f"\n[{('通过' if ok else '未通过')}] {gate.name}（{gate.why}）　{_fmt_cmd(gate.cmd)}")
        if not ok:
            failed.append(gate.name)
            print(tail)
    print()
    if failed:
        print(f"质量门：未通过（{', '.join(failed)}）")
        return 1
    print("质量门：全绿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
