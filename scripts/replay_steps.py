"""回放：把每个 `step-*` tag 拿出来，在独立工作目录里装环境、跑一遍质量门。

为什么要有它：一个只有最后一步是好用的仓库，看着也能演示；回放要证明的是
"每一步在当时都能独立通过"——某个 tag 红了，说明历史里有一处当时就没跑通。

用法：

    python scripts/replay_steps.py                 # 回放全部 step-* tag（按名字排序）
    python scripts/replay_steps.py step-M5         # 只回放一个
    python scripts/replay_steps.py --keep          # 保留工作目录，便于进去看现场

每个 tag 都建在 `.tmp/replay/<tag>`（已 gitignore），跑完删掉；环境的 uv 缓存指到
仓库内的 `.venv/.tmp/uv-cache`，避免写用户的全局缓存目录。

原始库 `data/bank-marketing.db` 不进 Git，回放前从主工作区拷进工作目录；冻结集与
模型包由各 tag 自己的命令重建——这一步顺带证明"从零重建"在这个版本上真的可用。

退出码：全过 0；任何一个 tag 红了 1。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".tmp" / "replay"
UV = ROOT / ".venv" / "Scripts" / "uv.exe"
CACHE = ROOT / ".venv" / ".tmp" / "uv-cache"
OK_LINE = "质量门：全绿"
TAIL = 12
DB_NAME = "bank-marketing.db"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(CACHE)
    env["UV_LINK_MODE"] = "copy"
    env["PYTHONUTF8"] = "1"
    return env


def step_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "-l", "step-*"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _python(worktree: Path) -> Path:
    return worktree / ".venv" / "Scripts" / "python.exe"


def _tail(output: str, lines: int = TAIL) -> str:
    text = output.strip().splitlines()
    return "\n".join(text[-lines:]) if text else "(无输出)"


def _seed_data(worktree: Path) -> str | None:
    """把不进 Git 的原始库拷进工作目录；拷不了就返回一句人能看懂的原因。"""
    source = ROOT / "data" / DB_NAME
    target = worktree / "data" / DB_NAME
    if target.is_file():
        return None
    if not source.is_file():
        return f"主工作区缺 data/{DB_NAME}：先把它放回 data/ 再回放"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return None


def _prep_commands(worktree: Path) -> list[list[str]]:
    """这个 tag 自己有的重建命令：先冻结，再导出模型包（没有这步代码就跳过）。"""
    py = str(_python(worktree))
    commands: list[list[str]] = []
    if (worktree / "packages" / "core" / "src" / "bank" / "freeze.py").is_file():
        commands.append([py, "-m", "bank.freeze"])
    if (worktree / "packages" / "training" / "src" / "bank_training" / "export.py").is_file():
        commands.append([py, "-m", "bank_training.export", "--version", "1.0.0", "--promote"])
    return commands


def replay(tag: str) -> tuple[bool, str]:
    """在一个干净的 worktree 里回放一个 tag，返回 (是否通过, 说明)。"""
    worktree = WORK / tag
    if worktree.exists():
        _remove_worktree(tag, worktree)
    add = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), tag],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if add.returncode != 0:
        return False, f"建不了工作目录：{add.stderr.strip()[:200]}"

    try:
        gate_script = worktree / "scripts" / "ci.py"
        if not gate_script.is_file():
            return False, "这个 tag 还没有 scripts/ci.py（早于质量门）"

        sync = subprocess.run(
            [str(UV), "sync", "--all-groups"],
            cwd=worktree,
            env=_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if sync.returncode != 0:
            return False, "装环境失败：" + _tail(sync.stdout + sync.stderr, 5)

        # 模型包与冻结产物不进 Git：按这个 tag 自己的脚本重建，再跑门。
        # 顺带证明"从零重建的命令"在这个版本上可用。
        problem = _seed_data(worktree)
        if problem:
            return False, problem
        for command in _prep_commands(worktree):
            prep = subprocess.run(
                command,
                cwd=worktree,
                env=_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if prep.returncode != 0:
                return False, f"重建失败（{' '.join(command[1:])}）：" + _tail(
                    prep.stdout + prep.stderr, 5
                )

        run = subprocess.run(
            [str(_python(worktree)), "scripts/ci.py"],
            cwd=worktree,
            env=_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        output = (run.stdout + run.stderr).strip().splitlines()
        last = output[-1] if output else "(无输出)"
        passed = run.returncode == 0 and OK_LINE in last
        return passed, last if passed else _tail("\n".join(output))
    finally:
        _remove_worktree(tag, worktree)


def _remove_worktree(tag: str, worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if worktree.exists():  # git 删不掉时（Windows 上 .venv 里有占用）再兜底删目录
        shutil.rmtree(worktree, ignore_errors=True)
    subprocess.run(
        ["git", "worktree", "prune"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 step-* tag 回放质量门")
    parser.add_argument("tags", nargs="*", help="只回放这些 tag（默认全部 step-*）")
    parser.add_argument("--keep", action="store_true", help="保留工作目录，不回删")
    args = parser.parse_args(argv)

    available = step_tags()
    if not available:
        print("没有 step-* tag：先打标签（git tag step-M8 <commit>）")
        return 1
    todo = args.tags or available
    unknown = [tag for tag in todo if tag not in available]
    if unknown:
        print(f"没有这些 tag：{', '.join(unknown)}")
        return 1

    WORK.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    for tag in todo:
        print(f"######## {tag} ########")
        passed, note = replay(tag)
        print(note)
        results.append((tag, passed, note))

    print("\n==== 回放结果 ====")
    for tag, passed, note in results:
        print(f"{'通过' if passed else '失败'} {tag}" + ("" if passed else f"（{note}）"))
    failed = [tag for tag, passed, _ in results if not passed]
    if failed:
        print(f"失败 {len(failed)} 个：{', '.join(failed)}")
        return 1
    print(f"全部通过（{len(results)} 个 tag）")
    if not args.keep:
        shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
