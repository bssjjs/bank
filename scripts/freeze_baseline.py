"""第一轮基线：把跑出来的结果抄一份进 `baselines/`，并给这份基线加文件封条。

为什么要有它：模型卡、README、答辩 PPT 上的每个数字都来自某一次运行。运行会被重跑、
环境会被换掉，所以需要一份**只读的参照**——"第一轮跑通时的结果长什么样"。以后数字对不上，
先和这里比，而不是凭记忆。

用法（两条，别的手改一律视为篡改）：

    python scripts/freeze_baseline.py --write   # 用当前运行产物重写基线（有意为之的更新）
    python scripts/freeze_baseline.py --check   # 校验基线没被改过（质量门/测试里用）

`--write` 做两件事：
1. 从 `data/processed/` 与 `models/serving/` 抄一份快照到 `baselines/v1/`：
   每次运行的指标、E1–E5 实验、冻结账单（FROZEN.json）、当前模型包的 bundle.json；
2. 对 `baselines/v1/` 下所有文件算 sha256，写 `baselines/SHA256SUMS`
   （按路径排序、正斜杠、LF 结尾——换机器也能逐字比对）。

`--check` 只做第 2 步的校验，分三类报：缺失、多出、内容变了。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from bank.repro import sha256_file

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "baselines"
V1 = BASELINE_DIR / "v1"
SUMS = BASELINE_DIR / "SHA256SUMS"
PROCESSED = ROOT / "data" / "processed"
FROZEN = ROOT / "data" / "frozen"
SERVING = ROOT / "models" / "serving"


def _snapshot(source: Path, target: Path) -> None:
    """抄一个 JSON 快照，顺手规范化成 UTF-8 + LF。

    不直接 `copyfile`：mlflow 写出来的 JSON 带 CRLF，而 `.gitattributes` 会在提交时
    把它换成 LF——不规范化的话，封条算的是工作区里的字节，别人 clone 下来就对不上。
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")


def _copy_sources() -> list[str]:
    """把当前运行产物抄进 baselines/v1/，返回抄了哪些文件（相对路径）。"""
    if V1.exists():
        shutil.rmtree(V1)
    (V1 / "runs").mkdir(parents=True)
    copied: list[str] = []

    runs_dir = PROCESSED / "runs"
    # 只抄四个训练 run 的指标（m3-*）；每条实验的逐条结果已汇总进 experiments.json，
    # 而且历史上改过编号，逐条抄会把过期编号一起封进去
    for source in sorted(runs_dir.glob("m3-*.json")):
        target = V1 / "runs" / source.name
        _snapshot(source, target)
        copied.append(str(target.relative_to(ROOT)))

    for source, name in (
        (PROCESSED / "experiments.json", "experiments.json"),
        (FROZEN / "FROZEN.json", "frozen.json"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"缺 {source.relative_to(ROOT)}，先跑出结果再冻结基线")
        target = V1 / name
        _snapshot(source, target)
        copied.append(str(target.relative_to(ROOT)))

    version = json.loads((SERVING / "current.json").read_text(encoding="utf-8"))["version"]
    bundle = SERVING / version / "bundle.json"
    if not bundle.is_file():
        raise FileNotFoundError(f"缺 {bundle.relative_to(ROOT)}，先导出模型包再冻结基线")
    target = V1 / f"bundle-{version}.json"
    _snapshot(bundle, target)
    copied.append(str(target.relative_to(ROOT)))
    return copied


def _baseline_files() -> list[Path]:
    return sorted(p for p in V1.rglob("*") if p.is_file())


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write() -> int:
    copied = _copy_sources()
    lines = [f"{sha256_file(path)}  {_relative(path)}" for path in _baseline_files()]
    SUMS.parent.mkdir(parents=True, exist_ok=True)
    SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"基线已写入：{len(copied)} 个快照文件，{len(lines)} 行封条 → {_relative(SUMS)}")
    for rel in copied:
        print(f"  + {rel}")
    return 0


def _read_sums() -> list[tuple[str, str]]:
    text = SUMS.read_text(encoding="utf-8")
    rows: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{_relative(SUMS)} 第 {number} 行格式不对：{line!r}")
        rows.append((parts[1], parts[0]))
    return rows


def check() -> int:
    if not SUMS.is_file():
        print(f"缺 {_relative(SUMS)}：先跑 --write", file=sys.stderr)
        return 1
    listed = _read_sums()
    expected = dict(listed)
    missing: list[str] = []
    changed: list[str] = []
    for rel, digest in listed:
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != digest:
            changed.append(rel)
    extra = sorted(_relative(p) for p in _baseline_files() if _relative(p) not in expected)

    if not missing and not changed and not extra:
        print(f"基线校验通过：{len(listed)} 个文件，内容与 {_relative(SUMS)} 一致")
        return 0
    for rel in missing:
        print(f"缺失：{rel}")
    for rel in changed:
        print(f"内容变了：{rel}")
    for rel in extra:
        print(f"多出（不在封条里）：{rel}")
    print("基线校验未通过：基线是只读参照，要更新请跑 --write 并说明为什么")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基线快照与文件封条")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="用当前运行产物重写基线")
    group.add_argument("--check", action="store_true", help="校验基线没被改过")
    args = parser.parse_args(argv)
    return write() if args.write else check()


if __name__ == "__main__":
    raise SystemExit(main())
