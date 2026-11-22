"""冻结：原始库 → 清洗台账 → 派生/标签 → 时间切分 → 两表 + manifest + FROZEN.json。

一条命令：``python -m bank.freeze``；校验：``python -m bank.freeze --check``。
产出（``data/frozen/``，进 Git）：
- ``train.parquet`` / ``valid.parquet``：冻结后的建模两表
- ``manifest.json``：契约①（源文件指纹、两表行数/正例/日期范围、内容指纹、列清单、构建器、配置指纹）
- ``FROZEN.json``：封条（源/配置指纹 + 两表内容指纹 + 各文件 sha256）

设计取舍见 ``docs/adr/0003-指纹与冻结.md``：内容指纹跨机器一致，文件 sha256 只作本地封条。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bank import clean, dataset, raw, repro, split
from bank import config as C
from bank import schema as S
from bank import timeaxis as T

BUILDER = "bank.freeze/1"
CONTRACT_MANIFEST = "manifest.v1"
CONTRACT_FROZEN = "frozen.v1"
FROZEN_FILES = ("train.parquet", "valid.parquet", "manifest.json")

MIN_ROWS, MAX_ROWS, MIN_POSITIVES = 15_000, 30_000, 1_000


@dataclass
class Built:
    manifest: dict
    frozen: dict
    train: pd.DataFrame
    valid: pd.DataFrame
    frozen_dir: Path
    findings: list[clean.Finding]


def build(cfg: C.Config | None = None) -> Built:
    """只算不写：``--check`` 靠它重算后与 FROZEN.json 比对。"""
    cfg = cfg or C.load_config()
    db = S.ROOT / cfg.data.raw_db
    frozen_dir = S.ROOT / cfg.data.frozen_dir

    ordered = raw.read_ordered(cfg.data.raw_table, db)
    df = T.with_time_axis(ordered)

    findings = clean.run_checks(df)
    clean.assert_clean(findings)

    data = dataset.build_dataset(df)
    facts = dataset.validate(data, df)
    train, valid, split_facts = split.split_and_check(data, cfg)

    # 容量与正例：任务书硬性要求，不满足就别往下走
    n_total = len(train) + len(valid)
    n_pos = int(train[S.TARGET].sum()) + int(valid[S.TARGET].sum())
    if not MIN_ROWS <= n_total <= MAX_ROWS:
        raise ValueError(
            f"建模样本 {n_total:,} 行，不在任务书要求的 {MIN_ROWS:,}–{MAX_ROWS:,} 之间"
        )
    if n_pos < MIN_POSITIVES:
        raise ValueError(f"正例 {n_pos:,} 条，少于任务书要求的 {MIN_POSITIVES:,}")
    if len(dataset.model_features(cfg.features.set)) < 10:
        raise ValueError("可用特征少于 10 列")

    train_sha = repro.sha256_table(train)
    valid_sha = repro.sha256_table(valid)
    config_sha = C.config_fingerprint(cfg)
    source_sha = repro.sha256_file(db)

    manifest = {
        "contract": CONTRACT_MANIFEST,
        "builder": BUILDER,
        "config": {"path": _rel(C.CONFIG_PATH), "sha256": config_sha},
        "source": {
            "path": _rel(db),
            "table": cfg.data.raw_table,
            "sha256": source_sha,
            "bytes": db.stat().st_size,
        },
        "raw": {
            "rows": len(ordered),
            "cols": len(ordered.columns) - 1,  # 去掉 序号 后就是原始 17 列
            "positives": facts["n_positive"],
            "pos_rate": facts["pos_rate"],
            "sequence_column": T.SEQUENCE,
            "columns": [c for c in ordered.columns if c != T.SEQUENCE],
        },
        "window": {
            **split_facts["window"],
            "start": str(cfg.window.start),
            "end": str(cfg.window.end),
            "cut": split_facts["cut"],
            "buffer_days": split_facts["buffer_days"],
            "feature_set": cfg.features.set,
        },
        "tables": {
            "train": {
                **split_facts["train"],
                "path": _rel(frozen_dir / "train.parquet"),
                "content_sha256": train_sha,
            },
            "valid": {
                **split_facts["valid"],
                "path": _rel(frozen_dir / "valid.parquet"),
                "content_sha256": valid_sha,
            },
        },
        "capacity": {
            "rows": n_total,
            "positives": n_pos,
            "pos_rate": round(n_pos / n_total, 4),
            "n_features": len(dataset.model_features(cfg.features.set)),
            "quota_n": split.quota_n(len(valid), cfg),
            "quota_frac": cfg.quota.frac,
        },
        "cleaning": [
            {"rule": f.rule, "rows": f.rows, "fatal": f.fatal, "note": f.note} for f in findings
        ],
        "columns": list(train.columns),
        "fingerprint_rule": "canonical csv: lineterminator=LF, float=%.10g, na='' → sha256",
    }

    frozen = {
        "contract": CONTRACT_FROZEN,
        "builder": BUILDER,
        "source": {"path": _rel(db), "sha256": source_sha},
        "config": {"path": _rel(C.CONFIG_PATH), "sha256": config_sha},
        "tables": {
            "train": {
                "rows": len(train),
                "positives": int(train[S.TARGET].sum()),
                "content_sha256": train_sha,
            },
            "valid": {
                "rows": len(valid),
                "positives": int(valid[S.TARGET].sum()),
                "content_sha256": valid_sha,
            },
        },
        "files": {},  # 写入后由 write() 填
    }
    return Built(manifest, frozen, train, valid, frozen_dir, findings)


def write(built: Built) -> list[Path]:
    """按固定顺序落盘。顺序固定是为了 FROZEN.json 能覆盖 manifest.json 的指纹。"""
    frozen_dir = built.frozen_dir
    frozen_dir.mkdir(parents=True, exist_ok=True)

    built.train.to_parquet(frozen_dir / "train.parquet", index=False)
    built.valid.to_parquet(frozen_dir / "valid.parquet", index=False)
    _write_json(frozen_dir / "manifest.json", built.manifest)

    files = {}
    for name in FROZEN_FILES:
        path = frozen_dir / name
        files[name] = {"sha256": repro.sha256_file(path), "bytes": path.stat().st_size}
    built.frozen["files"] = files
    _write_json(frozen_dir / "FROZEN.json", built.frozen)

    return [frozen_dir / name for name in (*FROZEN_FILES, "FROZEN.json")]


def check(cfg: C.Config | None = None) -> int:
    """重算并与 FROZEN.json 比对。退出码 0 = 一致；1 = 有不一致。"""
    cfg = cfg or C.load_config()
    frozen_dir = S.ROOT / cfg.data.frozen_dir
    frozen_path = frozen_dir / "FROZEN.json"
    if not frozen_path.exists():
        print(f"找不到 {frozen_path}，先跑 python -m bank.freeze")
        return 1

    saved = json.loads(frozen_path.read_text(encoding="utf-8"))
    built = build(cfg)
    problems: list[str] = []
    warnings: list[str] = []

    if saved["source"]["sha256"] != built.frozen["source"]["sha256"]:
        problems.append("源文件指纹变了：原始库被换过")
    if saved["config"]["sha256"] != built.frozen["config"]["sha256"]:
        problems.append("配置指纹变了：口径改过，需要重新冻结")
    for name in ("train", "valid"):
        old, new = saved["tables"][name], built.frozen["tables"][name]
        for key in ("rows", "positives", "content_sha256"):
            if old[key] != new[key]:
                problems.append(f"{name}.{key} 不一致：{old[key]} → {new[key]}")

    for name, meta in saved.get("files", {}).items():
        path = frozen_dir / name
        if not path.exists():
            problems.append(f"缺文件：{name}")
            continue
        if repro.sha256_file(path) != meta["sha256"]:
            # parquet 字节会随 pyarrow 版本变化，内容指纹一致就不算失败
            warnings.append(f"文件字节不同（内容指纹一致即可）：{name}")

    for line in warnings:
        print(f"警告：{line}")
    if problems:
        for line in problems:
            print(f"失败：{line}")
        return 1
    print("冻结校验通过：源文件、配置、两表内容指纹全部一致")
    return 0


def summarize(built: Built) -> list[str]:
    m = built.manifest
    tr, va, cap = m["tables"]["train"], m["tables"]["valid"], m["capacity"]
    window = (
        f"窗口      {m['window']['start']} ~ {m['window']['end']}"
        f"    {m['window']['rows']:,} 行 / 正例 {m['window']['positives']:,}"
        f"（{m['window']['pos_rate']:.2%}）"
    )
    train_line = (
        f"训练      {tr['rows']:,} 行 / 正例 {tr['positives']:,}（{tr['pos_rate']:.2%}）"
        f"   {tr['date_min']} ~ {tr['date_max']}"
    )
    valid_line = (
        f"验证      {va['rows']:,} 行 / 正例 {va['positives']:,}（{va['pos_rate']:.2%}）"
        f"   {va['date_min']} ~ {va['date_max']}"
    )
    return [
        "冻结完成",
        "-" * 46,
        f"源文件    {m['source']['path']}  sha256 {m['source']['sha256'][:16]}…",
        f"配置      {m['config']['path']}  sha256 {m['config']['sha256'][:16]}…",
        window,
        train_line,
        valid_line,
        f"名额      {cap['quota_n']} 人（验证集 {va['rows']:,} × {cap['quota_frac']:.0%}）",
        f"内容指纹  train {tr['content_sha256'][:16]}…  valid {va['content_sha256'][:16]}…",
        *clean.summarize(built.findings),
        f"写出      {_rel(built.frozen_dir)}/manifest.json、FROZEN.json",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="冻结原始数据为训练/验证两表并记指纹")
    parser.add_argument("--check", action="store_true", help="只校验，不重写")
    args = parser.parse_args(argv)

    if args.check:
        sys.exit(check())

    built = build()
    write(built)
    for line in summarize(built):
        print(line)


def _write_json(path: Path, obj: dict) -> None:
    """固定格式：2 空格缩进、不转义中文、行尾 LF、末尾一个换行。"""
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(S.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
