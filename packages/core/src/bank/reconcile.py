"""对账点 1：从原始库重算一遍，和冻结时的账单逐项比。

一条命令：``python -m bank.reconcile``。退出码 0 = 全部一致；1 = 有不一致。

和 ``bank.freeze --check`` 的区别（两条命令都要留着）：

- ``freeze --check``：**重跑同一套代码**，确认「今天跑出来的两表」和「冻结时的两表」一致，
  回答的是「口径有没有漂、库有没有被换」。
- ``reconcile``：**不读冻结的 parquet**，直接从原始库重切一遍，再和 ``manifest.json``
  里记下的账单逐项比（源文件指纹、配置指纹、原始表行数/正例、窗口行数/正例/日期范围、
  两张表的行数/正例/内容指纹/日期范围、列清单、名额）。回答的是「账单和重建结果对不对得上」。

对账**必须能失败**：任何一项不一致就打印差在哪、退出码 1，并且绝不回写 ``data/frozen``。
测试里有一条专门把账单改坏、确认它会红（``tests/unit/test_reconcile.py``）。

为什么不比 parquet 的文件字节（讲义 K 系列问题里的原话）：parquet 字节会随 pyarrow 版本变，
比它等于把「换台机器」当成「数据被改了」；所以比的是内容指纹（规范化 CSV 的 sha256）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from bank import config as C
from bank import dataset, freeze
from bank import schema as S

CONTRACT = "reconcile.v1"
MANIFEST = "manifest.json"
FROZEN = "FROZEN.json"


@dataclass(frozen=True)
class Item:
    """一项对账：名字、账单里的值、重算出来的值、是否一致。"""

    name: str
    expected: object
    actual: object

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


def load_bill(frozen_dir: Path | None = None) -> tuple[dict, dict, Path]:
    """读冻结时的账单；缺文件就抛 FileNotFoundError 让调用方给出人话。"""
    directory = frozen_dir or S.ROOT / C.load_config().data.frozen_dir
    manifest_path = directory / MANIFEST
    frozen_path = directory / FROZEN
    if not manifest_path.exists() or not frozen_path.exists():
        raise FileNotFoundError(
            f"{directory} 下缺少 {MANIFEST} 或 {FROZEN}，先跑 python -m bank.freeze"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    return manifest, frozen, directory


def expected_items(manifest: dict) -> list[Item]:
    """把账单摊平成对账项。重算值另外填。"""
    caps = manifest["capacity"]
    window = manifest["window"]
    tr, va = manifest["tables"]["train"], manifest["tables"]["valid"]
    return [
        Item("raw.rows", manifest["raw"]["rows"], None),
        Item("raw.positives", manifest["raw"]["positives"], None),
        Item("raw.columns", tuple(manifest["raw"]["columns"]), None),
        Item("window.rows", window["rows"], None),
        Item("window.positives", window["positives"], None),
        Item("window.range", (window["start"], window["end"]), None),
        Item("train.rows", tr["rows"], None),
        Item("train.positives", tr["positives"], None),
        Item("train.content_sha256", tr["content_sha256"], None),
        Item("train.range", (tr["date_min"], tr["date_max"]), None),
        Item("valid.rows", va["rows"], None),
        Item("valid.positives", va["positives"], None),
        Item("valid.content_sha256", va["content_sha256"], None),
        Item("valid.range", (va["date_min"], va["date_max"]), None),
        Item("capacity.quota_n", caps["quota_n"], None),
        Item("capacity.n_features", caps["n_features"], None),
    ]


def actual_items(built: freeze.Built) -> dict[str, object]:
    """从重算结果里取出和账单同名的值。"""
    m = built.manifest
    tr, va = m["tables"]["train"], m["tables"]["valid"]
    return {
        "raw.rows": m["raw"]["rows"],
        "raw.positives": m["raw"]["positives"],
        "raw.columns": tuple(m["raw"]["columns"]),
        "window.rows": m["window"]["rows"],
        "window.positives": m["window"]["positives"],
        "window.range": (m["window"]["start"], m["window"]["end"]),
        "train.rows": tr["rows"],
        "train.positives": tr["positives"],
        "train.content_sha256": tr["content_sha256"],
        "train.range": (tr["date_min"], tr["date_max"]),
        "valid.rows": va["rows"],
        "valid.positives": va["positives"],
        "valid.content_sha256": va["content_sha256"],
        "valid.range": (va["date_min"], va["date_max"]),
        "capacity.quota_n": m["capacity"]["quota_n"],
        "capacity.n_features": m["capacity"]["n_features"],
    }


def compare(manifest: dict, built: freeze.Built) -> list[Item]:
    actual = actual_items(built)
    return [Item(item.name, item.expected, actual[item.name]) for item in expected_items(manifest)]


def source_and_config(
    manifest: dict, frozen: dict, built: freeze.Built, cfg: C.Config
) -> list[Item]:
    """源文件与配置：两边都要对得上（manifest 与 FROZEN 也得彼此一致）。"""
    db = S.ROOT / cfg.data.raw_db
    actual_source = built.manifest["source"]["sha256"]
    actual_config = C.config_fingerprint(cfg)
    return [
        Item("source.sha256", manifest["source"]["sha256"], actual_source),
        Item("source.sha256(frozen)", frozen["source"]["sha256"], actual_source),
        Item("source.exists", True, db.exists()),
        Item("config.sha256", manifest["config"]["sha256"], actual_config),
        Item("config.sha256(frozen)", frozen["config"]["sha256"], actual_config),
        Item("features.agrees_with_catalog", True, _catalog_agrees(cfg)),
    ]


def _catalog_agrees(cfg: C.Config) -> bool:
    """特征目录要覆盖当前特征集，且三条禁用列仍然禁用；对账时顺手再守一次。

    目录登记的是全集（特征集 A，24 列）；配置里的特征集 B 是它的子集（去掉月份三列），
    所以这里比的是「子集关系」，不是「集合相等」。
    """
    from bank import catalog

    spec = catalog.load_catalog()
    used = set(dataset.model_features(cfg.features.set))
    declared = set(spec.feature_names())
    forbidden = set(spec.forbidden_names())
    return used <= declared and set(catalog.REQUIRED_FORBIDDEN) <= forbidden


def reconcile(frozen_dir: Path | None = None, cfg: C.Config | None = None) -> int:
    """重算 + 逐项比。不写任何文件。"""
    cfg = cfg or C.load_config()
    try:
        manifest, frozen, directory = load_bill(frozen_dir)
    except FileNotFoundError as exc:
        print(f"失败：{exc}")
        return 1

    # 只读原始库、只读配置；冻结的 parquet 不参与
    built = freeze.build(cfg)
    items = source_and_config(manifest, frozen, built, cfg) + compare(manifest, built)

    bad = [item for item in items if not item.ok]
    print(f"对账 {len(items)} 项（账单：{directory.as_posix()}）")
    print("-" * 60)
    for item in items:
        mark = "一致" if item.ok else "不一致"
        if item.ok:
            print(f"[{mark}] {item.name:<28} {_show(item.actual)}")
        else:
            print(
                f"[{mark}] {item.name:<28} 账单 {_show(item.expected)} → 重算 {_show(item.actual)}"
            )

    if bad:
        print("-" * 60)
        print(f"失败：{len(bad)} 项对不上，冻结数据没有改动。先看是哪一项：")
        for item in bad:
            print(f"  - {item.name}")
        return 1

    print("-" * 60)
    print("对账通过：从原始库重建的两表与冻结账单逐项一致")
    return 0


def _show(value: object) -> str:
    """长指纹只显示前 16 位，日期范围/列清单照原样。"""
    if isinstance(value, str) and len(value) == 64:
        return f"{value[:16]}…"
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, str) for v in value):
        return f"{value[0]} ~ {value[1]}"
    if isinstance(value, tuple):
        return f"{len(value)} 列" if len(value) > 6 else " ".join(map(str, value))
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="从原始库重算并与冻结账单逐项对账")
    parser.add_argument("--frozen-dir", type=Path, default=None, help="冻结目录，默认取配置里的")
    args = parser.parse_args(argv)
    sys.exit(reconcile(args.frozen_dir))


if __name__ == "__main__":
    main()
