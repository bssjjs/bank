"""模型注册表：版本、别名、状态机、审计日志。

四个状态、四条合法迁移：

```
candidate ──► staging ──► production ──► retired
                                          │
                 ▲   （retired 可以退回 staging）
                 └────────────────────────┘
```

规则（讲义 D-L2 与 D-L3）：

- **门禁先行**：只有 `gate.passed` 为真的版本才能进 `production`；门禁没过的版本也登记、也能查，
  但不许发布。
- **审计先行**：每一次状态变化，先把事件写进 `audit.jsonl` 并 `fsync`，再改 `registry.json`。
  反过来的顺序一旦中途崩掉，就会出现「状态变了但没有记录」——审计就不算数了。
- **别名而不是删除**：发布 = 把 `aliases.production` 指向某版本、并同步 `current.json`；
  回退 = 指回旧版本。旧版本目录永远不删、不覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bank import bundle as B

CONTRACT = "model_registry.v1"
REGISTRY_FILE = "registry.json"
AUDIT_FILE = "audit.jsonl"
SERVING_DIR = B.SERVING_DIR

STATES = ("candidate", "staging", "production", "retired")
TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"staging"},
    "staging": {"production", "retired"},
    "production": {"staging", "retired"},
    "retired": {"staging"},
}


class RegistryError(RuntimeError):
    """注册表操作被拒绝：版本不存在、状态迁移不合法、门禁没过。"""


@dataclass(frozen=True)
class GateRecord:
    """发布门禁的结果：过了、以及没过时卡在哪几条。"""

    passed: bool
    checks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": list(self.checks)}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def registry_path(root: Path = SERVING_DIR) -> Path:
    return root / REGISTRY_FILE


def audit_path(root: Path = SERVING_DIR) -> Path:
    return root / AUDIT_FILE


def _empty() -> dict[str, Any]:
    return {"contract": CONTRACT, "updated_at": "", "versions": {}, "aliases": {}}


def load(root: Path = SERVING_DIR) -> dict[str, Any]:
    """读注册表；文件不存在时返回空骨架（第一次登记会自动建）。"""
    path = registry_path(root)
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RegistryError(f"{path} 不是 JSON 对象")
    if data.get("contract") != CONTRACT:
        raise RegistryError(f"{path} 的 contract 不是 {CONTRACT}：{data.get('contract')!r}")
    versions = data.get("versions")
    if not isinstance(versions, dict):
        raise RegistryError(f"{path} 缺 versions")
    for name, entry in versions.items():
        state = entry.get("status") if isinstance(entry, dict) else None
        if state not in STATES:
            raise RegistryError(f"版本 {name} 的状态 {state!r} 不在 {STATES}")
    return data


def _save(data: dict[str, Any], root: Path = SERVING_DIR) -> Path:
    data["updated_at"] = _now()
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _record(
    event: str,
    version: str,
    *,
    actor: str,
    reason: str,
    root: Path,
    detail: dict[str, Any] | None = None,
) -> None:
    """把事件追加进审计日志并落盘；必须在改注册表**之前**调用。"""
    path = audit_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "ts": _now(),
            "event": event,
            "version": version,
            "actor": actor,
            "reason": reason,
            "detail": detail or {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def events(root: Path = SERVING_DIR) -> list[dict[str, Any]]:
    """读全部审计事件（追加写，旧行不会被改写）。"""
    path = audit_path(root)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path} 第 {index} 行不是合法 JSON：{exc}") from exc
    return rows


def _checksum(root: Path, version: str) -> str:
    doc = json.loads((root / version / B.BUNDLE_JSON).read_text(encoding="utf-8"))
    return str(doc.get("checksum", {}).get("value", ""))


def register(
    version: str,
    *,
    run_id: str,
    metrics: dict[str, float],
    gate: GateRecord,
    note: str = "",
    root: Path = SERVING_DIR,
    actor: str = "bank_training.export",
) -> dict[str, Any]:
    """登记一个新版本（状态 `candidate`），门禁结果一并存档。"""
    if not version:
        raise RegistryError("版本号不能为空")
    data = load(root)
    if version in data["versions"]:
        raise RegistryError(f"版本 {version} 已经登记过；模型包与注册表都不覆盖旧版本")
    if not (root / version / B.BUNDLE_JSON).exists():
        raise RegistryError(f"版本 {version} 没有 bundle.json：{root / version}")

    _record(
        "register",
        version,
        actor=actor,
        reason=note or "导出模型包",
        root=root,
        detail={"run_id": run_id, "gate": gate.as_dict()},
    )
    data["versions"][version] = {
        "status": "candidate",
        "registered_at": _now(),
        "run_id": run_id,
        "metrics": dict(metrics),
        "gate": gate.as_dict(),
        "note": note,
        "bundle_checksum": _checksum(root, version),
    }
    _save(data, root)
    return data


def status(version: str, root: Path = SERVING_DIR) -> str:
    data = load(root)
    entry = data["versions"].get(version)
    if entry is None:
        raise RegistryError(f"版本 {version} 没有登记")
    return str(entry["status"])


def _move(
    version: str,
    to_state: str,
    *,
    event: str,
    actor: str,
    reason: str,
    root: Path,
    check_gate: bool = False,
) -> dict[str, Any]:
    data = load(root)
    entry = data["versions"].get(version)
    if entry is None:
        raise RegistryError(f"版本 {version} 没有登记；先跑 export 再发布")
    current = str(entry["status"])
    if to_state not in TRANSITIONS.get(current, set()):
        raise RegistryError(f"版本 {version} 不能从 {current} 变到 {to_state}")
    if check_gate and not entry["gate"].get("passed"):
        raise RegistryError(f"版本 {version} 的门禁没过（{entry['gate'].get('checks')}），不能发布")

    demoted = [
        name
        for name, other in data["versions"].items()
        if other["status"] == "production" and name != version
    ]
    detail: dict[str, Any] = {"from": current, "to": to_state}
    if demoted:
        detail["demoted"] = demoted

    # 审计先行：事件落盘之后才允许改状态
    _record(event, version, actor=actor, reason=reason, root=root, detail=detail)

    for name in demoted:
        data["versions"][name]["status"] = "staging"
        data["versions"][name]["demoted_at"] = _now()
    data["versions"][version]["status"] = to_state
    data["versions"][version][f"{to_state}_at"] = _now()
    if to_state == "production":
        data["aliases"]["production"] = version
    _save(data, root)
    if to_state == "production":
        B.write_current(version, root)
    return data


def stage(version: str, *, actor: str, reason: str, root: Path = SERVING_DIR) -> dict[str, Any]:
    """候选 → 待发布（预发/演练），不改线上指针。"""
    return _move(version, "staging", event="stage", actor=actor, reason=reason, root=root)


def _require_gate(version: str, root: Path) -> dict[str, Any]:
    """门禁先行：不许先迁状态再发现门禁没过，那会留下一条查不清的事件。"""
    entry = load(root)["versions"].get(version)
    if entry is None:
        raise RegistryError(f"版本 {version} 没有登记；先跑 export 再发布")
    if not entry["gate"].get("passed"):
        raise RegistryError(f"版本 {version} 的门禁没过（{entry['gate'].get('checks')}），不能发布")
    return entry


def promote(version: str, *, actor: str, reason: str, root: Path = SERVING_DIR) -> dict[str, Any]:
    """待发布 → 线上；门禁没过的版本在这里被拦下。

    顺序是「先查门禁，再迁状态」：门禁没过就一步都不动，审计里也不会多一行。
    刚登记、还停在 `candidate` 的版本会自动走一步 `stage`（并留下审计事件），
    省得每次发布都要手动敲两次命令。
    """
    entry = _require_gate(version, root)
    if entry["status"] == "candidate":
        _move(version, "staging", event="stage", actor=actor, reason="发布前自动过预发", root=root)
    return _move(
        version,
        "production",
        event="promote",
        actor=actor,
        reason=reason,
        root=root,
        check_gate=True,
    )


def rollback(version: str, *, actor: str, reason: str, root: Path = SERVING_DIR) -> dict[str, Any]:
    """回退：把线上指针指回旧版本（旧版本必须还在 `staging` 或 `retired`）。"""
    data = load(root)
    entry = data["versions"].get(version)
    if entry is None:
        raise RegistryError(f"版本 {version} 没有登记")
    if entry["status"] == "production":
        raise RegistryError(f"版本 {version} 已经在线上了")
    return _move(
        version,
        "production",
        event="rollback",
        actor=actor,
        reason=reason,
        root=root,
        check_gate=True,
    )


def retire(version: str, *, actor: str, reason: str, root: Path = SERVING_DIR) -> dict[str, Any]:
    return _move(version, "retired", event="retire", actor=actor, reason=reason, root=root)


def production(root: Path = SERVING_DIR) -> str | None:
    return load(root)["aliases"].get("production")


def show(root: Path = SERVING_DIR) -> str:
    """人类可读的一张表：版本、状态、门禁、指标、别名。"""
    data = load(root)
    lines = [
        f"注册表 {registry_path(root)}",
        f"契约 {data['contract']}，更新于 {data['updated_at'] or '（空）'}",
        "",
        "| 版本 | 状态 | 门禁 | AP | 名额内精确率 | 别名 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    aliases: dict[str, list[str]] = {}
    for alias, version in data["aliases"].items():
        aliases.setdefault(version, []).append(alias)
    for version in sorted(data["versions"]):
        entry = data["versions"][version]
        gate = "过" if entry["gate"].get("passed") else "未过"
        metrics = entry.get("metrics", {})
        lines.append(
            f"| {version} | {entry['status']} | {gate} | "
            f"{float(metrics.get('ap', 0.0)):.4f} | "
            f"{float(metrics.get('precision_at_quota', 0.0)):.4f} | "
            f"{'、'.join(aliases.get(version, [])) or '—'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模型注册表：查看 / 晋升 / 回退")
    parser.add_argument("--actor", default="负责人", help="谁做的这次操作（写进审计）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="看注册表")
    for name in ("stage", "promote", "rollback", "retire"):
        item = sub.add_parser(name, help=f"{name} 一个版本")
        item.add_argument("version")
        item.add_argument("--reason", required=True, help="为什么（写进审计，必填）")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "show":
            print(show())
            print(f"审计事件：{len(events())} 条（{audit_path()}）")
            return 0
        actions = {"stage": stage, "promote": promote, "rollback": rollback, "retire": retire}
        data = actions[args.cmd](args.version, actor=args.actor, reason=args.reason)
        entry = data["versions"][args.version]
        print(
            f"{args.cmd} {args.version}：状态 {entry['status']}，"
            f"线上 {production()}，审计事件 {len(events())} 条"
        )
        return 0
    except RegistryError as exc:
        print(f"[注册表拒绝] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
