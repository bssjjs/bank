"""预测记录 v2：服务写、回流读，两边共用同一份字段定义。

v1 的记录只有 `{ts, request_id, model_version, score, level, control}`，够用来做"今天打了几个电话"，
但不够做回流：要接住"这个 request_id 后来到底买没买"，得有一个能对上号、又**不存客户特征值**的键。
v2 加了三样：

- `contract`：记录格式自己带版本号，回流读到不认识的版本就拒绝，不猜。
- `features_hash`：请求输入（15 个必填字段）规范化 JSON 的 sha256。同样一组输入哈希相同，
  所以能用来去重、对账、发现"同一个人被打了三次"；但哈希不可逆，日志里仍然没有特征值。
- `quota_frac` / `quota_n`：这一条是在什么名额口径下打的等级，回看历史决策时不用再去翻模型包。

对照组（`control=True`）是 F-L2 的要求：按 request_id 哈希做**确定性**抽样，默认 10%。
确定性意味着重放同一批请求得到同一批对照组，不用额外存随机数；对照组不进重训样本（见 ADR 0008）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from bank import repro

CONTRACT = "prediction_record.v2"
CONTROL_RATE = 0.10
CONTROL_SEED = 20261122


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def is_control(request_id: str, *, rate: float = CONTROL_RATE, seed: int = CONTROL_SEED) -> bool:
    """按 request_id 做确定性抽样：同一个请求号永远得到同一个答案。"""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"对照组比例必须在 [0, 1] 里，收到 {rate}")
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    return value < rate


def features_hash(payload: dict[str, Any]) -> str:
    """输入指纹：只留哈希，不留值。键序不影响结果。"""
    return repro.sha256_json({str(key): str(value) for key, value in payload.items()})


def record(
    *,
    request_id: str,
    model_version: str,
    score: float,
    level: str,
    control: bool,
    payload: dict[str, Any],
    quota_frac: float,
    quota_n: int,
) -> dict[str, Any]:
    return {
        "ts": _now(),
        "contract": CONTRACT,
        "request_id": request_id,
        "model_version": model_version,
        "score": round(float(score), 6),
        "level": level,
        "control": bool(control),
        "features_hash": features_hash(payload),
        "quota_frac": float(quota_frac),
        "quota_n": int(quota_n),
    }


def append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {index} 行不是合法 JSON：{exc}") from exc
    return rows


def by_request_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """同一个 request_id 只留最后一条（重试会写多行）。"""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("request_id", ""))
        if rid:
            latest[rid] = row
    return latest
