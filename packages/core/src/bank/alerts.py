"""告警：级别、告警对象与三种投递渠道。

设计取舍：
- 渠道失败**不算监控失败**：投递失败本身变成一条 warn 级告警（`channel/webhook_failed`），
  否则「告警发不出去」这件事就永远没人知道。
- 渠道是"把一批告警送出去"，不是"每来一条发一次"：定时任务一次跑完只发一批，
  避免一条 PSI 告警把人的手机震十下。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

LEVELS = ("info", "warn", "alert")
SEVERITY = {"info": 0, "warn": 1, "alert": 2}


def now() -> str:
    """统一的时间戳：UTC、秒级、带 Z，便于跨机器比对。"""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Alert:
    """一条可以被人读、也可以被机器筛的告警。"""

    level: str
    source: str
    code: str
    message: str
    value: float | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"未知告警级别 {self.level!r}：只能是 {LEVELS}")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": self.level,
            "source": self.source,
            "code": self.code,
            "message": self.message,
        }
        if self.value is not None:
            payload["value"] = round(float(self.value), 6)
        return payload


class Channel(Protocol):
    """投递渠道：送出去，返回送不出去的那些（转成告警）。"""

    name: str

    def send(self, alerts: list[Alert]) -> list[Alert]: ...


class ConsoleChannel:
    """打到控制台。定时任务在终端里跑的时候，这是唯一不需要额外设施的路。"""

    name = "console"

    def send(self, alerts: list[Alert]) -> list[Alert]:
        for alert in alerts:
            print(f"[{alert.level}] {alert.source}/{alert.code}: {alert.message}")
        return []


class FileChannel:
    """追加到 JSONL。每行一条、带时间戳，坏了只坏一行，方便人用 grep 查。"""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, alerts: list[Alert]) -> list[Alert]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for alert in alerts:
                entry = {"ts": now(), **alert.as_dict()}
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return []


class WebhookChannel:
    """POST 一个 JSON 到外部地址（钉钉/企业微信这类转发器由对方接）。

    超时 3 秒：监控的定时任务不能因为对方挂了而卡住整轮检查。
    """

    name = "webhook"

    def __init__(self, url: str, *, timeout: float = 3.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, alerts: list[Alert]) -> list[Alert]:
        payload = {"alerts": [a.as_dict() for a in alerts]}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"content-type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status >= 400:
                    return [_failed(f"HTTP {response.status}")]
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return [_failed(str(exc))]
        return []


def _failed(reason: str) -> Alert:
    message = f"告警投递失败（本体已经产生，渠道没送出去）：{reason}"
    return Alert("warn", "channel", "webhook_failed", message)


def highest(alerts: list[Alert]) -> str | None:
    """这批告警里最高的级别；空列表返回 None。"""
    if not alerts:
        return None
    return max(alerts, key=lambda a: SEVERITY[a.level]).level
