"""E-L2 服务指标：进程内的请求计数、延迟、预测分布（契约 `service_metrics.v1`）。

为什么不做 Prometheus 客户端：本项目是单进程演示服务，没有采集端；
指标通过 `GET /metrics` 直接读 JSON，答辩时 `curl` 一下就能看到。
为什么计数要加锁：uvicorn 默认单 worker，但测试/后续多线程仍要保证自洽。
"""

from __future__ import annotations

import threading
import time
from typing import Any

__all__ = ["CONTRACT", "Counters"]

CONTRACT = "service_metrics.v1"


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._by_route: dict[str, int] = {}
        self._by_status: dict[str, int] = {}
        self._by_level: dict[str, int] = {}
        self.predictions = 0
        self.latency_ms_sum = 0.0
        self.latency_ms_max = 0.0

    def observe(self, *, route: str, status: int, latency_ms: float) -> None:
        with self._lock:
            key = f"{route} {status}"
            self._by_route[key] = self._by_route.get(key, 0) + 1
            code = str(status)
            self._by_status[code] = self._by_status.get(code, 0) + 1
            self.latency_ms_sum += latency_ms
            self.latency_ms_max = max(self.latency_ms_max, latency_ms)

    def count_prediction(self, level: str) -> None:
        with self._lock:
            self.predictions += 1
            self._by_level[level] = self._by_level.get(level, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = sum(self._by_status.values())
            mean = self.latency_ms_sum / requests if requests else 0.0
            return {
                "contract": CONTRACT,
                "uptime_s": round(time.monotonic() - self._started, 3),
                "requests": requests,
                "by_route": dict(sorted(self._by_route.items())),
                "by_status": dict(sorted(self._by_status.items())),
                "predictions": self.predictions,
                "by_level": dict(sorted(self._by_level.items())),
                "latency_ms_mean": round(mean, 3),
                "latency_ms_max": round(self.latency_ms_max, 3),
            }
