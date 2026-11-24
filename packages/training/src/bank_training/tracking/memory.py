"""内存追踪后端：只活在当前进程里，给测试和「先把流程跑通」的场合用。

它没有继承 `Tracker`，但方法都对得上协议（`Protocol` 只看「长什么样」），所以能当 `Tracker` 用。
它记下的是同样的东西：参数（统一转字符串）、标签、指标、产物路径。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bank_training.tracking.base import TrackerOptions, as_text, tracker


@dataclass
class RunRecord:
    """一次运行记下来的东西；运行进行中要不断往里写，所以不是 frozen。"""

    run_name: str
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


class InMemoryTracker:
    """把运行记录留在内存里，进程结束就没了。"""

    def __init__(self, experiment: str = "test") -> None:
        self.experiment = experiment
        self.name = "memory"
        self.runs: dict[str, RunRecord] = {}
        self._current: str | None = None

    @contextmanager
    def start_run(self, run_name: str, *, run_id: str | None = None) -> Iterator[str]:
        chosen = run_id or uuid.uuid4().hex
        self.runs.setdefault(chosen, RunRecord(run_name=run_name))
        self._current = chosen
        try:
            yield chosen
        finally:
            self._current = None

    def _run(self, run_id: str | None = None) -> RunRecord:
        target = run_id or self._current
        if target is None:
            raise RuntimeError("不在运行中：先 with tracker.start_run(...)")
        if target not in self.runs:
            raise RuntimeError(f"没有这次运行 {target!r}：run_id 是不是写错了")
        return self.runs[target]

    def log_params(self, params: Mapping[str, Any]) -> None:
        run = self._run()
        run.params.update({str(key): as_text(value) for key, value in params.items()})

    def set_tags(self, tags: Mapping[str, str]) -> None:
        run = self._run()
        run.tags.update({str(key): as_text(value) for key, value in tags.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        run = self._run()
        run.metrics.update({str(key): float(value) for key, value in metrics.items()})

    def log_metric(self, name: str, value: float) -> None:
        self.log_metrics({name: value})

    def set_tag(self, name: str, value: str) -> None:
        self.set_tags({name: value})

    def log_artifact(
        self, path: Path | str, *, artifact_path: str | None = None, run_id: str | None = None
    ) -> None:
        run = self._run(run_id)
        target = Path(path)
        run.artifacts.append(
            {
                "path": target.as_posix(),
                "artifact_path": artifact_path,
                "exists": target.is_file(),
            }
        )


@tracker("memory")
def build(options: TrackerOptions) -> InMemoryTracker:
    """内存后端的工厂函数。"""
    return InMemoryTracker(options.experiment)
