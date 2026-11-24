"""MLflow 追踪后端：本仓库里**唯一**允许 import mlflow 的文件。

规矩由 `tests/architecture/test_layers.py::TrackingTest` 用 AST 扫着：别的模块写 `import mlflow`
也算错。这样换后端、升 MLflow 大版本，只改这一个文件。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from bank_training.tracking.base import TrackerOptions, as_text, tracker


class MlflowTracker:
    """把协议上的六件事翻译成 MLflow 的调用。"""

    def __init__(self, options: TrackerOptions) -> None:
        self.options = options
        self.name = "mlflow"
        mlflow.set_tracking_uri(options.tracking_uri)
        mlflow.set_experiment(options.experiment)

    @contextmanager
    def start_run(self, run_name: str, *, run_id: str | None = None) -> Iterator[str]:
        with mlflow.start_run(run_name=run_name, run_id=run_id) as active:
            yield active.info.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params({str(key): as_text(value) for key, value in params.items()})

    def set_tags(self, tags: Mapping[str, str]) -> None:
        mlflow.set_tags({str(key): as_text(value) for key, value in tags.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()})

    def log_metric(self, name: str, value: float) -> None:
        mlflow.log_metric(name, float(value))

    def set_tag(self, name: str, value: str) -> None:
        mlflow.set_tag(name, as_text(value))

    def log_artifact(
        self, path: Path | str, *, artifact_path: str | None = None, run_id: str | None = None
    ) -> None:
        # 给了 run_id 就往那次运行里挂（train 给主模型补评估报告时这么用）；
        # 没给就用当前活动运行。
        if run_id is None:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            return
        client = mlflow.MlflowClient(tracking_uri=self.options.tracking_uri)
        client.log_artifact(run_id, str(path), artifact_path=artifact_path)


@tracker("mlflow")
def build(options: TrackerOptions) -> MlflowTracker:
    """MLflow 后端的工厂函数。"""
    return MlflowTracker(options)
