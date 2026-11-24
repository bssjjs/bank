"""追踪协议（讲义 C-L1c）：一次运行要记下的东西，都从同一个接口出去。

为什么要协议：
- 编排层（`train.py`、`export.py`、`drift.py`……）不直接调 MLflow，测试用内存后端就能把整条流程跑完，
  不需要起 `mlflow.db`；
- MLflow 升级、换后端（比如换成一个 HTTP 报表服务）时只改适配文件一处，其它文件不用动。

`mlflow` 这个库只允许在 `bank_training.tracking.mlflow_backend` 里 import，
`tests/architecture/test_layers.py` 用 AST 扫着这条规矩。

`configs/tracking.yaml` 选后端；后端用 `@tracker("名字")` 登记，跟清洗规则、算法注册表是同一套写法。
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bank import schema as S
from bank.config import ConfigError
from bank.yamlkit import YamlError, load_mapping
from pydantic import BaseModel, ConfigDict, Field, ValidationError

TRACKING_CONFIG_PATH = S.ROOT / "configs" / "tracking.yaml"
DEFAULT_TRACKING_URI = f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "bank-marketing"


class TrackingError(RuntimeError):
    """追踪配置或后端用错时抛出。"""


def as_text(value: Any) -> str:
    """参数与标签统一成字符串：布尔用小写、结构化值转成 JSON，两个后端记下的东西才一样。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(value, key=str), ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class TrackerOptions:
    """创建后端要的参数：实验名、仓库根、追踪服务地址。"""

    experiment: str = DEFAULT_EXPERIMENT
    root: Path = S.ROOT
    tracking_uri: str = DEFAULT_TRACKING_URI


class Tracker(Protocol):
    """一次运行要记的东西。

    `start_run` 返回上下文管理器，形状是 `with tracker.start_run("名字") as run_id`；
    运行块里的 `log_*` 都写进这次运行。参数一律转成字符串再记（MLflow 的参数本来就是字符串，
    内存后端也照这个口径存，测试才测得出差别）。
    """

    def start_run(self, run_name: str, *, run_id: str | None = None) -> AbstractContextManager[str]:
        """开始一次运行；`run_id` 给了就接着那次运行写，用于往已有运行里补产物。"""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def log_metric(self, name: str, value: float) -> None:
        """记一个指标；等价于 `log_metrics({name: value})`，循环里逐个记时更顺手。"""
        ...

    def set_tag(self, name: str, value: str) -> None:
        """记一个标签；等价于 `set_tags({name: value})`。"""

    def log_artifact(
        self, path: Path | str, *, artifact_path: str | None = None, run_id: str | None = None
    ) -> None:
        """把文件挂到这次运行下；`run_id` 给了就挂到那次运行（不在运行块里也能用）。"""

    name: str
    """后端名字（`mlflow` / `memory`）：报告里要写真名，别让日志撒谎。"""


TrackerFactory = Callable[[TrackerOptions], Tracker]
TRACKERS: dict[str, TrackerFactory] = {}


def tracker(name: str) -> Callable[[TrackerFactory], TrackerFactory]:
    """登记一个追踪后端。重名直接报错——一个名字只能对应一个实现。"""

    def register(factory: TrackerFactory) -> TrackerFactory:
        if name in TRACKERS:
            raise ValueError(f"追踪后端 {name!r} 重复登记：一个名字只能对应一个实现")
        TRACKERS[name] = factory
        return factory

    return register


def _builtin() -> None:
    """把两个内置后端导入进来（导入即登记）。"""
    for module in ("memory", "mlflow_backend"):
        importlib.import_module(f"{__name__.rsplit('.', 1)[0]}.{module}")


def available() -> tuple[str, ...]:
    """已登记的后端名字，排序后返回。"""
    _builtin()
    return tuple(sorted(TRACKERS))


def resolve_tracking_uri(uri: str, root: Path = S.ROOT) -> str:
    """`sqlite:///mlflow.db` 这种相对地址按仓库根解析；别的地址原样返回。"""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    path = Path(uri[len(prefix) :])
    if path.is_absolute():
        return uri
    return prefix + (root / path).as_posix()


def get_tracker(name: str | None = None, options: TrackerOptions | None = None) -> Tracker:
    """按名字取后端；`name` 为空时读 `configs/tracking.yaml`。"""
    _builtin()
    if options is not None:
        config = None
    else:
        config = load_tracking_config()
    chosen = name or (config.backend if config else "")
    if chosen not in TRACKERS:
        raise TrackingError(f"未知的追踪后端 {chosen!r}：可用的有 {', '.join(available())}")
    if options is None:
        assert config is not None
        options = TrackerOptions(
            experiment=config.experiment,
            tracking_uri=resolve_tracking_uri(config.tracking_uri),
        )
    return TRACKERS[chosen](options)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrackingCfg(_Strict):
    backend: str = Field(min_length=1)
    experiment: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


class TrackingConfig(_Strict):
    schema_version: int
    tracking: TrackingCfg


def load_tracking_config(path: Path = TRACKING_CONFIG_PATH) -> TrackingCfg:
    """读追踪配置；键拼错、文件缺失、地址为空都报得出来。"""
    try:
        raw = load_mapping(path, what="追踪配置")
        config = TrackingConfig.model_validate(raw)
    except YamlError as exc:
        raise ConfigError(str(exc)) from exc
    except ValidationError as exc:
        lines = [f"追踪配置 {path} 校验失败："]
        for item in exc.errors():
            where = ".".join(str(part) for part in item["loc"])
            lines.append(f"  {where}: {item['msg']}")
        raise ConfigError("\n".join(lines)) from exc
    if config.schema_version != 1:
        raise ConfigError(f"追踪配置版本 {config.schema_version} 不认识：当前只支持 1")
    return config.tracking
