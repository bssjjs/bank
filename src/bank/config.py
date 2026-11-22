"""配置层：口径数字只写一次（``configs/base.yaml``），代码只读不猜。

两条硬规矩：
1. **拼错的键必须立刻报错**（``extra="forbid"``），不能静默忽略；
2. **配置一旦被读进来就冻结**（``frozen=True``），跑一半改口径是不可能的。
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bank import repro
from bank import schema as S

CONFIG_PATH = S.ROOT / "configs" / "base.yaml"


class ConfigError(RuntimeError):
    """配置读不进来、校验不过时抛这个，别让 KeyError 漏到上层。"""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataCfg(_Strict):
    raw_db: str
    raw_table: str
    frozen_dir: str


class WindowCfg(_Strict):
    start: date
    end: date


class SplitCfg(_Strict):
    cut: date
    buffer_days: int


class QuotaCfg(_Strict):
    frac: float


class FeatureCfg(_Strict):
    set: Literal["A", "B"]


class TargetCfg(_Strict):
    column: str
    positive: str
    negative: str
    window_days: int


class Config(_Strict):
    schema_version: int
    seed: int
    data: DataCfg
    window: WindowCfg
    split: SplitCfg
    quota: QuotaCfg
    features: FeatureCfg
    target: TargetCfg

    @model_validator(mode="after")
    def _sane(self) -> Config:
        if self.schema_version != 1:
            raise ValueError(f"未知的配置版本 {self.schema_version}，读不动就别跑")
        if not self.window.start < self.window.end:
            raise ValueError("window.start 必须早于 window.end")
        if not (self.window.start < self.split.cut <= self.window.end):
            raise ValueError("split.cut 必须落在 window 内部且晚于起点")
        if not 0 <= self.split.buffer_days <= 365:
            raise ValueError("split.buffer_days 必须在 0~365 之间")
        if not 0 < self.quota.frac <= 1:
            raise ValueError("quota.frac 必须在 (0,1] 之间")
        if self.target.window_days != 0:
            raise ValueError("本数据集标签通话结束即得，window_days 只能是 0（ADR 0002）")
        if self.target.column != S.TARGET_RAW:
            raise ValueError(f"target.column 必须与 schema 一致：{S.TARGET_RAW}")
        if (self.target.positive, self.target.negative) != (S.POSITIVE, S.NEGATIVE):
            raise ValueError("target 的正负标签取值必须与 schema 一致")
        if self.data.raw_table != S.RAW_TABLE:
            raise ValueError(f"data.raw_table 必须与 schema 一致：{S.RAW_TABLE}")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML 默认允许重复键、后者覆盖前者；口径文件里重复键一定是复制粘贴出错。"""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(f"配置键重复：{key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise ConfigError(f"找不到配置文件：{path}")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:  # 缩进/语法错
        raise ConfigError(f"配置文件不是合法 YAML：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置文件顶层必须是映射（key: value）")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        # 把 pydantic 的长报错压成"哪个键、为什么"，现场看一行就够
        lines = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "<根>"
            lines.append(f"  {where}: {err['msg']}")
        raise ConfigError("配置校验失败：\n" + "\n".join(lines)) from exc


def config_fingerprint(cfg: Config) -> str:
    """配置指纹：口径改了，指纹就变，manifest 里立刻看得出来。"""
    return repro.sha256_json(cfg.model_dump(mode="json"))


def quota_size(n_valid: int, cfg: Config) -> int:
    """名额人数 = ceil(验证集行数 × frac)。分母是验证集，不是全表（踩坑清单第 9 条）。"""
    if n_valid <= 0:
        raise ValueError("验证集为空，名额无从谈起")
    # 先按 9 位小数收敛，避免 0.1 这类二进制误差把 3550 算成 3551
    return math.ceil(round(n_valid * cfg.quota.frac, 9))
