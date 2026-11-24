"""训练参数（`configs/model.yaml`）：与数据口径分开，改超参数不动冻结指纹。

和 `bank.config` 一样两条规矩：拼错的键立刻报错、读进来就冻结。
"""

from __future__ import annotations

from pathlib import Path

from bank import schema as S
from bank.yamlkit import YamlError, load_mapping
from pydantic import BaseModel, ConfigDict, ValidationError

from bank_training import algorithms

ROOT = S.ROOT
MODEL_CONFIG_PATH = ROOT / "configs" / "model.yaml"


class ModelConfigError(RuntimeError):
    """`configs/model.yaml` 读不动或值不合理。"""


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    c: float
    max_iter: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    model: ModelParams


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> ModelConfig:
    try:
        raw = load_mapping(path, what="训练参数")
    except YamlError as exc:
        raise ModelConfigError(str(exc)) from exc
    try:
        cfg = ModelConfig.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"  {'.'.join(str(p) for p in err['loc']) or '<根>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ModelConfigError("训练参数校验失败：\n" + "\n".join(lines)) from exc
    if cfg.schema_version != 1:
        raise ModelConfigError(f"未知的训练参数版本 {cfg.schema_version}")
    # 算法名以注册表为准：加一种算法只加一个文件，不用回来改这里的枚举。
    if cfg.model.kind not in algorithms.available():
        available = "、".join(algorithms.available()) or "（空）"
        raise ModelConfigError(f"未知的模型种类 {cfg.model.kind!r}：可用的有 {available}")
    if not 0 < cfg.model.c <= 100:
        raise ModelConfigError("model.c 必须在 (0,100] 之间")
    if cfg.model.max_iter < 100:
        raise ModelConfigError("model.max_iter 至少 100，太小会训练不收敛")
    return cfg
