"""请求/响应模型。

请求模型是**按模型包动态生成**的：字段名、类型、范围、枚举全部来自 `bundle.json` 的
`inputs`，而 `inputs` 又来自 `configs/features.yaml`。前端拿 `GET /v1/model` 生成表单，
后端拿同一份契约校验请求——两边不可能对不上。
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from bank.bundle import Bundle
from pydantic import BaseModel, Field, create_model

LEVEL_ADVICE: dict[str, str] = {
    "优先": "建议今天优先外呼：分数在名额内且靠前。",
    "备选": "建议作为次日候选：今天的名额先给更高分的客户。",
    "不建议": "不建议外呼：名额有限，优先给高分客户。",
}

_NUMERIC_TYPES = ("numeric", "time", "date", "id")


class Factor(BaseModel):
    feature: str = Field(description="列名，类别特征带 =取值")
    value: str | None = Field(description="这一列的原始取值")
    contribution: float = Field(
        description="对 log-odds 的贡献，正数推高、负数压低（单位见 contribution_unit）"
    )


class Prediction(BaseModel):
    request_id: str = Field(description="请求号，同一条批量请求共用一个")
    model_version: str = Field(description="模型包版本")
    score: float = Field(description="认购概率预测（0–1）")
    level: str = Field(description="优先 / 备选 / 不建议")
    factors: list[Factor] = Field(description="贡献最大的几列")
    base_value: float = Field(description="基准值（截距，log-odds）：解释的起点")
    base_probability: float = Field(description="基准值经 sigmoid 后的概率（0–1）")
    contribution_unit: str = Field(description="factors 与 base_value 的单位，固定 log_odds")
    advice: str = Field(description="用途提示：这个等级该怎么用")


class BatchResult(BaseModel):
    model_version: str
    request_id: str
    items: list[Prediction]


def _literal(categories: tuple[str, ...]) -> Any:
    """类别字段用运行时拼出来的 `Literal`。

    用 `Enum` 的话，`/docs` 里会多出一个 component，名字是类名（中文会被 pydantic
    改写成 `__`），既不可读也没必要；`Literal` 直接把 `enum` 内联在这一列上。
    """
    return Literal[categories]


def build_input_model(bundle: Bundle) -> type[BaseModel]:
    """把模型包的 15 个原始输入列变成一个 pydantic 模型（也进 OpenAPI）。"""
    fields: dict[str, Any] = {}
    for entry in bundle.doc.inputs:
        unit = f"（{entry.unit}）" if entry.unit else ""
        description = f"{entry.display}{unit}"
        if entry.type in _NUMERIC_TYPES:
            # allow_inf_nan=False：NaN/inf 在 JSON 里能混进来，进模型前就拦掉，
            # 否则会一路走到 sklearn 才报一个没人看得懂的错。
            constraints: dict[str, Any] = {"description": description, "allow_inf_nan": False}
            if entry.min is not None:
                constraints["ge"] = entry.min
            if entry.max is not None:
                constraints["le"] = entry.max
            fields[entry.name] = (float, Field(**constraints))
        else:
            if not entry.categories:
                raise ValueError(f"字段 {entry.name} 是 {entry.type}，却没有枚举值")
            fields[entry.name] = (
                _literal(entry.categories),
                Field(description=description),
            )
    return create_model("PredictInput", **fields)


def to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([payload])
