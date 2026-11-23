"""单条解释：把"为什么给这个分"落到具体列。

逻辑回归的贡献值 = 系数 × 标准化后的取值，与线性 SHAP 等价、零额外依赖。
树模型不在线，所以这里只实现线性这一条（M3 主模型就是逻辑回归）。

放在 `bank`（core）是因为服务端每次预测都要调它，而 `bank_api` 不许导入训练代码。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

EXPLAIN_METHOD = "linear_coef_x_scaled"


def column_names(preprocessor: Any) -> list[str]:
    """按 ColumnTransformer 的实际输出顺序拼出列名。

    参数标注为 `Any`：sklearn 本身不带类型信息，`ColumnTransformer.transformers_`
    在 `object` 上是看不到的；与其写 `# type: ignore`，不如让调用方传已 fit 的预处理步。
    """
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        cols = list(columns)
        if name == "num":
            names.extend(cols)
        elif name.startswith("ord_"):
            names.append(cols[0])
        elif name == "cat":
            for column, categories in zip(cols, transformer.categories_, strict=True):
                names.extend(f"{column}={value}" for value in categories)
        else:
            names.extend(cols)
    return names


def _linear_model(model: Pipeline) -> object:
    """取 Pipeline 的最后一步当分类器：不依赖训练脚本给它起的名字。"""
    if not model.steps:
        raise RuntimeError("Pipeline 里没有步骤")
    estimator = model.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        raise RuntimeError(f"解释只支持线性模型，当前最后一步是 {type(estimator).__name__}")
    return estimator


def explain_one(model: Pipeline, row: pd.DataFrame, top_k: int = 5) -> list[dict[str, object]]:
    """返回贡献最大的前 `top_k` 列（按绝对值排序），带原始取值与贡献。"""
    pre = model.steps[0][1]
    lr = _linear_model(model)
    transformed = np.asarray(pre.transform(row))
    coefficients = np.asarray(getattr(lr, "coef_")).reshape(-1)
    names = column_names(pre)
    if transformed.shape[1] != len(names):
        raise RuntimeError(
            f"解释列数 {len(names)} 与变换后列数 {transformed.shape[1]} 不一致，"
            "改过预处理就得同步改 column_names"
        )
    values = transformed.reshape(-1)
    contributions = coefficients * values
    order = np.argsort(-np.abs(contributions))[:top_k]
    out: list[dict[str, object]] = []
    for index in order:
        feature_name = names[index]
        raw_column = feature_name.split("=")[0]
        raw_value = row.iloc[0][raw_column] if raw_column in row.columns else None
        if isinstance(raw_value, float):
            raw_value = round(raw_value, 4)
        out.append(
            {
                "feature": feature_name,
                "value": None if raw_value is None else str(raw_value),
                "contribution": round(float(contributions[index]), 6),
            }
        )
    return out


def explain_many(
    model: Pipeline, rows: Sequence[pd.DataFrame], top_k: int = 5
) -> list[list[dict[str, object]]]:
    return [explain_one(model, row, top_k=top_k) for row in rows]
