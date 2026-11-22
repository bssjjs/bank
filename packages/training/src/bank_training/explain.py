"""单条解释：把"为什么给这个分"落到具体列。

逻辑回归的贡献值 = 系数 × 标准化后的取值，与线性 SHAP 等价、零额外依赖。
树模型不在线，所以这里只实现线性这一条（M3 主模型就是逻辑回归）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _column_names(preprocessor: object) -> list[str]:
    """按 ColumnTransformer 的实际输出顺序拼出列名。"""
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:  # type: ignore[attr-defined]
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


def explain_one(model: Pipeline, row: pd.DataFrame, top_k: int = 5) -> list[dict[str, object]]:
    """返回贡献最大的前 `top_k` 列（按绝对值排序），带原始取值与贡献。"""
    pre = model.named_steps["pre"]
    lr = model.named_steps["model"]
    transformed = np.asarray(pre.transform(row))
    coefficients = np.asarray(lr.coef_).reshape(-1)
    names = _column_names(pre)
    if transformed.shape[1] != len(names):
        raise RuntimeError(
            f"解释列数 {len(names)} 与变换后列数 {transformed.shape[1]} 不一致，"
            "改过预处理就得同步改 _column_names"
        )
    values = transformed.reshape(-1)
    contributions = coefficients * values
    order = np.argsort(-np.abs(contributions))[:top_k]
    out: list[dict[str, object]] = []
    for index in order:
        feature_name = names[index]
        raw_column = feature_name.split("=")[0]
        raw_value = row.iloc[0][raw_column] if raw_column in row.columns else None
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
