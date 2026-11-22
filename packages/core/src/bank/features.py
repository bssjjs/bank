"""特征处理：训练与服务共用的唯一一套编码、填补、标准化实现。

放在 `bank`（core）而不是训练包里，是因为服务端必须能反序列化同一个 Pipeline：
`OrderedEncoder` 这个类一旦定义在 `bank_training`，`bank_api` 加载模型包时就不得不
导入训练代码，依赖方向就破了。模型对象本身是 sklearn 的 `Pipeline`，它的预处理
支路全部来自本模块。

类型不在这里硬编码：全部从 `configs/features.yaml`（`CatalogSpec`）读。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank.catalog import CatalogSpec, FieldSpec

NUMERIC_TYPES = ("numeric", "time", "date", "id")


class OrderedEncoder(BaseEstimator, TransformerMixin):
    """把一列按给定顺序映射成 0..n-1；不在表里的一律 -1（未知类别不报错）。

    不直接用 `OrdinalEncoder`：`教育程度序` 是整数列，直接编码要么报 unknown
    category，要么得先转字符串；这里统一 `astype(str)` 后查表，训练与服务同构。
    """

    def __init__(self, order: Sequence[str]) -> None:
        # sklearn 的 clone 会逐项比较构造参数，这里必须原样存，不能 copy 一份。
        self.order = order

    def fit(self, X: pd.DataFrame, y: object = None) -> OrderedEncoder:
        _ = y
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        table = {str(value): index for index, value in enumerate(self.order)}
        column = X.iloc[:, 0].astype(str).map(lambda value: table.get(value, -1))
        return column.to_numpy(dtype="float64").reshape(-1, 1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        _ = input_features
        return np.asarray([f"{type(self).__name__}_{len(self.order)}"], dtype=object)


def order_for(field: FieldSpec) -> list[str]:
    """有序编码顺序；`是/否` 固定成 否=0、是=1，其余按 YAML 里 categories 的顺序。"""
    categories = [str(c) for c in (field.categories or [])]
    if field.type == "binary" and set(categories) == {"是", "否"}:
        return ["否", "是"]
    return categories


def build_preprocessor(
    spec: CatalogSpec, feature_names: Sequence[str], *, dense: bool = False
) -> ColumnTransformer:
    """数值支路（中位数填补+标准化）、有序/布尔支路（按序映射+标准化）、类别支路（one-hot）。"""
    numeric: list[str] = []
    ordered: list[FieldSpec] = []
    categorical: list[str] = []
    for name in feature_names:
        field = spec.field(name)
        if field.type in NUMERIC_TYPES:
            numeric.append(name)
        elif field.type in ("binary", "ordinal"):
            ordered.append(field)
        elif field.type == "categorical":
            categorical.append(name)
        else:
            raise ValueError(f"特征 {name} 的类型 {field.type} 不认识，检查 configs/features.yaml")

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    for field in ordered:
        transformers.append(
            (
                f"ord_{field.name}",
                Pipeline(
                    [
                        ("order", OrderedEncoder(order_for(field))),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("scale", StandardScaler()),
                    ]
                ),
                [field.name],
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                    dtype="float64",
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def select_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """按模型包的列顺序取特征；缺一列就报错，不交给下游去猜。"""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"缺少特征列：{missing}")
    return frame.loc[:, list(columns)].copy()
