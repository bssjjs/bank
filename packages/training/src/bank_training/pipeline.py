"""特征处理与模型装配。

训练与服务走同一个 `build_preprocessor`：填补、编码、标准化、派生列全在里面。
服务端拿到原始字段后调用 `bank.dataset.build_dataset` 派生，再用这个 Pipeline
的 `.predict_proba`，不存在"训练一套、线上另写一套"的第二条路径。

特征类型不在这里硬编码：从 `configs/features.yaml`（`CatalogSpec`）读，
所以加一列只改 YAML 和 `schema.py`。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import CatalogSpec, FieldSpec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_KINDS = ("lr", "rf")

NUMERIC_TYPES = ("numeric", "time", "date", "id")


class OrderedEncoder(BaseEstimator, TransformerMixin):
    """把一列按给定顺序映射成 0..n-1；不在表里的一律 -1（未知类别不报错）。

    不直接用 `OrdinalEncoder`：`教育程度序` 是整数列，直接编码要么报 unknown
    category，要么得先转字符串；这里统一 `astype(str)` 后查表，训练与服务同构。
    """

    def __init__(self, order: Sequence[object]) -> None:
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


def _order(field: FieldSpec) -> list[str]:
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
                        ("order", OrderedEncoder(_order(field))),
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


def build_model(
    kind: str,
    spec: CatalogSpec,
    feature_names: Sequence[str],
    *,
    c: float = 1.0,
    max_iter: int = 5000,
) -> Pipeline:
    """`lr` 是主模型；`rf` 只做对照，不上线。"""
    if kind == "lr":
        return Pipeline(
            [
                ("pre", build_preprocessor(spec, feature_names)),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        max_iter=max_iter,
                        random_state=S.SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if kind == "rf":
        return Pipeline(
            [
                ("pre", build_preprocessor(spec, feature_names, dense=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=S.SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f"未知模型类型 {kind!r}，只能从 {MODEL_KINDS} 里选")
