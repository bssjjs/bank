"""模型装配：把 core 的预处理器和具体分类器拼成一条 Pipeline。

预处理器（填补、编码、标准化、有序映射）在 `bank.features`，训练与服务共用；
这里只管「用哪个模型」。`lr` 是主模型；`rf` 只做对照，不上线。
"""

from __future__ import annotations

from collections.abc import Sequence

from bank import schema as S
from bank.catalog import CatalogSpec
from bank.features import NUMERIC_TYPES, OrderedEncoder, build_preprocessor, order_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

__all__ = [
    "MODEL_KINDS",
    "NUMERIC_TYPES",
    "OrderedEncoder",
    "build_model",
    "build_preprocessor",
    "order_for",
]

MODEL_KINDS = ("lr", "rf")


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
