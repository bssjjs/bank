"""模型装配：预处理 + 算法的入口；算法本身在 `bank_training.algorithms` 里，一种一个文件。

预处理器（填补、编码、标准化、有序映射）在 `bank.features`，训练与服务共用；这里只做分发。
`lr` 是主模型；`rf` 只做对照，不上线。
"""

from __future__ import annotations

from collections.abc import Sequence

from bank.catalog import CatalogSpec
from bank.features import NUMERIC_TYPES, OrderedEncoder, build_preprocessor, order_for
from sklearn.pipeline import Pipeline

from bank_training import algorithms

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
    """`lr` 是主模型；`rf` 只做对照，不上线。可用名字以注册表为准（`MODEL_KINDS` 是它的快照）。"""
    return algorithms.build(kind, spec, feature_names, c=c, max_iter=max_iter)
