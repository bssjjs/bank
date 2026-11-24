"""逻辑回归：主模型（ADR 0005）。

只负责把预处理器和 `LogisticRegression` 接起来；预处理器的所有规则（缺失填补、有序编码、
独热编码、标准化）都在 `bank.features` 里，训练与服务走同一条代码路径。
"""

from __future__ import annotations

from collections.abc import Sequence

from bank import catalog, schema
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bank_training.algorithms import algorithm
from bank_training.pipeline import build_preprocessor


@algorithm("lr")
def build(
    spec: catalog.CatalogSpec,
    feature_names: Sequence[str],
    *,
    c: float = 1.0,
    max_iter: int = 5000,
) -> Pipeline:
    return Pipeline(
        [
            ("pre", build_preprocessor(spec, feature_names)),
            (
                "model",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    random_state=schema.SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
