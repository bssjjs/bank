"""随机森林：对照模型（ADR 0005 里它只用来证明"换更强的模型没换来更好的人数"）。

`n_jobs=1` 是必须的：Windows 沙箱挡了 joblib 的命名管道，多进程/线程池会抛
`PermissionError: [WinError 5]`，所以单线程跑（数据量 1.7 万行，够快）。
"""

from __future__ import annotations

from collections.abc import Sequence

from bank import catalog, schema
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from bank_training.algorithms import algorithm
from bank_training.pipeline import build_preprocessor


@algorithm("rf")
def build(
    spec: catalog.CatalogSpec,
    feature_names: Sequence[str],
    *,
    c: float = 1.0,
    max_iter: int = 5000,
) -> Pipeline:
    """`c` 与 `max_iter` 是逻辑回归的参数，这里用不上，保留签名是为了注册表统一调用。"""
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
                    random_state=schema.SEED,
                ),
            ),
        ]
    )
