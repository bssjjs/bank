"""训练侧：读冻结集、装配同一条特征处理路径、算指标、训练与记录。

只读 `data/frozen/`，不再从原始库现算；派生列已经在冻结集里（由
`bank.dataset.build_dataset` 生成，训练与服务共用那一份实现）。
"""

from bank_training.data import FrozenError, load_frozen, split_xy
from bank_training.metrics import evaluate
from bank_training.pipeline import build_model, build_preprocessor

__all__ = [
    "FrozenError",
    "build_model",
    "build_preprocessor",
    "evaluate",
    "load_frozen",
    "split_xy",
]
