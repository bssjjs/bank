"""读冻结集，切出设计矩阵。

训练只吃 `data/frozen/`：`train.parquet` / `valid.parquet` / `manifest.json`。
冻结集由 `python -m bank.freeze` 生成，指纹与行数写在 `FROZEN.json` 里。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from bank import schema as S
from bank.dataset import model_features

FROZEN_DIR: Any = S.ROOT / "data" / "frozen"
TRAIN_PATH: Any = FROZEN_DIR / "train.parquet"
VALID_PATH: Any = FROZEN_DIR / "valid.parquet"
MANIFEST_PATH: Any = FROZEN_DIR / "manifest.json"


class FrozenError(RuntimeError):
    """冻结集缺失或不完整。"""


def load_frozen() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """返回 (train, valid, manifest)，缺文件就报错并给出修复命令。"""
    for path in (TRAIN_PATH, VALID_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FrozenError(f"缺 {path}，先跑 python -m bank.freeze")
    train = pd.read_parquet(TRAIN_PATH)
    valid = pd.read_parquet(VALID_PATH)
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return train, valid, manifest


def split_xy(frame: pd.DataFrame, feature_set: str = "A") -> tuple[pd.DataFrame, pd.Series]:
    """按特征集取出 X 与 y；列顺序与 `model_features()` 一致。"""
    names = model_features(feature_set)
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise FrozenError(f"冻结集缺特征列：{missing}")
    return frame.loc[:, list(names)].copy(), frame[S.TARGET].astype("int8").copy()
