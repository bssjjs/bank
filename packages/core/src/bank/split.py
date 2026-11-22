"""时间切分：窗口裁剪 + 按时间切训练/验证 + 名额计算。

严格按 ADR 0002：不打乱、不分层、缓冲 0 天、切点落月份边界。
这里只做"切"，不做任何统计量的 fit——统计量属于训练集，由 M3 的 Pipeline 负责。
"""

from __future__ import annotations

import pandas as pd

from bank import config as C
from bank import schema as S
from bank import timeaxis as T


def window_mask(dates: pd.Series, cfg: C.Config) -> pd.Series:
    """建模窗口 [start, end]，两端闭区间。"""
    return (dates >= pd.Timestamp(cfg.window.start)) & (dates <= pd.Timestamp(cfg.window.end))


def train_mask(dates: pd.Series, cfg: C.Config) -> pd.Series:
    """训练集：窗口内、且早于切点。缓冲期 0 → 没有额外的间隔月。"""
    cut = pd.Timestamp(cfg.split.cut)
    return window_mask(dates, cfg) & (dates < cut)


def valid_mask(dates: pd.Series, cfg: C.Config) -> pd.Series:
    """验证集：窗口内、且不早于切点（切点当天归验证集）。"""
    cut = pd.Timestamp(cfg.split.cut)
    return window_mask(dates, cfg) & (dates >= cut)


def split_and_check(df: pd.DataFrame, cfg: C.Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """切出训练/验证两表，并校验结构不变量。返回 (train, valid, facts)。"""
    if T.DATE not in df.columns:
        raise KeyError(f"缺少 {T.DATE}：切分必须基于重建日期，先跑 timeaxis")

    dates = df[T.DATE]
    tr = df[train_mask(dates, cfg)]
    va = df[valid_mask(dates, cfg)]
    win = df[window_mask(dates, cfg)]

    if tr.empty or va.empty:
        raise ValueError("训练集或验证集为空，切分口径不可用")
    if len(tr) + len(va) != len(win):
        raise AssertionError("训练 + 验证 不等于窗口行数：有行被漏掉或重复计入")
    if tr[T.DATE].max() >= pd.Timestamp(cfg.split.cut):
        raise AssertionError("训练集出现切点及之后的日期")
    if va[T.DATE].min() < pd.Timestamp(cfg.split.cut):
        raise AssertionError("验证集出现切点之前的日期")
    if set(tr.index) & set(va.index):
        raise AssertionError("训练集与验证集有重叠行")

    facts = {
        "window": _slice(tr, va),
        "train": _one(tr),
        "valid": _one(va),
        "cut": str(cfg.split.cut),
        "buffer_days": cfg.split.buffer_days,
    }
    return tr, va, facts


def quota_n(n_valid: int, cfg: C.Config) -> int:
    """名额人数：分母是验证集行数（踩坑清单第 9 条）。"""
    return C.quota_size(n_valid, cfg)


def _one(frame: pd.DataFrame) -> dict:
    n = len(frame)
    pos = int(frame[S.TARGET].sum())
    return {
        "rows": n,
        "positives": pos,
        "pos_rate": round(pos / n, 4) if n else None,
        "date_min": str(frame[T.DATE].min().date()),
        "date_max": str(frame[T.DATE].max().date()),
    }


def _slice(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    both = pd.concat([tr, va])
    n = len(both)
    pos = int(both[S.TARGET].sum())
    return {
        "rows": n,
        "positives": pos,
        "pos_rate": round(pos / n, 4) if n else None,
        "date_min": str(both[T.DATE].min().date()),
        "date_max": str(both[T.DATE].max().date()),
    }
