"""分布漂移（PSI）：训练集当基准，线上批次当对照。

口径固定三条，改之前先改 `docs/adr/0008-漂移监控与回流.md`：

1. 数值列的分箱边界**只用基准集算**（等频 10 箱），线上批次沿用同一边界——
   两边各算各的分位数，漂移会被"自适应分箱"抹平。
2. 缺失单独占一箱，不与任何区间合并。`距上次联系天数 = -1` 这类哨兵在派生列里
   已经变过形态，这里只处理真正的空值。
3. 类别列按取值分布算；基准集没见过的取值合并成「其他」，否则线上冒出新职业
   会被算成 10 个箱都归零，PSI 直接爆表。

PSI 的经验阈值（教科书口径，写进 ADR 供答辩）：< 0.1 稳定，0.1–0.25 关注，> 0.25 严重。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BINS = 10
EPS = 1e-6
STABLE = 0.1
WATCH = 0.25

NUMERIC_TYPES = ("numeric", "time", "date", "id")


@dataclass(frozen=True)
class Drift:
    feature: str
    kind: str
    psi: float
    level: str
    detail: dict[str, Any]


def level_of(psi: float) -> str:
    if psi < STABLE:
        return "稳定"
    if psi < WATCH:
        return "关注"
    return "严重"


def quantile_edges(values: pd.Series, bins: int = BINS) -> list[float]:
    """等频分箱边界（含 -inf/+inf 之外只到 min/max），重复边界去重。"""
    clean = values.dropna().to_numpy(dtype="float64")
    if clean.size == 0:
        raise ValueError("基准集这一列全是空值，无法定箱")
    edges = np.quantile(clean, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = clean.min()
    edges[-1] = clean.max()
    unique = np.unique(edges)
    if unique.size < 2:
        unique = np.array([clean.min(), clean.max()], dtype="float64")
    return [float(item) for item in unique]


def shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    """把一列映射成 (箱数 + 1) 的比例向量，最后一箱是缺失。"""
    inner = edges[1:-1]
    codes = np.searchsorted(inner, values.to_numpy(dtype="float64"), side="right")
    shares = np.zeros(len(inner) + 2, dtype="float64")
    shares[0] = float(np.count_nonzero(np.isnan(values.to_numpy(dtype="float64"))))
    for code in codes:
        shares[int(code) + 1] += 1.0
    return shares / shares.sum()


def psi_from_shares(reference: np.ndarray, current: np.ndarray) -> float:
    ref = reference + EPS
    cur = current + EPS
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = BINS) -> Drift:
    edges = quantile_edges(reference, bins)
    value = psi_from_shares(shares(reference, edges), shares(current, edges))
    return Drift(
        feature=str(reference.name),
        kind="numeric",
        psi=value,
        level=level_of(value),
        detail={"edges": [round(edge, 6) for edge in edges], "missing_bin": True},
    )


def psi_categorical(reference: pd.Series, current: pd.Series) -> Drift:
    ref = reference.astype("object").fillna("<缺失>")
    cur = current.astype("object").fillna("<缺失>")
    known = ref.value_counts(normalize=True)
    seen = cur.where(cur.isin(known.index), "<未见过>")
    other = float((seen == "<未见过>").mean())
    now = seen.value_counts(normalize=True).reindex(known.index, fill_value=0.0)
    shares_now = now.to_numpy(dtype="float64")
    if other:
        shares_now = np.append(shares_now, other)
    shares_ref = known.to_numpy(dtype="float64")
    if other:
        shares_ref = np.append(shares_ref, 0.0)
    value = psi_from_shares(shares_ref, shares_now)
    return Drift(
        feature=str(reference.name),
        kind="categorical",
        psi=value,
        level=level_of(value),
        detail={"categories": len(known), "unseen_share": round(other, 6)},
    )


def psi(reference: pd.Series, current: pd.Series, *, numeric: bool, bins: int = BINS) -> Drift:
    if numeric:
        return psi_numeric(reference, current, bins)
    return psi_categorical(reference, current)


def report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    numeric_columns: set[str],
    bins: int = BINS,
) -> list[Drift]:
    rows: list[Drift] = []
    for name in columns:
        if name not in current.columns:
            continue
        rows.append(
            psi(
                reference[name],
                current[name],
                numeric=name in numeric_columns,
                bins=bins,
            )
        )
    return sorted(rows, key=lambda item: item.psi, reverse=True)


def score_psi(reference_scores: np.ndarray, current_scores: np.ndarray) -> Drift:
    """模型分数本身的漂移：拿训练集分数当基准，看线上分数分布有没有平移。"""
    ref = pd.Series(np.asarray(reference_scores, dtype="float64"), name="score")
    cur = pd.Series(np.asarray(current_scores, dtype="float64"), name="score")
    edges = quantile_edges(ref)
    value = psi_from_shares(shares(ref, edges), shares(cur, edges))
    return Drift(
        feature="score",
        kind="score",
        psi=value,
        level=level_of(value),
        detail={"edges": [round(edge, 6) for edge in edges]},
    )


def label_rate_shift(reference_rate: float, current_rate: float) -> float:
    """正例率的相对变化（%），用来解释「为什么模型分数整体偏低」。"""
    if reference_rate <= 0:
        return math.inf
    return (current_rate - reference_rate) / reference_rate * 100.0
