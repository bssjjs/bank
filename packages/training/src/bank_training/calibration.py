"""校准：把分数按箱切开，比对「说多少」与「实际多少」。

结论先行（ADR 0005、模型卡第 5 节）：本模型**不校准**，也不许拿验证集拟合校准器。
这里只出证据——可靠性表与三个校准指标，供答辩时解释"为什么分数偏低但排序可用"。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from bank_training import metrics as M

BINS = 10


def _arrays(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="int8")
    s = np.asarray(score, dtype="float64")
    if len(y) != len(s):
        raise ValueError(f"标签 {len(y)} 行与分数 {len(s)} 行不一致")
    if len(y) == 0:
        raise ValueError("评估集是空的")
    return y, s


def reliability(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, bins: int = BINS
) -> list[dict[str, float]]:
    """可靠性表：每箱的人数、平均分数、实际正例率、两者之差。

    分箱边界取 12 位小数（`linspace` 的浮点误差会把 0.3 分到下一箱，见 ADR 0011）。
    """
    y, s = _arrays(y_true, score)
    edges = np.round(np.linspace(0.0, 1.0, bins + 1), 12)
    index = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, float]] = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": float(b),
                    "low": float(edges[b]),
                    "high": float(edges[b + 1]),
                    "count": 0.0,
                    "mean_score": 0.0,
                    "observed_rate": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        mean_score = float(s[mask].mean())
        observed = float(y[mask].mean())
        rows.append(
            {
                "bin": float(b),
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "count": float(count),
                "mean_score": mean_score,
                "observed_rate": observed,
                "gap": abs(mean_score - observed),
            }
        )
    return rows


def summary(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, bins: int = BINS
) -> dict[str, Any]:
    """校准摘要：平均分 vs 基率、Brier、ECE、以及最偏的一箱差多少。"""
    y, s = _arrays(y_true, score)
    table = reliability(y, s, bins)
    worst = max(table, key=lambda row: row["gap"] * row["count"])
    return {
        "n": float(len(y)),
        "base_rate": round(float(y.mean()), 6),
        "mean_score": round(float(s.mean()), 6),
        "brier": round(float(brier_score_loss(y, np.clip(s, 1e-6, 1 - 1e-6))), 6),
        "ece": round(M.expected_calibration_error(y, s, bins), 6),
        "worst_gap": round(float(worst["gap"]), 4),
        "worst_bin": f"[{worst['low']:.1f}, {worst['high']:.1f})",
    }


def render(summary_value: dict[str, Any], table: list[dict[str, float]]) -> list[str]:
    """把摘要与可靠性表渲染成 Markdown 行（caller 负责拼进报告）。"""
    lines = [
        "| 箱（分数区间） | 人数 | 平均分 | 实际正例率 | 差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in table:
        if row["count"] == 0:
            continue
        lines.append(
            f"| [{row['low']:.1f}, {row['high']:.1f}) | {row['count']:.0f} | "
            f"{row['mean_score']:.4f} | {row['observed_rate']:.4f} | {row['gap']:.4f} |"
        )
    tail = (
        f"平均分 {float(summary_value['mean_score']):.4f} vs "
        f"基率 {float(summary_value['base_rate']):.4f}；"
        f"Brier {float(summary_value['brier']):.4f}；"
        f"ECE {float(summary_value['ece']):.4f}；"
        f"最偏的一箱 {summary_value['worst_bin']} 差 "
        f"{float(summary_value['worst_gap']):.4f}。"
    )
    lines += ["", tail]
    return lines
