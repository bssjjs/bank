"""切片评估：模型对谁不够好。

名额是全局的（每天 356 人），所以切片里的名额按该切片人数比例分配：
`k_slice = max(1, round(n_slice × 名额比例))`。这样做的好处是每个切片都回答同一个问题
——「如果今天只从这群人里按比例取人，能打中几个」；坏处是切片越小名额越少，数字越抖，
所以小于 `MIN_SIZE` 的切片不报（在表里留一行说明被跳过）。

切片维度来自 `configs/features.yaml` 里的原始列，不新造特征；年龄用 5 档粗分箱。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from bank import schema as S

from bank_training import metrics as M

MIN_SIZE = 100

AGE_BAND = "年龄档"
AGE_EDGES = (0, 30, 40, 50, 60, 200)
AGE_LABELS = ("18-29", "30-39", "40-49", "50-59", "60+")

DEFAULT_SLICES = (S.OCCUPATION, S.EDU, S.CONTACT, S.MARITAL, AGE_BAND)


def age_band(frame: pd.DataFrame) -> pd.Series:
    """把年龄切成 5 档；18 岁以下（不该出现）落进第一档，不丢行。"""
    return pd.cut(frame[S.AGE], bins=list(AGE_EDGES), labels=list(AGE_LABELS), right=False).astype(
        str
    )


def _values(frame: pd.DataFrame, by: str) -> pd.Series:
    if by == AGE_BAND:
        return age_band(frame).astype(str)
    if by not in frame.columns:
        raise KeyError(f"切片列 {by} 不在评估集里；检查 configs/features.yaml")
    return frame[by].astype(str)


def slice_table(
    frame: pd.DataFrame,
    *,
    score: pd.Series | np.ndarray,
    quota_n: int,
    by: str,
    min_size: int = MIN_SIZE,
) -> list[dict[str, Any]]:
    """按 `by` 列切片，每片报人数、基率、名额、名额内精确率/召回率、AP。

    `score` 必须与 `frame` 逐行对齐（调用方保证）；名额按比例分配、四舍五入到整数。
    """
    y = frame[S.TARGET].to_numpy(dtype="int8")
    s = np.asarray(score, dtype="float64")
    if len(y) != len(s):
        raise ValueError(f"评估集 {len(y)} 行与分数 {len(s)} 行不一致")
    if quota_n < 1 or quota_n > len(y):
        raise ValueError(f"名额 {quota_n} 超出评估集 {len(y)} 人")
    groups = _values(frame, by)
    ratio = quota_n / len(y)
    rows: list[dict[str, Any]] = []
    for value in sorted(groups.unique()):
        mask = (groups == value).to_numpy()
        n = int(mask.sum())
        if n < min_size:
            rows.append({"slice": by, "value": value, "n": n, "skipped": True})
            continue
        k = max(1, min(n, round(n * ratio)))
        metrics = M.evaluate(y[mask], s[mask], k)
        rows.append(
            {
                "slice": by,
                "value": value,
                "n": n,
                "skipped": False,
                "positives": int(metrics["positives"]),
                "base_rate": metrics["base_rate"],
                "quota": k,
                "precision": metrics["precision_at_quota"],
                "recall": metrics["recall_at_quota"],
                "ap": metrics["ap"],
            }
        )
    return rows


def default_tables(
    frame: pd.DataFrame,
    *,
    score: pd.Series | np.ndarray,
    quota_n: int,
    min_size: int = MIN_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    """跑 `DEFAULT_SLICES` 里的全部维度，返回 `{维度: 行列表}`。"""
    return {
        by: slice_table(frame, score=score, quota_n=quota_n, by=by, min_size=min_size)
        for by in DEFAULT_SLICES
    }


def render(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    """把切片表渲染成 Markdown 行；被跳过的切片单独列一行说明。"""
    lines: list[str] = []
    for by, rows in tables.items():
        lines += [
            f"### 按{by}",
            "",
            "| 取值 | 人数 | 基率 | 名额 | 名额内精确率 | 名额内召回率 | AP |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        skipped: list[str] = []
        for row in rows:
            if row["skipped"]:
                skipped.append(f"{row['value']}（{row['n']} 人）")
                continue
            lines.append(
                f"| {row['value']} | {row['n']} | {float(row['base_rate']):.4f} | "
                f"{row['quota']} | {float(row['precision']):.4f} | "
                f"{float(row['recall']):.4f} | {float(row['ap']):.4f} |"
            )
        if skipped:
            lines.append("")
            lines.append(f"少于 {MIN_SIZE} 人的切片不报：{'、'.join(skipped)}。")
        lines.append("")
    return lines
