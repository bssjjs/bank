"""指标口径：名额内精确率/召回率、AP、ROC-AUC、Brier。

所有报告都必须带上"哪个集合、多少人、名额几个人"，这里把这三个数字一起返回，
免得下游只贴一个 AP 就完事。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, quota_n: int
) -> dict[str, float]:
    """按分数取前 `quota_n` 人作为"要打的名单"，算名额内指标。

    排序规则与线上一致：分数降序，同分按原行序（稳定排序），不随机。
    """
    y = np.asarray(y_true, dtype="int8")
    s = np.asarray(score, dtype="float64")
    if len(y) != len(s):
        raise ValueError(f"标签 {len(y)} 行与分数 {len(s)} 行不一致")
    if quota_n < 1 or quota_n > len(y):
        raise ValueError(f"名额 {quota_n} 超出验证集人数 {len(y)}")

    order = np.argsort(-s, kind="stable")
    picked = y[order[:quota_n]]
    positives = int(y.sum())
    base_rate = float(y.mean())
    tp = int(picked.sum())
    precision = tp / quota_n
    recall = tp / positives if positives else 0.0
    return {
        "n": float(len(y)),
        "positives": float(positives),
        "quota_n": float(quota_n),
        "base_rate": round(base_rate, 6),
        "ap": round(float(average_precision_score(y, s)), 6),
        "roc_auc": round(float(roc_auc_score(y, s)), 6),
        "precision_at_quota": round(precision, 6),
        "recall_at_quota": round(recall, 6),
        "lift_at_quota": round(precision / base_rate, 4) if base_rate else 0.0,
        "brier": round(float(brier_score_loss(y, np.clip(s, 1e-6, 1 - 1e-6))), 6),
        "score_mean": round(float(s.mean()), 6),
    }


def threshold_metrics(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, threshold: float
) -> dict[str, float]:
    """阈值 0.5 的对照口径：不是上线口径，只用来在文档里说明"为什么用名额不是阈值"。"""
    y = np.asarray(y_true, dtype="int8")
    s = np.asarray(score, dtype="float64")
    picked = s >= threshold
    tp = int((y[picked] == 1).sum())
    fp = int((y[picked] == 0).sum())
    fn = int((y[~picked] == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "selected": float(picked.sum()),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }
