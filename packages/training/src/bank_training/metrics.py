"""指标口径：名额内精确率/召回率、AP、ROC-AUC、Brier，以及两种名额口径与置信区间。

所有报告都必须带上"哪个集合、多少人、名额几个人"，这里把这三个数字一起返回，
免得下游只贴一个 AP 就完事。

口径约定（写进 `docs/model_card.md`，答辩要能背）：

- **名额口径**（上线口径）：按分数降序取前 `quota_n` 人，第 10 个名额以外的正例一律算漏掉。
- **排序口径**：ROC-AUC 与 AP，只看排序质量，不含名额。
- **校准口径**：Brier、对数损失、ECE，衡量"分数是不是概率"；本项目跨期漂移，分数偏保守，
  校准指标只报告、不设门槛（ADR 0005）。
- **阈值对照**：阈值 0.5 选中多少人、精确率多少，只用来解释"为什么用名额不用阈值"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from bank_training import metric_specs

SEED = 42
CI_SAMPLES = 200
CI_LEVEL = 0.95
ECE_BINS = 10

#: 13 个正式指标：名额口径 6 + 排序口径 2 + 校准口径 3 + 混淆矩阵 3（其中 3 个是人数）
METRIC_NAMES = (
    "precision_at_quota",
    "recall_at_quota",
    "f1_at_quota",
    "lift_at_quota",
    "accuracy_at_quota",
    "tp_at_quota",
    "fp_at_quota",
    "fn_at_quota",
    "roc_auc",
    "ap",
    "brier",
    "log_loss",
    "ece",
)

#: 导入时就校验「指标清单 == 指标出口登记表」：加指标忘了登记，这里先炸。
metric_specs.assert_complete(METRIC_NAMES)


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


def _picked(y: np.ndarray, score: np.ndarray, quota_n: int) -> np.ndarray:
    if quota_n < 1 or quota_n > len(y):
        raise ValueError(f"名额 {quota_n} 超出评估集人数 {len(y)}")
    order = np.argsort(-score, kind="stable")
    mask = np.zeros(len(y), dtype=bool)
    mask[order[:quota_n]] = True
    return mask


def confusion(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, quota_n: int
) -> dict[str, int]:
    """名额口径的混淆矩阵：前 `quota_n` 人算"要打的名单"，其余算"今天不打"。"""
    y, s = _arrays(y_true, score)
    picked = _picked(y, s, quota_n)
    tp = int((picked & (y == 1)).sum())
    fp = int((picked & (y == 0)).sum())
    fn = int((~picked & (y == 1)).sum())
    tn = int((~picked & (y == 0)).sum())
    if tp + fp != quota_n or tp + fn != int(y.sum()):
        raise AssertionError("混淆矩阵和名额/正例对不上")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def expected_calibration_error(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, bins: int = ECE_BINS
) -> float:
    """ECE：按分数等宽分箱，|平均分数 − 实际正例率| 的加权平均。"""
    y, s = _arrays(y_true, score)
    # 等宽分箱；边界取 12 位小数，避免 linspace 的浮点误差把 0.3 分到 0.2 那一箱
    edges = np.round(np.linspace(0.0, 1.0, bins + 1), 12)
    index = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        gap = abs(float(s[mask].mean()) - float(y[mask].mean()))
        total += gap * float(mask.mean())
    return float(total)


def evaluate(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, quota_n: int
) -> dict[str, float]:
    """13 个正式指标一次算全；同分按原行序（稳定排序），不随机。

    自洽关系（测试逐项核对）：``tp + fp == quota_n``、``tp + fn == 正例数``、
    ``precision = tp / quota_n``、``recall = tp / 正例数``、``f1 = 2PR/(P+R)``、
    ``lift = precision / 基率``、``accuracy = (tp + tn) / n``。
    """
    y, s = _arrays(y_true, score)
    cm = confusion(y, s, quota_n)
    n = len(y)
    positives = int(y.sum())
    base_rate = positives / n
    precision = cm["tp"] / quota_n
    recall = cm["tp"] / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clipped = np.clip(s, 1e-6, 1 - 1e-6)
    return {
        "n": float(n),
        "positives": float(positives),
        "quota_n": float(quota_n),
        "base_rate": round(base_rate, 6),
        "precision_at_quota": round(precision, 6),
        "recall_at_quota": round(recall, 6),
        "f1_at_quota": round(f1, 6),
        "lift_at_quota": round(precision / base_rate, 4) if base_rate else 0.0,
        "accuracy_at_quota": round((cm["tp"] + cm["tn"]) / n, 6),
        "tp_at_quota": float(cm["tp"]),
        "fp_at_quota": float(cm["fp"]),
        "fn_at_quota": float(cm["fn"]),
        "roc_auc": round(float(roc_auc_score(y, s)), 6),
        "ap": round(float(average_precision_score(y, s)), 6),
        "brier": round(float(brier_score_loss(y, clipped)), 6),
        "log_loss": round(float(log_loss(y, clipped, labels=[0, 1])), 6),
        "ece": round(expected_calibration_error(y, s), 6),
        "score_mean": round(float(s.mean()), 6),
    }


def threshold_metrics(
    y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray, threshold: float
) -> dict[str, float]:
    """阈值口径的对照：不是上线口径，只用来在文档里说明"为什么用名额不是阈值"。"""
    y, s = _arrays(y_true, score)
    picked = s >= threshold
    tp = int((y[picked] == 1).sum())
    fp = int((y[picked] == 0).sum())
    fn = int((y[~picked] == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "selected": float(picked.sum()),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def two_views(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    quota_n: int,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """两种名额口径并列：前 k 名 vs 固定阈值。两人数通常不同，报告里必须分开写。"""
    y, s = _arrays(y_true, score)
    quota = evaluate(y, s, quota_n)
    return {
        "top_k": {
            "selected": float(quota_n),
            "precision": quota["precision_at_quota"],
            "recall": quota["recall_at_quota"],
            "f1": quota["f1_at_quota"],
        },
        "fixed_threshold": threshold_metrics(y, s, threshold),
    }


def bootstrap_ci(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    quota_n: int,
    *,
    samples: int = CI_SAMPLES,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> dict[str, tuple[float, float]]:
    """按行重抽的 95% 区间：名额掩码跟着行一起抽，名额按比例缩放（讲义 C-L2a）。

    每次重抽都把该行的 (score, label) 一起带走，所以「第 k 名」是重抽后的第 k 名，
    等价于线上的同一套排序逻辑；名额按 `ceil(n_boot × frac)` 缩放，保持名额比例不变。
    """
    y, s = _arrays(y_true, score)
    n = len(y)
    frac = quota_n / n
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(samples):
        index = rng.integers(0, n, n)
        y_boot, s_boot = y[index], s[index]
        k_boot = max(1, min(n, int(np.ceil(round(n * frac, 9)))))
        metrics = evaluate(y_boot, s_boot, k_boot)
        for name in METRIC_NAMES:
            values[name].append(float(metrics[name]))
    alpha = (1.0 - level) / 2.0
    return {
        name: (float(np.quantile(series, alpha)), float(np.quantile(series, 1 - alpha)))
        for name, series in values.items()
    }


def format_ci(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"
