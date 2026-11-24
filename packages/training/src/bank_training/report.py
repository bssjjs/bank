"""评估报告：13 个指标 + 两种名额口径 + 95% 区间 + 对照表，一份 Markdown 交给训练运行。

讲义（第二轮卷一 C-L2a）的要求：指标要报全、两种名额口径并列、区间要给出、
报告作为产物留在训练运行里。这里只负责"把数字摆成一张能读的表"，
算数全部在 `bank_training.metrics` 里，报告不重新计算任何指标。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bank_training import metrics as M

#: 指标的中文说明与显示格式；顺序即报告里的顺序。
METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("precision_at_quota", "名额内精确率", ".4f"),
    ("recall_at_quota", "名额内召回率", ".4f"),
    ("f1_at_quota", "名额内 F1", ".4f"),
    ("lift_at_quota", "名额内提升（相对基率）", ".4f"),
    ("accuracy_at_quota", "名额内准确率", ".4f"),
    ("tp_at_quota", "名额内打中的正例", ".0f"),
    ("fp_at_quota", "名额内多打的人", ".0f"),
    ("fn_at_quota", "名额外漏掉的正例", ".0f"),
    ("roc_auc", "ROC-AUC", ".4f"),
    ("ap", "AP（平均精确率）", ".4f"),
    ("brier", "Brier（越小越好）", ".4f"),
    ("log_loss", "对数损失（越小越好）", ".4f"),
    ("ece", "ECE（校准误差，越小越好）", ".4f"),
)


@dataclass(frozen=True)
class Comparison:
    """一行对照：别人/别的模型在同一评估集、同一名额下的数字。"""

    name: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class Report:
    title: str
    n: int
    positives: int
    quota_n: int
    base_rate: float
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    views: dict[str, dict[str, float]] = field(default_factory=dict)
    comparisons: Sequence[Comparison] = ()
    notes: Sequence[str] = ()


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt(value: float, spec: str) -> str:
    return format(float(value), spec)


def build(
    *,
    title: str,
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    quota_n: int,
    comparisons: Sequence[Comparison] = (),
    notes: Sequence[str] = (),
    intervals: dict[str, tuple[float, float]] | None = None,
    samples: int = M.CI_SAMPLES,
    seed: int = M.SEED,
) -> Report:
    """算全 13 指标、两种口径与区间，打包成 `Report`（纯函数，不写文件）。"""
    metrics = M.evaluate(y_true, score, quota_n)
    if intervals is None:
        intervals = M.bootstrap_ci(y_true, score, quota_n, samples=samples, seed=seed)
    return Report(
        title=title,
        n=int(metrics["n"]),
        positives=int(metrics["positives"]),
        quota_n=int(quota_n),
        base_rate=metrics["base_rate"],
        metrics=metrics,
        intervals=intervals,
        views=M.two_views(y_true, score, quota_n),
        comparisons=tuple(comparisons),
        notes=tuple(notes),
    )


def render(report: Report) -> str:
    """把 `Report` 渲染成 Markdown；所有数字都来自 `Report.metrics`，不在这里重算。"""
    lines: list[str] = [
        f"# {report.title}",
        "",
        "## 口径",
        "",
        f"- 评估集：{report.n:,} 行，正例 {report.positives:,}（基率 {_pct(report.base_rate)}）",
        f"- 名额：**{report.quota_n} 人**（按分数降序取前 {report.quota_n} 名，同分按原行序）",
        "- 指标定义见 `bank_training/metrics.py`；13 个指标同源，测试核对自洽。",
        "",
        "## 13 个指标（名额口径 + 排序口径 + 校准口径）",
        "",
        "| 指标 | 值 | 95% 区间 |",
        "| --- | --- | --- |",
    ]
    for name, label, spec in METRIC_ROWS:
        value = report.metrics[name]
        interval = report.intervals.get(name)
        if interval is None:
            cell = "—"
        elif spec == ".0f":
            cell = f"[{interval[0]:.0f}, {interval[1]:.0f}]"
        else:
            cell = M.format_ci(interval)
        lines.append(f"| {label}（`{name}`） | {_fmt(value, spec)} | {cell} |")
    lines += [
        "",
        f"对照：名额平均分 {report.metrics['score_mean']:.4f}，实际基率 {report.base_rate:.4f}",
        "分数系统性偏低（时期漂移），原因见模型卡的「校准与漂移」。",
        "",
        "## 两种名额口径（同一份分数，选中人数不同）",
        "",
        "| 口径 | 选中人数 | 精确率 | 召回率 | F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in (("top_k", "前 k 名（上线口径）"), ("fixed_threshold", "固定阈值 0.5")):
        view = report.views[key]
        lines.append(
            f"| {label} | {view['selected']:.0f} | {view['precision']:.4f} | "
            f"{view['recall']:.4f} | {view['f1']:.4f} |"
        )
    if report.comparisons:
        lines += [
            "",
            "## 对照（同一评估集、同一名额）",
            "",
            "| 对象 | 名额内精确率 | 名额内召回率 | AP | ROC-AUC | Brier | 名额内打中 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in report.comparisons:
            m = row.metrics
            lines.append(
                f"| {row.name} | {m['precision_at_quota']:.4f} | {m['recall_at_quota']:.4f} | "
                f"{m['ap']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} | "
                f"{m['tp_at_quota']:.0f} |"
            )
    if report.notes:
        lines += ["", "## 怎么读这张表", ""]
        lines += [f"- {note}" for note in report.notes]
    lines.append("")
    return "\n".join(lines)


def write(path: Path, report: Report) -> Path:
    """把报告写到 `path`（UTF-8、LF、末尾恰好一个换行），返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8", newline="\n")
    return path


def comparison(name: str, metrics: dict[str, float]) -> Comparison:
    """从 `M.evaluate` 的返回值造一行对照，顺手校验字段齐不齐。"""
    missing = [key for key in ("precision_at_quota", "recall_at_quota", "ap") if key not in metrics]
    if missing:
        raise KeyError(f"对照 {name} 缺少指标：{missing}")
    return Comparison(name=name, metrics=dict(metrics))
