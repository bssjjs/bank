"""评估报告：13 个指标 + 两种名额口径 + 95% 区间 + 对照表，一份 Markdown 交给训练运行。

讲义（第二轮卷一 C-L2a）的要求：指标要报全、两种名额口径并列、区间要给出、
报告作为产物留在训练运行里。这里只负责"把数字摆成一张能读的表"，
算数全部在 `bank_training.metrics` 里，报告不重新计算任何指标。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bank_training import calibration as C
from bank_training import metric_specs as MS
from bank_training import metrics as M
from bank_training import slices as SL

#: 指标表（名字、中文名、显示格式）由 `metric_specs` 派生：
#: 中文名与小数位只登记在那里，报告、告警、模型卡才不会各说一套。
METRIC_ROWS: tuple[tuple[str, str, str], ...] = tuple(
    (item.name, item.label, f".{item.digits}f") for item in MS.METRIC_SPECS
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
    calibration: dict[str, float] = field(default_factory=dict)
    reliability: Sequence[dict[str, float]] = ()
    slices: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


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
    frame: pd.DataFrame | None = None,
) -> Report:
    """算全 13 指标、两种口径与区间，打包成 `Report`（纯函数，不写文件）。

    给了 `frame`（必须与 `y_true`/`score` 逐行对齐）就顺带算校准表与切片表；
    没给就只有指标部分，切片留空。
    """
    metrics = M.evaluate(y_true, score, quota_n)
    if intervals is None:
        intervals = M.bootstrap_ci(y_true, score, quota_n, samples=samples, seed=seed)
    calibration = C.summary(y_true, score)
    reliability = C.reliability(y_true, score)
    tables = SL.default_tables(frame, score=score, quota_n=quota_n) if frame is not None else {}
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
        calibration=calibration,
        reliability=tuple(reliability),
        slices=tables,
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
    if report.calibration:
        gap = report.calibration["base_rate"] - report.calibration["mean_score"]
        line_one = (
            f"- 平均分 {report.calibration['mean_score']:.4f}，"
            f"基率 {report.calibration['base_rate']:.4f}，两者差 {gap:.4f}"
        )
        line_two = (
            f"- Brier {report.calibration['brier']:.4f}，"
            f"ECE {report.calibration['ece']:.4f}，"
            f"最偏的一箱 {report.calibration['worst_bin']} 差 "
            f"{report.calibration['worst_gap']:.4f}"
        )
        line_three = (
            "- 只报告、不校准：验证集不许用来拟合校准器（那等于拿验证集调参），"
            "等线上攒够回流标签再单独做。"
        )
        lines += [
            "",
            "## 校准（分数说多少 vs 实际是多少）",
            "",
            line_one,
            line_two,
            line_three,
            "",
        ]
        lines += C.render(report.calibration, list(report.reliability))
    if report.slices:
        lines += ["", "## 切片（模型对谁不够好）", ""]
        lines += SL.render(report.slices)
    if report.notes:
        lines += ["", "## 怎么读这张表", ""]
        lines += [f"- {note}" for note in report.notes]
    lines.append("")
    return "\n".join(lines)


def write(path: Path, report: Report) -> Path:
    """把 Markdown 报告写到 `path`（UTF-8、LF、末尾恰好一个换行），返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8", newline="\n")
    return path


def write_json(path: Path, report: Report) -> Path:
    """把同一份 `Report` 写成机器可读的 JSON（UTF-8、LF、末尾恰好一个换行）。

    数字与 Markdown 报告同源：都是 `Report` 的字段，不在写文件时重算。
    用 `allow_nan=False` 让 NaN/Inf 当场报错，而不是写出不合法的 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        asdict(report),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    return path


def comparison(name: str, metrics: dict[str, float]) -> Comparison:
    """从 `M.evaluate` 的返回值造一行对照，顺手校验字段齐不齐。"""
    missing = [key for key in ("precision_at_quota", "recall_at_quota", "ap") if key not in metrics]
    if missing:
        raise KeyError(f"对照 {name} 缺少指标：{missing}")
    return Comparison(name=name, metrics=dict(metrics))
