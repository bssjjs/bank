"""全局解释：模型的"靠什么排序"落到整张表，而不是单条。

单条解释（`bank.explain`）回答"这一位为什么是这个分"；全局解释回答"整体上哪些列
在左右排序"。两种口径，互相印证：

1. **线性贡献重要度**：对验证集每一行算「系数 × 标准化后取值」（与单条解释同一套等式，
   与线性 SHAP 等价），取各行 |贡献| 的平均作为该列的全局重要度。它回答「分数的变动
   有多少来自这一列」，正负方向也能读出来。
2. **置换重要度**：把某一列打乱后重算 AP 的下降幅度，下降越多说明模型越依赖它。
   它不假设线性，但每列要重算多次（种子固定，结果可复现）。

两套重要度并列，是为了不把「线性模型眼里的重要」当成「模型真的依赖」——置换重要度
是模型无关的参照（讲义 C-L1b 的全局解释要求）。

输出：`data/processed/global_explain.md`（表 + 结论）+ `global_explain_beeswarm.png`
（蜂群图，每点一行、按列分组、颜色随贡献正负），并作为产物记进主模型 run。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import load_catalog
from bank.config import load_config
from bank.explain import column_names
from sklearn.pipeline import Pipeline

from bank_training import metrics as M
from bank_training.data import load_frozen, split_xy
from bank_training.model_config import load_model_config
from bank_training.pipeline import build_model

OUT_DIR = S.ROOT / "data" / "processed"
OUT_MD = OUT_DIR / "global_explain.md"
OUT_PNG = OUT_DIR / "global_explain_beeswarm.png"

#: 置换重要度的重复次数与种子：固定种子保证同一批数据重跑结果一致。
PERM_REPEATS = 5
PERM_SEED = 42

#: 蜂群图最多画的行数（验证集 3,556 行全画会糊成一团，抽样到可读规模）。
BEESWARM_MAX_POINTS = 500

#: 报告里只列前 N 列，其余折叠成"其它"。
TOP_N = 12


@dataclass(frozen=True)
class GlobalRow:
    """一行全局解释：一个特征的两套重要度与方向。"""

    feature: str
    linear_importance: float
    linear_direction: str
    perm_importance: float


def _permutation_importance(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    quota_n: int,
    *,
    repeats: int = PERM_REPEATS,
    seed: int = PERM_SEED,
) -> dict[str, float]:
    """逐列打乱后算 AP 下降量；下降越大越依赖。返回 {特征: 平均下降}。

    用 AP 而不是准确率：这是排序问题，AP 才是与名额口径一致的排序指标。
    """
    baseline = M.evaluate(y.to_numpy(), model.predict_proba(x)[:, 1], quota_n)["ap"]
    rng = np.random.default_rng(seed)
    drops: dict[str, list[float]] = {}
    for _ in range(repeats):
        for col in x.columns:
            shuffled = x.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            score = model.predict_proba(shuffled)[:, 1]
            ap = M.evaluate(y.to_numpy(), score, quota_n)["ap"]
            drops.setdefault(col, []).append(baseline - ap)
    return {col: float(np.mean(values)) for col, values in drops.items()}


def _raw_column(name: str) -> str:
    """one-hot 展开后的列名形如 `职业=蓝领工人`，取 `=` 前的原始列名归并。"""
    return name.split("=")[0]


def _group_by_raw(names: list[str], per_row: np.ndarray) -> tuple[list[str], np.ndarray]:
    """把 one-hot 展开的逐行贡献按原始列归并（同一列的多个 one-hot 项贡献相加）。

    这样线性重要度与置换重要度都在「原始列」这个粒度上，两套口径才可比。
    """
    raw_names: list[str] = []
    grouped: list[np.ndarray] = []
    for name in names:
        raw = _raw_column(name)
        if raw not in raw_names:
            raw_names.append(raw)
            grouped.append(np.zeros(per_row.shape[0], dtype="float64"))
        grouped[raw_names.index(raw)] += per_row[:, names.index(name)]
    return raw_names, np.column_stack(grouped)


def global_rows(model: Pipeline, x: pd.DataFrame, y: pd.Series, quota_n: int) -> list[GlobalRow]:
    """算线性重要度（平均 |贡献|，带方向）与置换重要度，按置换重要度降序。"""
    pre = model.steps[0][1]
    lr = model.steps[-1][1]
    names = column_names(pre)
    transformed = np.asarray(pre.transform(x))
    coef = np.asarray(getattr(lr, "coef_")).reshape(-1)
    per_row = coef[None, :] * transformed
    raw_names, grouped = _group_by_raw(names, per_row)
    abs_mean = np.abs(grouped).mean(axis=0)
    direction = np.sign(grouped.sum(axis=0))
    perm = _permutation_importance(model, x, y, quota_n)

    rows: list[GlobalRow] = []
    for i, name in enumerate(raw_names):
        rows.append(
            GlobalRow(
                feature=name,
                linear_importance=float(abs_mean[i]),
                linear_direction="+" if direction[i] >= 0 else "−",
                perm_importance=perm.get(name, 0.0),
            )
        )
    # 主序按置换重要度（模型无关参照）降序；并列再按线性重要度。
    rows.sort(key=lambda r: (r.perm_importance, r.linear_importance), reverse=True)
    return rows


def render(rows: list[GlobalRow], *, quota_n: int, n: int) -> str:
    """把全局解释渲染成 Markdown（表 + 结论）。"""
    lines: list[str] = [
        "# 全局解释：模型靠什么排序",
        "",
        f"- 评估集：验证集 {n:,} 人，名额 {quota_n} 人；主模型逻辑回归 + 特征集 B。",
        "- 线性重要度 = 各行 |系数 × 标准化取值| 的平均（与单条解释同一套等式，等价线性 SHAP）；",
        "  「方向」是贡献均值的符号，+ 推高分数、− 压低分数。",
        "- 置换重要度 = 打乱该列后 AP 的下降量（重复 5 次取平均，种子 42），模型无关参照。",
        "",
        "| 特征 | 线性重要度 | 方向 | 置换重要度（ΔAP） |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows[:TOP_N]:
        lines.append(
            f"| {row.feature} | {row.linear_importance:.4f} | {row.linear_direction} | "
            f"{row.perm_importance:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _beeswarm(model: Pipeline, x: pd.DataFrame) -> None:
    """画蜂群图：每点一行、按列分组、颜色随贡献正负，纵轴是 |贡献| 的抖动。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 中文字体：Windows 用 Microsoft YaHei，取不到再退回默认（避免图形里全是方框）。
        for candidate in ("Microsoft YaHei", "SimHei"):
            if any(candidate == f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [candidate]
                plt.rcParams["axes.unicode_minus"] = False
                break
    except ImportError as exc:  # 图形是锦上添花，缺 matplotlib 不影响文字结论
        print(f"跳过蜂群图：{exc}")
        return

    pre = model.steps[0][1]
    lr = model.steps[-1][1]
    names = column_names(pre)
    coef = np.asarray(getattr(lr, "coef_")).reshape(-1)
    transformed = np.asarray(pre.transform(x))
    per_row = coef[None, :] * transformed
    raw_names, grouped = _group_by_raw(names, per_row)
    rng = np.random.default_rng(PERM_SEED)
    idx = rng.choice(
        grouped.shape[0], size=min(BEESWARM_MAX_POINTS, grouped.shape[0]), replace=False
    )
    sample = grouped[idx]

    top = min(TOP_N, sample.shape[1])
    order = np.argsort(-np.abs(sample).mean(axis=0))[:top]
    cols = [raw_names[i] for i in order]

    # 每列各自 z 标准化，否则贡献最大的那一列会把其它列压成贴着 0 的一条线。
    # 标准化只改显示尺度，不改排序，也不改文字结论里的原始重要度数字。
    std = sample.std(axis=0)
    std[std == 0] = 1.0
    scaled = sample / std

    fig, ax = plt.subplots(figsize=(11, 6))
    for j, name in enumerate(cols):
        vals = scaled[:, order[j]]
        jitter = rng.uniform(-0.4, 0.4, size=len(vals))
        colors = ["#b3261e" if sample[i, order[j]] >= 0 else "#1e6fe6" for i in range(len(vals))]
        ax.scatter(jitter, vals, s=8, c=colors, alpha=0.5, linewidths=0)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("单行贡献（各列各自标准化，便于跨列比较）")
    ax.set_title("蜂群图：每个点一行验证样本，红=推高分数、蓝=压低分数")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def run_for(model: Pipeline, x: pd.DataFrame, y: pd.Series, quota_n: int) -> tuple[Any, Any]:
    """给定已训练的模型与评估集，产出 (md 路径, png 路径或 None)。供 `train.py` 复用。"""
    rows = global_rows(model, x, y, quota_n)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(rows, quota_n=quota_n, n=len(y)), encoding="utf-8", newline="\n")
    _beeswarm(model, x)
    return OUT_MD, (OUT_PNG if OUT_PNG.exists() else None)


def main() -> int:
    cfg = load_config()
    train, valid, manifest = load_frozen()
    mcfg = load_model_config()
    spec_ = load_catalog()
    feature_set = cfg.features.set
    quota_n = int(manifest["capacity"]["quota_n"])

    x_train, y_train = split_xy(train, feature_set)
    x_valid, y_valid = split_xy(valid, feature_set)
    model = build_model(
        mcfg.model.kind, spec_, list(x_train.columns), c=mcfg.model.c, max_iter=mcfg.model.max_iter
    )
    model.fit(x_train, y_train)

    md, png = run_for(model, x_valid, y_valid, quota_n)
    print(f"全局解释：{md.relative_to(S.ROOT)}")
    if png is not None:
        print(f"蜂群图：{png.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
