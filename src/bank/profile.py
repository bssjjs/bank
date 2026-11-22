"""原始数据画像：M1 数据卡里的每个数字都由这里打印，避免文档与命令脱节。

一条命令：``python -m bank.profile``
产物：``data/processed/profile_raw.json``（不进 Git），终端打印关键事实。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from bank import schema as S
from bank import timeaxis as T

OUT_PATH = S.ROOT / "data" / "processed" / "profile_raw.json"


def load_ordered(db: Path = S.DB_PATH, table: str = S.RAW_TABLE) -> pd.DataFrame:
    """带行号读原始表。行号是时间序的载体，必须落进数据。"""
    with sqlite3.connect(db) as conn:
        return pd.read_sql(f'SELECT rowid AS "{T.SEQUENCE}", * FROM "{table}"', conn)


def with_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """补 年份 / 月份序_重建 / 重建日期 三列（不改变行序）。"""
    axis = T.build_time_axis(S.month_to_num(df[S.MONTH]).to_numpy(), df[S.DAY].to_numpy())
    return pd.concat([df.reset_index(drop=True), axis], axis=1)


def _half_year(dates: pd.Series) -> pd.Series:
    half = (dates.dt.month - 1) // 6
    return dates.dt.year.astype(str) + "H" + (half + 1).astype(str)


def profile(df: pd.DataFrame) -> dict:
    """把数据卡需要的全部事实算成字典。只读，不改 df。"""
    label = (df[S.TARGET_RAW] == S.POSITIVE).astype("int8")
    n = len(df)
    dates = df[T.DATE]

    month_keys = sorted(dates.dt.strftime("%Y-%m").unique())
    expected = pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M")
    missing = [str(p) for p in expected if str(p) not in set(month_keys)]

    hy = _half_year(dates)
    drift = {}
    for key, part in df.assign(_hy=hy, _label=label).groupby("_hy", sort=True):
        drift[str(key)] = {
            "rows": int(len(part)),
            "positives": int(part["_label"].sum()),
            "pos_rate": round(float(part["_label"].mean()), 4),
            "campaign_mean": round(float(part[S.CAMPAIGN].mean()), 3),
            "never_contacted_share": round(
                float((part[S.PDAYS] == S.PDAYS_SENTINEL).mean()), 4
            ),
            "poutcome_known_share": round(float((part[S.POUTCOME] != S.UNKNOWN).mean()), 4),
        }

    windows = {
        "A_2008-11_to_2010-11": _slice(dates, "2008-11-01", None, label),
        "B_2009-01_to_2010-11": _slice(dates, "2009-01-01", None, label),
    }
    splits = {
        "A_train_2008-11_to_2009-09": _slice(dates, "2008-11-01", "2009-10-01", label),
        "A_valid_2009-10_to_2010-11": _slice(dates, "2009-10-01", None, label),
        "B_train_2009-01_to_2009-12": _slice(dates, "2009-01-01", "2010-01-01", label),
        "B_valid_2010-01_to_2010-11": _slice(dates, "2010-01-01", None, label),
    }

    duration = df[S.DURATION].astype("float64")
    conflict = int(((df[S.PDAYS] != S.PDAYS_SENTINEL) & (df[S.POUTCOME] == S.UNKNOWN)).sum())

    return {
        "source": {
            "db": S.DB_PATH.name,
            "table": S.RAW_TABLE,
            "rows": n,
            "cols": int(df.shape[1] - 4),  # 去掉 序号/年份/月份序_重建/重建日期
            "db_bytes": S.DB_PATH.stat().st_size,
        },
        "label": {
            "rule": f'{S.TARGET_RAW} == "{S.POSITIVE}"',
            "positives": int(label.sum()),
            "pos_rate": round(float(label.mean()), 4),
        },
        "time": {
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "n_distinct_dates": int(dates.nunique()),
            "n_month_blocks": len(month_keys),
            "missing_months": missing,
            "monotonic": bool(dates.is_monotonic_increasing),
        },
        "drift_by_half_year": drift,
        "unknown_counts": {c: int((df[c] == S.UNKNOWN).sum()) for c in S.UNKNOWN_COLS},
        "pdays": {
            "never_contacted": int((df[S.PDAYS] == S.PDAYS_SENTINEL).sum()),
            "contacted": int((df[S.PDAYS] != S.PDAYS_SENTINEL).sum()),
            "conflict_pdays_set_but_poutcome_unknown": conflict,
        },
        "windows": windows,
        "splits": splits,
        "leak_duration": {
            "auc_univariate": round(float(roc_auc_score(label, duration)), 4),
            "mean": round(float(duration.mean()), 2),
            "p50": round(float(duration.median()), 2),
            "p95": round(float(duration.quantile(0.95)), 2),
            "max": int(duration.max()),
        },
    }


def _slice(dates: pd.Series, start: str, end: str | None, label: pd.Series) -> dict:
    mask = dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates < pd.Timestamp(end)
    sub = label[mask]
    return {
        "start": start,
        "end": end or "数据末尾",
        "rows": int(mask.sum()),
        "positives": int(sub.sum()),
        "pos_rate": round(float(sub.mean()), 4) if len(sub) else None,
    }


def summarize(facts: dict) -> list[str]:
    """人要读的几行，顺序与数据卡的叙述顺序一致。"""
    src, lab, tm = facts["source"], facts["label"], facts["time"]
    lines = [
        f"原始表：{src['table']}（{src['db']}，{src['db_bytes'] / 1_048_576:.1f} MiB）",
        f"行数 {src['rows']:,}，列数 {src['cols']}，正例 {lab['positives']:,}"
        f"（{lab['pos_rate']:.2%}）",
        f"重建时间轴：{tm['date_min']} → {tm['date_max']}，"
        f"{tm['n_distinct_dates']} 个日期、{tm['n_month_blocks']} 个月块，"
        f"缺失月份 {tm['missing_months']}，单调={tm['monotonic']}",
        f"从未被联系过 {facts['pdays']['never_contacted']:,} 行，"
        f"被联系过 {facts['pdays']['contacted']:,} 行，"
        f"pdays≠-1 但 poutcome=未知 {facts['pdays']['conflict_pdays_set_but_poutcome_unknown']} 行",
        f"通话时长单变量 AUC = {facts['leak_duration']['auc_univariate']}"
        f"（均值 {facts['leak_duration']['mean']}s，P95 {facts['leak_duration']['p95']}s）",
        "",
        "半年切片：",
    ]
    for key, v in facts["drift_by_half_year"].items():
        lines.append(
            f"  {key}  行 {v['rows']:>6,}  正例 {v['positives']:>5,}  "
            f"正例率 {v['pos_rate']:.2%}  本次联系次数均值 {v['campaign_mean']}  "
            f"从未联系占比 {v['never_contacted_share']:.1%}  "
            f"poutcome 已知占比 {v['poutcome_known_share']:.1%}"
        )
    lines.append("")
    lines.append("候选窗口与切分：")
    for name, v in {**facts["windows"], **facts["splits"]}.items():
        rate = "—" if v["pos_rate"] is None else f"{v['pos_rate']:.2%}"
        lines.append(
            f"  {name:<34} 行 {v['rows']:>6,}  正例 {v['positives']:>5,}  正例率 {rate}"
        )
    return lines


def main() -> None:
    raw = load_ordered()
    df = with_time_axis(raw)
    facts = profile(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in summarize(facts):
        print(line)
    print(f"\n写出 {OUT_PATH.relative_to(S.ROOT)}")


if __name__ == "__main__":
    main()