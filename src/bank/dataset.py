"""数据集构建层：把口径规则套到原始库上，产出可建模的数据集与特征清单。

分工：
- schema.py  只放常量与纯函数（可单测、换数据不变）
- dataset.py 负责读库、串联规则、自检、写 parquet / feature_spec.json

原始库全程只读，不写不改不删表。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from bank import schema as S

DATASET_PATH = S.ROOT / "data" / "model_dataset.parquet"
SPEC_PATH = S.ROOT / "data" / "feature_spec.json"


def load_raw(db: Path = S.DB_PATH, table: str = S.RAW_TABLE) -> pd.DataFrame:
    """全量读入原始表。库只有一张表，不加筛选条件。"""
    with sqlite3.connect(db) as conn:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)


def build_dataset(raw: pd.DataFrame | None = None, db: Path = S.DB_PATH) -> pd.DataFrame:
    """套用全部口径规则。行数与顺序保持不变，只新增列，不删原始列。

    新增列：
      label            目标 0/1
      是否被联系过      pdays != -1
      距上次联系天数_log  log1p，未联系记 0
      距上次联系天数档   0=未联系, 1~5 按 30/90/180/365 分档
      信息完整度        3 - 未知个数
      教育程度序        小学/中学/高等教育 = 1/2/3，未知留空
      教育程度_未知     哑变量
      月份序/月份_sin/月份_cos/日期_sin/日期_cos
    """
    df = load_raw(db) if raw is None else raw.copy()

    df[S.TARGET] = (df[S.TARGET_RAW] == S.POSITIVE).astype("int8")

    df[S.CONTACTED] = S.pdays_contacted(df[S.PDAYS])
    df[S.PDAYS_LOG] = S.pdays_log(df[S.PDAYS])
    df[S.PDAYS_BUCKET] = S.pdays_bucket(df[S.PDAYS])

    df[S.INFO_COMPLETENESS] = S.info_completeness(df)

    df[S.EDU_ORDINAL] = S.edu_ordinal(df[S.EDU])
    df[S.EDU_UNKNOWN_FLAG] = S.edu_unknown(df[S.EDU])

    df[S.MONTH_ORDINAL] = S.month_to_num(df[S.MONTH])
    df[S.MONTH_SIN], df[S.MONTH_COS] = S.cyclical(df[S.MONTH_ORDINAL], 12)
    df[S.DAY_SIN], df[S.DAY_COS] = S.cyclical(df[S.DAY], S.DAY_PERIOD)

    return df


def model_features(version: str = "A") -> list[str]:
    """A 版含时间（离线最优），B 版剔除月份（可上线）。

    A/B 差值就是"时期红利"的量化，两份都要跑，不能只报 A 版。
    """
    if version not in {"A", "B"}:
        raise ValueError(f"version 只能是 A 或 B，收到 {version!r}")

    cols: list[str] = list(S.NUMERIC) + list(S.CATEGORICAL) + list(S.ORDINAL)
    cols += [S.EDU_ORDINAL]
    cols += list(S.TIME_COLS if version == "A" else S.DAY_KEEP)

    # 排除列即使被误写进分组也不许出现在最终清单里
    leaked = [c for c in cols if c in S.EXCLUDE_COLS]
    if leaked:
        raise AssertionError(f"排除列混进特征清单：{leaked}")
    return cols


def validate(df: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """确定性自检：口径层做错时这里必须报错，而不是让模型静默学歪。"""
    n = len(raw)
    if len(df) != n:
        raise AssertionError(f"行数变化：{n} → {len(df)}")

    n_pos_raw = int((raw[S.TARGET_RAW] == S.POSITIVE).sum())
    if int(df[S.TARGET].sum()) != n_pos_raw:
        raise AssertionError("标签合计与原始「是否认购=是」不一致")

    n_contacted = int((raw[S.PDAYS] != S.PDAYS_SENTINEL).sum())
    if int(df[S.CONTACTED].sum()) != n_contacted:
        raise AssertionError("是否被联系过 与 pdays != -1 条数不一致")

    n_never = int((raw[S.PDAYS] == S.PDAYS_SENTINEL).sum())
    if int((df[S.PDAYS_LOG] == 0).sum()) != n_never:
        raise AssertionError("_log 取 0 的条数不等于从未联系条数，存在 0 碰撞")

    # log 列在"联系过"的行上必须严格大于 0，否则 0 就不唯一
    touched = df.loc[df[S.CONTACTED] == 1, S.PDAYS_LOG]
    if not (touched > 0).all():
        raise AssertionError("联系过的行出现 _log <= 0，0 不再唯一")

    if not df[S.PDAYS_BUCKET].between(0, 5).all():
        raise AssertionError("分档出现 0~5 之外的值")

    # 分档 0 当且仅当未联系；其余档必须被联系过
    if int((df[S.PDAYS_BUCKET] == 0).sum()) != n_never:
        raise AssertionError("分档为 0 的条数不等于从未联系条数")

    if df[S.INFO_COMPLETENESS].nunique() > 4:
        raise AssertionError("信息完整度超出 0~3")

    if df[S.EDU_ORDINAL].notna().sum() != int((raw[S.EDU] != S.UNKNOWN).sum()):
        raise AssertionError("教育程度序的非空条数与真实学历条数不一致")

    if df[S.MONTH_ORDINAL].between(1, 12).sum() != n:
        raise AssertionError("月份序存在 1~12 之外的值")

    return {
        "n_rows": n,
        "n_cols_raw": raw.shape[1],
        "n_cols_out": df.shape[1],
        "n_positive": n_pos_raw,
        "pos_rate": round(float(df[S.TARGET].mean()), 4),
        "n_contacted": n_contacted,
        "n_never_contacted": n_never,
        "n_edu_unknown": int(df[S.EDU_UNKNOWN_FLAG].sum()),
        "pdays_bucket_counts": {
            S.PDAYS_BUCKET_LABELS[k]: int(v)
            for k, v in df[S.PDAYS_BUCKET].value_counts().sort_index().items()
        },
    }


def build_spec(df: pd.DataFrame, facts: dict) -> dict:
    """特征清单落成 JSON，供建模阶段直接读，避免两边各写一份列名。"""
    return {
        "source": {
            "db": S.DB_PATH.name,
            "table": S.RAW_TABLE,
            "raw_rows": facts["n_rows"],
            "raw_cols": facts["n_cols_raw"],
        },
        "target": {
            "column": S.TARGET,
            "positive_rule": f'{S.TARGET_RAW} == "{S.POSITIVE}"',
            "n_positive": facts["n_positive"],
            "pos_rate": facts["pos_rate"],
        },
        "versions": {
            "A_with_time": {
                "n_features": len(model_features("A")),
                "columns": model_features("A"),
                "note": "含月份/季节编码，离线指标最优；月份学到的是时期而非客户属性，不可外推",
            },
            "B_deployable": {
                "n_features": len(model_features("B")),
                "columns": model_features("B"),
                "note": f"剔除 {list(S.TIME_LEAKY)}，用于上线评估；A-B 差值即时期红利",
            },
        },
        "groups": {
            "numeric": list(S.NUMERIC),
            "categorical": list(S.CATEGORICAL),
            "ordinal": {
                S.PDAYS_BUCKET: {
                    "range": [0, 5],
                    "labels": {str(k): v for k, v in S.PDAYS_BUCKET_LABELS.items()},
                    "usage": "树模型用整数，线性模型用 one-hot（每档自由，不假设等差）",
                },
                S.EDU_ORDINAL: {
                    "range": [1, 3],
                    "labels": S.EDU_ORDER,
                    "usage": "未知留空，由 教育程度_未知 表达，不占顺序位",
                },
            },
            "time": list(S.TIME_COLS),
        },
        "excluded": {
            c: S.EXCLUDE_REASON[c] for c in S.EXCLUDE_COLS
        },
        "rules": {
            "pdays_sentinel": {
                "value": S.PDAYS_SENTINEL,
                "meaning": "从未被联系过",
                "derived": [S.CONTACTED, S.PDAYS_LOG, S.PDAYS_BUCKET],
                "bounds": list(S.PDAYS_BOUNDS),
            },
            "unknown_values": {
                "columns": list(S.UNKNOWN_COLS),
                "policy": "保留为独立类别，不填补不删除；派生信息完整度",
                "counts": {c: int((df[c] == S.UNKNOWN).sum()) for c in S.UNKNOWN_COLS},
            },
        },
        "facts": facts,
        "pipeline_side": {
            "note": "以下统计量必须 fit 在训练集，不得写进本数据集",
            "items": [
                "账户余额 / 本次联系次数 / 历史联系次数 的 1% 与 99% 分位裁剪阈值",
                "数值列标准化参数",
                "独热编码的类别表（handle_unknown='ignore'）",
            ],
        },
    }


def write_outputs(df: pd.DataFrame, spec: dict) -> tuple[Path, Path]:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    SPEC_PATH.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return DATASET_PATH, SPEC_PATH
