"""口径规则层：所有与数据分布无关的确定性转换都写在这里。

为什么单独一层：这里的每条规则换一批数据也不会变（-1 永远是"从未联系"、
"五月"永远是第 5 月），所以可以一次性算好落盘、写文档、做单测。
依赖训练集分布的统计量（分位裁剪阈值、缺失填补值）不在这里，它们必须在
sklearn Pipeline 里 fit，否则测试集信息会回流到训练过程。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 路径与表
def _find_root() -> Path:
    """往上找仓库根：认 `configs/base.yaml`，不数目录层级。

    原因：代码从 `src/bank` 搬进 `packages/core/src/bank` 后，`parents[N]` 的
    N 会变；用标记文件找根，搬家、装成 wheel、在别处 checkout 都不会错。
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "base.yaml").is_file():
            return parent
    raise RuntimeError("找不到仓库根（缺 configs/base.yaml），拒绝猜路径")


ROOT = _find_root()
DB_PATH = ROOT / "data" / "bank-marketing.db"
RAW_TABLE = "银行营销数据"
SEED = 42

# ---------------------------------------------------------------- 目标列
TARGET_RAW = "是否认购"
TARGET = "label"
POSITIVE = "是"
NEGATIVE = "否"

# ---------------------------------------------------------------- 其它口径列
CAMPAIGN = "本次联系次数"
PREVIOUS = "历史联系次数"
POUTCOME = "上次营销结果"
CONTACT = "联系方式"
OCCUPATION = "职业"
MARITAL = "婚姻状态"
AGE = "年龄"

# ---------------------------------------------------------------- 哨兵值
# 「距上次联系天数」用 -1 表示从未被联系过，占 81.74%。
# 直接入模有两个错法：树模型会把 -1 当成"最近联系过"（比 1 天前更近），
# 线性模型会得到负天数。所以原列拆成三列后不再入模。
PDAYS = "距上次联系天数"
PDAYS_SENTINEL = -1
PDAYS_LOG = "距上次联系天数_log"
PDAYS_BUCKET = "距上次联系天数档"
CONTACTED = "是否被联系过"

# 分档上界：业务阈值（月/季/半年/一年），不是分位数 → 属于口径层。
# 实测认购率不单调（31~90 天 42.1% > 91~180 天 27.4% > 181~365 天 17.8%），
# 所以除了 log 连续列，还要保留分档让模型能表达高低起伏。
PDAYS_BOUNDS = (30, 90, 180, 365)
PDAYS_BUCKET_LABELS = {
    0: "未联系",
    1: "(0,30]",
    2: "(30,90]",
    3: "(90,180]",
    4: "(180,365]",
    5: "365+",
}

# ---------------------------------------------------------------- 未知值
# 这三列的「未知」是真实客群标签，不是缺失：联系方式=未知 13,020 条（28.80%），
# 认购率 4.07%，是全表最低的一档。填成众数等于把最难触达的人群改造成普通人。
UNKNOWN = "未知"
UNKNOWN_COLS = ("职业", "教育程度", "联系方式")
INFO_COMPLETENESS = "信息完整度"

# ---------------------------------------------------------------- 有序类别
# 教育程度不能直接按 1/2/3 编码：实测认购率 小学 8.6% < 中学 10.6%
# < 未知 13.6% < 高等教育 15.0%，「未知」夹在中间而不是最低档。
# 所以有序编码只授予三档真实学历，未知单独一个哑变量。
EDU = "教育程度"
EDU_ORDINAL = "教育程度序"
EDU_UNKNOWN_FLAG = "教育程度_未知"
EDU_ORDER = {"小学": 1, "中学": 2, "高等教育": 3}

MONTH = "联系月份"
MONTH_ORDINAL = "月份序"
MONTH_SIN = "月份_sin"
MONTH_COS = "月份_cos"
MONTH_ORDER = {
    "一月": 1,
    "二月": 2,
    "三月": 3,
    "四月": 4,
    "五月": 5,
    "六月": 6,
    "七月": 7,
    "八月": 8,
    "九月": 9,
    "十月": 10,
    "十一月": 11,
    "十二月": 12,
}

DAY = "联系日期"
DAY_SIN = "日期_sin"
DAY_COS = "日期_cos"
DAY_PERIOD = 31  # 缺少年份，只能按"月内第几天"做周期编码

# ---------------------------------------------------------------- 排除列
# 通话时长是通话结束后才知道的字段，属于结果泄漏。这里显式排除，
# 不从数据集里删列：一是要做归因，二是要做泄漏反证实验。
DURATION = "通话时长"
EXCLUDE_COLS = (DURATION, PDAYS)
EXCLUDE_REASON = {
    DURATION: "结果泄漏：通话结束后才有值，线上打分时拿不到",
    PDAYS: "已拆为 是否被联系过 / _log / 档 三列；原列 -1 会被误读为最小天数",
}

# ---------------------------------------------------------------- 列分组
# 只放与时间无关的数值列；时间类数值统一归 TIME_COLS，避免两边重复计入。
NUMERIC = (
    "年龄",
    "账户余额",
    "本次联系次数",
    "历史联系次数",
    PDAYS_LOG,
    INFO_COMPLETENESS,
)
CATEGORICAL = (
    "职业",
    "婚姻状态",
    EDU,
    "是否信贷违约",
    "住房贷款",
    "个人贷款",
    "联系方式",
    "上次营销结果",
    EDU_UNKNOWN_FLAG,
    CONTACTED,
)
ORDINAL = (PDAYS_BUCKET,)
# 时间列单列一组：B 版（可上线）要整组剔除，因为月份学到的是时期而非客户特征。
TIME_COLS = (MONTH_ORDINAL, MONTH_SIN, MONTH_COS, DAY, DAY_SIN, DAY_COS)
# B 版剔除项：月份/季节无法外推到未来（实测三月 52.0%、五月 6.7%，
# 差异来自不同时期的宏观环境，不是客户属性）。联系日期与日期周期编码保留，
# 因为"今天几号"在打分时是已知的。
TIME_LEAKY = (MONTH_ORDINAL, MONTH_SIN, MONTH_COS)
DAY_KEEP = tuple(c for c in TIME_COLS if c not in TIME_LEAKY)


def _as_series(values: pd.Series | np.ndarray) -> pd.Series:
    """统一成 Series，纯函数才好单测（传入 list / ndarray 也不会炸）。"""
    return values if isinstance(values, pd.Series) else pd.Series(values)


def pdays_contacted(days: pd.Series | np.ndarray) -> pd.Series:
    """是否被联系过：-1 → 0，其余 → 1。"""
    return (_as_series(days) != PDAYS_SENTINEL).astype("int8")


def pdays_log(days: pd.Series | np.ndarray) -> pd.Series:
    """对数压缩；未联系过记 0.0。

    取 0 不会与真实值相撞：天数 ≥1 时 log1p ≥ 0.693，所以 0 是"未联系"的唯一标记，
    配合 是否被联系过 可以完整还原原始信息。
    """
    s = _as_series(days).astype("int64")
    filled = s.where(s != PDAYS_SENTINEL, 0)
    out = np.log1p(filled.to_numpy(dtype="float64"))
    return pd.Series(out, index=s.index, name=s.name)


def pdays_bucket(days: pd.Series | np.ndarray) -> pd.Series:
    """按业务阈值分 6 档：0=未联系，1~5 对应 (0,30] ... 365+。"""
    s = _as_series(days).astype("int64")
    idx = np.searchsorted(np.asarray(PDAYS_BOUNDS), s.to_numpy(), side="left") + 1
    out = np.where(s.to_numpy() == PDAYS_SENTINEL, 0, idx)
    return pd.Series(out.astype("int8"), index=s.index)


def edu_ordinal(edu: pd.Series) -> pd.Series:
    """三档真实学历编 1/2/3，未知留空（由哑变量单独表达）。"""
    return edu.map(EDU_ORDER).astype("float64")


def edu_unknown(edu: pd.Series) -> pd.Series:
    return (edu == UNKNOWN).astype("int8")


def month_to_num(month: pd.Series) -> pd.Series:
    """中文月份 → 1~12 有序整数；未知月份直接报错，不静默变 NaN。"""
    mapped = month.map(MONTH_ORDER)
    bad = mapped.isna()
    if bad.any():
        raise ValueError(f"出现无法识别的月份：{sorted(month[bad].unique().tolist())}")
    return mapped.astype("int8")


def cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """周期编码：让十二月与一月在数值上相邻，避免把周期性当线性。"""
    rad = 2 * np.pi * values.to_numpy(dtype="float64") / period
    return (
        pd.Series(np.sin(rad), index=values.index),
        pd.Series(np.cos(rad), index=values.index),
    )


def unknown_flags(df: pd.DataFrame) -> pd.Series:
    """几列填了「未知」的个数（0~3）。实测与认购率单调：14.6% → 5.5% → 5.0% → 1.9%。"""
    return (df[list(UNKNOWN_COLS)] == UNKNOWN).sum(axis=1).astype("int8")


def info_completeness(df: pd.DataFrame) -> pd.Series:
    """信息完整度 = 3 - 未知个数，取高者更可靠，方向与直觉一致。"""
    return (len(UNKNOWN_COLS) - unknown_flags(df)).astype("int8")
