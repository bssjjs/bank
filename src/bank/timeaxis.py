"""时间轴重建：原始表没有年份，但行序就是时间序。

证据见 `docs/data_card.md`「时间轴」一节：30 个月块从 2008-05 走到 2010-11，
月序只回绕过两次（跨年），块内日号从不回退；重建出的 561 个日期全局单调。
所以「按时间切分」是可行的，不需要退回随机切分。

用法约束：读原始库必须 `SELECT rowid AS 序号, *`，把行号落进数据集；
不能依赖 `SELECT *` 的返回顺序，也不能在中间环节重排。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bank import schema as S

# 首个月份就是原始表的第一行（UCI additional 版从 2008 年 5 月开始）。
# 起始年写在这里而不是猜：换一份数据必须显式指定，否则宁可报错。
START_YEAR = 2008
START_MONTH = 5

SEQUENCE = "序号"
YEAR = "年份"
MONTH_REBUILT = "月份序_重建"
DATE = "重建日期"

_NANOS_PER_DAY = 86_400_000_000_000


def build_time_axis(
    month_num: np.ndarray | pd.Series,
    day: np.ndarray | pd.Series,
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
) -> pd.DataFrame:
    """按行序把「中文月份 + 月内日号」还原成真实日期。

    规则：
    - 月份变化量取 ``(cur - prev) % 12``：12→1 得到 1（跨年），08→10 得到 2（中间月缺失）。
    - 月份只允许「前进」或「12→1 跨年」；其它回退一律报错——否则 6→5 会被算成前进 11 个月，
      日期照样单调，错误被掩盖。
    - 同一月内日号必须不回退，回退说明行序被打乱，直接报错而不是猜。
    - 首行月份必须等于 ``start_month``，不匹配就报错，避免把起始年猜错一年。

    返回三列：``年份``、``月份序_重建``、``重建日期``（datetime64）。
    """
    months = np.asarray(month_num, dtype=np.int64)
    days = np.asarray(day, dtype=np.int64)
    if len(months) != len(days):
        raise ValueError("月份与日期长度不一致")
    if len(months) == 0:
        raise ValueError("空表无法重建时间轴")
    if months[0] != start_month:
        raise ValueError(f"首行月份应为 {start_month}，实际 {months[0]}，起始年月配置不对")

    # 绝对月序号 = 年 * 12 + (月 - 1)，逐行累加，跨年自然进位。
    abs_month = np.empty(len(months), dtype=np.int64)
    abs_month[0] = start_year * 12 + (start_month - 1)
    for i in range(1, len(months)):
        prev, cur = int(months[i - 1]), int(months[i])
        if cur < prev and cur != 1:
            raise ValueError(
                f"第 {i} 行月份从 {prev} 退回 {cur}，只有 12→1 可以跨年；行序可能被打乱"
            )
        delta = (cur - prev) % 12
        if delta == 0:
            if days[i] < days[i - 1]:
                raise ValueError(
                    f"第 {i} 行日号回退（{days[i - 1]} → {days[i]}），行序不再是时间序"
                )
            abs_month[i] = abs_month[i - 1]
        else:
            abs_month[i] = abs_month[i - 1] + delta

    years = abs_month // 12
    month_of_year = abs_month % 12 + 1
    # 用「年-月-日」字符串转日期，避免不同平台的 datetime 构造差异。
    stamps = pd.to_datetime(
        [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, month_of_year, days)],
        format="%Y-%m-%d",
    )
    out = pd.DataFrame(
        {
            YEAR: years.astype("int16"),
            MONTH_REBUILT: month_of_year.astype("int8"),
            DATE: stamps,
        }
    )
    if not out[DATE].is_monotonic_increasing:
        raise ValueError("重建日期不是单调不减的，时间轴假设被推翻")
    return out


def with_time_axis(
    df: pd.DataFrame, start_year: int = START_YEAR, start_month: int = START_MONTH
) -> pd.DataFrame:
    """给带 ``序号`` 的原始表拼上三列时间轴（列顺序不变，新列追加在末尾）。

    这是冻结、画像、训练三条路径共用的唯一入口：时间轴只在这里算一次。
    """
    axis = build_time_axis(
        S.month_to_num(df[S.MONTH]), df[S.DAY].to_numpy(), start_year, start_month
    )
    return pd.concat([df.reset_index(drop=True), axis], axis=1)


def month_key(index_or_series) -> pd.Series:
    """把日期序列压成「年-月」字符串，便于按月分组统计。"""
    return index_or_series.dt.strftime("%Y-%m")
