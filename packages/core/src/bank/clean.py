"""清洗规则：能报错就报错，能记账就记账，**不许悄悄删行**。

每条规则都返回命中行数，写进 manifest；致命规则命中即中止，非致命规则只登记台账。
规则本身不修改数据——数据的派生由 `dataset.build_dataset` 负责，两者不混在一起。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bank import schema as S


class CleaningError(RuntimeError):
    """有致命规则命中时抛出，带上完整台账，方便一眼看出哪条规则炸了。"""


@dataclass(frozen=True)
class Finding:
    rule: str
    rows: int
    fatal: bool
    note: str

    def line(self) -> str:
        flag = "致命" if self.fatal else "记账"
        return f"  {self.rule:<28} 命中 {self.rows:>5} 行  [{flag}]  {self.note}"


def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def run_checks(df: pd.DataFrame) -> list[Finding]:
    """按固定顺序跑全部规则。顺序固定，台账才可比。"""
    findings: list[Finding] = []

    # R1 标签取值：只有 是/否 两种，其它值必须炸，不能当负例
    bad_label = ~df[S.TARGET_RAW].isin([S.POSITIVE, S.NEGATIVE])
    findings.append(
        Finding(
            "R1_标签取值",
            _count(bad_label),
            True,
            f"{S.TARGET_RAW} 只能是 {S.POSITIVE}/{S.NEGATIVE}",
        )
    )

    # R2 哨兵之外的负数：-1 是"从未联系"，其余负数没有业务含义
    pdays = pd.to_numeric(df[S.PDAYS], errors="coerce")
    bad_pdays = pdays < 0
    bad_pdays &= pdays != S.PDAYS_SENTINEL
    findings.append(Finding("R2_pdays非法负值", _count(bad_pdays), True, "-1 之外不允许负数"))

    # R3 口径冲突：联系过却没有上次结果。业务上可能（上次是别的活动），保留原值、只记账
    conflict = (pdays != S.PDAYS_SENTINEL) & (df[S.POUTCOME] == S.UNKNOWN)
    findings.append(
        Finding(
            "R3_pdays与上次结果冲突",
            _count(conflict),
            False,
            "距上次联系天数≠-1 但 上次营销结果=未知；保留原值，不修改",
        )
    )

    # R4 月份必须能识别，否则时间轴无法重建
    month_ok = df[S.MONTH].map(S.MONTH_ORDER).notna()
    findings.append(Finding("R4_月份无法识别", _count(~month_ok), True, "必须是中文一月~十二月"))

    # R5 日号必须 1~31
    day = pd.to_numeric(df[S.DAY], errors="coerce")
    bad_day = ~day.between(1, 31)
    findings.append(Finding("R5_日号越界", _count(bad_day), True, f"{S.DAY} 必须落在 1~31"))

    # R6 原始 17 列不允许真空值（「未知」是取值，不是空值）
    raw_cols = [c for c in df.columns if c in _RAW_COLUMNS]
    null_rows = df[raw_cols].isna().any(axis=1)
    findings.append(
        Finding("R6_真空值", _count(null_rows), True, "「未知」是取值；真空值说明上游有问题")
    )

    # R7 通话时长只做质量检查（该列禁用，但脏数据要能被发现）
    if S.DURATION in df.columns:
        dur = pd.to_numeric(df[S.DURATION], errors="coerce")
        findings.append(
            Finding(
                "R7_通话时长为负",
                _count(dur < 0),
                True,
                "该列禁用，但仍须保证数据本身合法",
            )
        )

    return findings


def assert_clean(findings: list[Finding]) -> None:
    fatal = [f for f in findings if f.fatal and f.rows > 0]
    if fatal:
        detail = "\n".join(f.line() for f in fatal)
        raise CleaningError(f"清洗发现 {len(fatal)} 条致命问题，中止构建：\n{detail}")


def summarize(findings: list[Finding]) -> list[str]:
    total = sum(f.rows for f in findings)
    fatal = sum(f.rows for f in findings if f.fatal)
    lines = [
        f"清洗 {len(findings)} 条规则，命中 {total} 行（致命 {fatal} 行，记账 {total - fatal} 行）"
    ]
    lines += [f.line() for f in findings if f.rows]
    return lines


# 原始 17 列：用 schema 的常量拼，不从 DataFrame 动态推断（推断会把派生列也算进来）
_RAW_COLUMNS = (
    "年龄",
    "职业",
    "婚姻状态",
    "教育程度",
    "是否信贷违约",
    "账户余额",
    "住房贷款",
    "个人贷款",
    "联系方式",
    "联系日期",
    "联系月份",
    "通话时长",
    "本次联系次数",
    "距上次联系天数",
    "历史联系次数",
    "上次营销结果",
    "是否认购",
)
