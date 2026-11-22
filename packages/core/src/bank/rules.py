"""注册表：清洗规则按"名字"登记，跑的时候按名字排好队。

为什么值得单独一层：规则一旦散落在 `run_checks` 的长函数里，
加一条要改三处、删一条容易漏、台账顺序还会随重构漂移。
登记之后，"有哪些规则、哪条致命、命中算不算事故"是一张可查的表，
测试和文档都能直接读它。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Rule:
    code: str
    title: str
    fatal: bool
    mask: Mask
    note: str


REGISTRY: dict[str, Rule] = {}


def rule(code: str, title: str, *, fatal: bool, note: str) -> Callable[[Mask], Mask]:
    """把一条规则登记进 `REGISTRY`；重复登记直接报错，避免"后一条盖掉前一条"。"""

    def decorator(mask: Mask) -> Mask:
        if code in REGISTRY:
            raise ValueError(f"规则 {code} 重复注册")
        REGISTRY[code] = Rule(code=code, title=title, fatal=fatal, mask=mask, note=note)
        return mask

    return decorator


def all_rules() -> tuple[Rule, ...]:
    """按编码排序，台账顺序稳定，两次运行可以直接 diff。"""
    return tuple(sorted(REGISTRY.values(), key=lambda r: r.code))
