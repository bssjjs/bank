"""算法注册表：一种算法一个文件，登记名就是配置里 `model.kind` 的取值。

为什么要有它：`configs/model.yaml` 的 `kind` 能填什么，不该散落在 `if/elif` 里。新增一种
算法只做两件事——在 `algorithms/` 下写一个文件、用 `@algorithm("名字")` 登记；登记文件由
这个包在第一次用到时自动导入，`pipeline.py` 不用改，`tests/contract/test_algorithms.py`
会自动把它纳入契约测试（可构建、可训练、可预测、两次训练结果一致、与冻结的参照指标一致）。

登记名字必须是 ASCII 小写字母开头、只含小写字母/数字/下划线：它要进配置、进 mlflow 的 run 名、
进模型包的 `bundle.json`，中文或空格会让这三处都对不上。
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Sequence

from bank import catalog
from sklearn.pipeline import Pipeline

Builder = Callable[..., Pipeline]

ALGORITHMS: dict[str, Builder] = {}

_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789_")
_LOADED = False


def algorithm(name: str) -> Callable[[Builder], Builder]:
    """把构建函数登记成一种算法。

    重复登记直接报错：两个文件抢同一个名字时，静默覆盖会让"配置里写 lr 到底跑的是谁"变成
    取决于 import 顺序的事情。
    """
    if not name or not name[0].isalpha() or not name[0].isascii() or not set(name) <= _ALLOWED:
        raise ValueError(f"算法名 {name!r} 不合规：只能用小写字母开头、字母/数字/下划线")
    if name in ALGORITHMS:
        raise ValueError(f"算法 {name} 重复登记：一个名字只能对应一个文件")

    def register(builder: Builder) -> Builder:
        ALGORITHMS[name] = builder
        return builder

    return register


def _load() -> None:
    """第一次用到时，把本包下的算法文件全部导入（每个文件的 `@algorithm` 完成登记）。

    用 importlib 而不是在文件顶部写 `from . import linear`：`linear.py` 要反过来用
    `pipeline.build_preprocessor`，顶部互相 import 会撞上半初始化的模块。
    """
    global _LOADED
    if _LOADED:
        return
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "parity":
            continue
        importlib.import_module(f"{__name__}.{info.name}")
    _LOADED = True


def available() -> tuple[str, ...]:
    """已登记的算法名，按字典序（配置校验与报错文案都用它，顺序稳定才能逐字比对）。"""
    _load()
    return tuple(sorted(ALGORITHMS))


def build(
    name: str,
    spec: catalog.CatalogSpec,
    feature_names: Sequence[str],
    *,
    c: float = 1.0,
    max_iter: int = 5000,
) -> Pipeline:
    """按名字构建「预处理器 + 模型」的整条 Pipeline。

    这里只做分发，不认识的名字把可用名单一起报出来——配置拼错时，用户看到的是有哪些可填，
    而不是一句 `KeyError`。
    """
    _load()
    if name not in ALGORITHMS:
        raise ValueError(f"未知算法 {name!r}：可用的有 {', '.join(available()) or '（空）'}")
    return ALGORITHMS[name](spec, feature_names, c=c, max_iter=max_iter)


__all__ = ["ALGORITHMS", "algorithm", "available", "build"]
