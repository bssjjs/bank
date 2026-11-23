"""YAML 读取小工具：两个配置文件（`base.yaml`、`features.yaml`）共用。

为什么要自己包一层：PyYAML 默认允许重复键、后一个覆盖前一个。口径文件里出现
重复键，几乎一定是复制粘贴改漏了，静默覆盖会让"改了配置却没生效"变成悬案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlError(RuntimeError):
    """YAML 读不进来、或重复键时抛这个。"""


class StrictLoader(yaml.SafeLoader):
    """遇到重复键直接报错，不做"后者覆盖前者"。"""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise YamlError(f"键重复：{key!r}（同一层里出现了两次）")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_mapping(path: Path, *, what: str = "配置文件") -> dict[str, Any]:
    """读成一个字典；顶层不是映射就报错，别让上层拿到 list 再猜。"""
    if not path.exists():
        raise YamlError(f"找不到{what}：{path}")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    except YamlError:
        raise
    except yaml.YAMLError as exc:
        raise YamlError(f"{what}不是合法 YAML：{exc}") from exc
    except ValueError as exc:
        # 比如 `cut: 2009-13-01`：PyYAML 把日期当隐式类型解析，直接抛 ValueError。
        # 不接住的话上层只能看到「month must be in 1..12」，不知道是哪个文件哪个键。
        raise YamlError(f"{what} {path} 里有不合法的取值：{exc}") from exc
    if not isinstance(raw, dict):
        raise YamlError(f"{what}顶层必须是映射（key: value）")
    return raw
