"""可复现性的底层工具：指纹、规范化序列化、git 状态。

为什么单独一层：指纹算法一变，"同一份数据"的定义就变了，所有历史数字作废。
所以这里只用最朴素的 sha256 + 规范化文本，不引第三方库、不依赖平台。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """按 1 MiB 分块算文件指纹：大文件也不会一次性读进内存。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default(obj: Any) -> str:
    """把 Path / 日期转成稳定字符串；其它类型直接报错，避免"悄悄变成 str(obj)"。"""
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"canonical_json 不认识 {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """键排序、无空格、不转义中文——同样的对象永远得到同样的字符串。"""
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    )


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_table(df: pd.DataFrame) -> str:
    """表内容指纹：规范化 CSV 的 sha256。

    规则（跨机器一致的关键）：
    - 列顺序 = DataFrame 的列顺序（不排序，调用方负责固定顺序）；
    - 行顺序 = DataFrame 的行顺序（数据集里是 序号，天然确定）；
    - 行尾 ``\\n``，不用平台默认换行；
    - 浮点用 ``%.10g``，避免 repr 差异；
    - 空值写空串（本数据集没有真正的空值，只有「未知」这个取值）。
    """
    text = df.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
        na_rep="",
    )
    return sha256_text(text)


def sha256_records(records: list[dict[str, Any]]) -> str:
    return sha256_json(records)


def git_sha() -> str | None:
    """不在 git 仓库里或没装 git 时返回 None，不抛异常。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())
