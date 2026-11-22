"""原始库的唯一入口：只读打开、带行号读出。

两条硬约束（踩坑清单第 8 条、AGENTS 第三节第 4 条）：
1. **只读**：用 ``mode=ro`` URI 打开，再叠一层 ``PRAGMA query_only``。任何写操作会直接抛
   ``sqlite3.OperationalError: attempt to write a readonly database``，而不是"悄悄改了不知道"。
2. **带行号**：``SELECT rowid AS 序号``。行号是时间序的载体（`timeaxis` 依赖它），
   不能依赖 ``SELECT *`` 的返回顺序。
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from bank import schema as S
from bank import timeaxis as T


def connect(db: Path = S.DB_PATH) -> sqlite3.Connection:
    """只读连接。调用方负责关闭（用 ``contextlib.closing`` 或 with 语句）。"""
    if not db.exists():
        raise FileNotFoundError(f"原始库不存在：{db}")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def read_ordered(table: str = S.RAW_TABLE, db: Path = S.DB_PATH) -> pd.DataFrame:
    """读原始表并带上 ``序号``（1 起、与库中物理顺序一致）。"""
    with closing(connect(db)) as conn:
        return pd.read_sql(f'SELECT rowid AS "{T.SEQUENCE}", * FROM "{table}"', conn)


def read_plain(table: str = S.RAW_TABLE, db: Path = S.DB_PATH) -> pd.DataFrame:
    """不带行号的原始表，只用于核对列与行数。"""
    with closing(connect(db)) as conn:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)


def table_names(db: Path = S.DB_PATH) -> list[str]:
    with closing(connect(db)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]
