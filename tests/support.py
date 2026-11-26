"""测试共用的小工具（不是产品代码）。

这里只放"让测试能在干净检出上跑起来"的准备工作，不放断言。
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from bank import bundle as B
from bank import config as C
from bank import schema as S

_MONTH_NAME = {number: name for name, number in S.MONTH_ORDER.items()}


def synthetic_raw_frame(rows: int = 200, *, per_day: int = 1, seed: int = 42) -> pd.DataFrame:
    """造一张 17 列都合法的原始表，不读任何真实数据。

    日期从 2008-05-05 起按天递增（``timeaxis`` 要求的首月是五月），同一天可以有多行；
    正例固定约 20%，``pdays`` 与「上次营销结果」保持一致，避免触发 R3 冲突。
    """
    rng = np.random.default_rng(seed)
    days = max(1, (rows + per_day - 1) // per_day)
    dates = np.repeat(
        [date(2008, 5, 5) + timedelta(days=offset) for offset in range(days)], per_day
    )[:rows]

    contacted = rng.random(rows) < 0.30
    pdays = np.where(contacted, rng.integers(1, 200, rows), S.PDAYS_SENTINEL)
    last_result = np.where(
        contacted,
        rng.choice(["成功", "失败"], rows),
        S.UNKNOWN,
    )

    return pd.DataFrame(
        {
            "年龄": rng.integers(20, 70, rows),
            "职业": rng.choice(["蓝领工人", "技术员", "教师", "行政人员", "退休人员"], rows),
            "婚姻状态": rng.choice(["已婚", "未婚", "离婚"], rows),
            "教育程度": rng.choice(["小学", "中学", "高等教育", S.UNKNOWN], rows),
            "是否信贷违约": rng.choice(["否", "是", S.UNKNOWN], rows),
            "账户余额": rng.integers(0, 5000, rows),
            "住房贷款": rng.choice(["是", "否", S.UNKNOWN], rows),
            "个人贷款": rng.choice(["是", "否", S.UNKNOWN], rows),
            "联系方式": rng.choice(["手机", "座机", S.UNKNOWN], rows),
            "联系日期": [d.day for d in dates],
            "联系月份": [_MONTH_NAME[d.month] for d in dates],
            "通话时长": rng.integers(30, 900, rows),
            "本次联系次数": rng.integers(1, 4, rows),
            "距上次联系天数": pdays,
            "历史联系次数": rng.integers(0, 6, rows),
            "上次营销结果": last_result,
            "是否认购": np.where(rng.random(rows) < 0.20, S.POSITIVE, S.NEGATIVE),
        }
    )


def write_synthetic_db(path: Path, rows: int = 20_000, *, per_day: int = 22) -> Path:
    """把合成原始表写成 SQLite，表名与真实库一致；不覆盖已存在的库。"""
    if path.exists():
        raise FileExistsError(f"合成库已存在，换个目录：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = synthetic_raw_frame(rows, per_day=per_day)
    with sqlite3.connect(path) as conn:
        frame.to_sql(S.RAW_TABLE, conn, index=False)
    return path


def synthetic_config(db: Path, frozen_dir: Path) -> C.Config:
    """以真实 ``base.yaml`` 为底，只替换原始库与冻结目录。"""
    base = C.load_config()
    data = base.data.model_copy(update={"raw_db": str(db), "frozen_dir": str(frozen_dir)})
    return base.model_copy(update={"data": data})


def scratch_dir(name: str) -> Path:
    """仓库内的临时目录；Windows 沙箱下 ``tempfile`` 的清理会撞 ACL。"""
    path = S.ROOT / ".tmp" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_served_bundle(version: str = "1.0.0") -> None:
    """缺 `models/serving/<version>/` 就现场导出一次。

    模型包不进 Git（二进制，10 KB 也不该进），所以干净检出、或者回放某个 tag 时，
    这里的模型包是不存在的。质量门不能建立在"我本机跑过 export"这个前提上：
    没有就按 README 里那条命令重新导一次，走的还是同一套导出逻辑。
    """
    if (B.SERVING_DIR / version / "bundle.json").is_file():
        return
    result = subprocess.run(
        [sys.executable, "-m", "bank_training.export", "--version", version, "--promote"],
        cwd=S.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"导出模型包失败：\n{result.stdout}\n{result.stderr}")
