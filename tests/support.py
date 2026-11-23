"""测试共用的小工具（不是产品代码）。

这里只放"让测试能在干净检出上跑起来"的准备工作，不放断言。
"""

from __future__ import annotations

import subprocess
import sys

from bank import bundle as B
from bank import schema as S


def ensure_served_bundle(version: str = "1.0.0") -> None:
    """缺 `models/serving/<version>/` 就现场导出一次。

    模型包不进 Git（二进制，10 KB 也不该进），所以干净检出、或者回放某个 tag 时，
    这里的模型包是不存在的。质量门不能建立在"我本机跑过 export"这个前提上：
    没有就按 README 里那条命令重新导一次，走的还是同一套导出逻辑。
    """
    if (B.SERVING_DIR / version / "bundle.json").is_file():
        return
    result = subprocess.run(
        [sys.executable, "-m", "bank_training.export", "--version", version],
        cwd=S.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"导出模型包失败：\n{result.stdout}\n{result.stderr}")
