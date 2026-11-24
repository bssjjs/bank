"""E-L2 热加载：换模型只换 `current.json` 指针，不重启进程。

做法（`docs/adr/0015-热加载与服务指标.md`）：

1. 每个请求开始前读一次 `<serving>/current.json` 的**文本**，与上次尝试装上的比；
   一样就什么都不做（一次小文件读，不加载模型）。
2. 不一样就整包重新加载：先 `B.load_bundle`，再 `build_input_model`，两样都成功才整体替换；
   任何一步失败，**旧模型继续服务**，只把错误记进 `last_reload_error` 与失败计数，
   并把这次指针文本记成"已尝试"，避免每次请求都重试同一个坏包。
3. 指针文件缺失或短暂损坏时同理：不把服务打死；等服务端修好、指针文本变了再试。

为什么比"每次请求都读一遍模型包"强：模型包 10 MB 级，每请求反序列化撑不住；
为什么比"定时轮询线程"简单：没有后台线程要管，也没有"多久轮询一次"的口径要调。
"""

from __future__ import annotations

import threading
from pathlib import Path

from bank import bundle as B
from pydantic import BaseModel

from bank_api.schemas import build_input_model

__all__ = ["BundleStore"]


class BundleStore:
    """持有一份"当前正在服务的"模型包，并支持按指针热替换。

    `root` 默认是 `models/serving`（按 `current.json` 找版本）；传了具体版本目录
    或 `fixed=True` 时不做热加载——测试与离线导出用得上。
    """

    def __init__(self, root: Path | None = None, *, fixed: bool = False) -> None:
        self.root = Path(root) if root is not None else B.SERVING_DIR
        self.fixed = fixed
        self._lock = threading.Lock()
        self._bundle: B.Bundle | None = None
        self._input_model: type[BaseModel] | None = None
        self._pointer: str | None = None
        self.loads = 0
        self.reload_failures = 0
        self.last_reload_error: str | None = None

    # ---- 读指针与安装 -------------------------------------------------

    def _pointer_text(self) -> str | None:
        path = self.root / B.CURRENT_FILE
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:  # 权限/占用之类：当作一次失败，交给调用方记账
            self.last_reload_error = f"{type(error).__name__}: {error}"
            return None

    def _install(self, pointer: str | None) -> bool:
        """装一次；无论成败都记住这次试的是哪个指针。"""
        try:
            bundle = B.load_bundle(self.root)
            model = build_input_model(bundle)
        except Exception as error:  # 坏包/半截包：保留旧版本，记一条失败
            self.reload_failures += 1
            self.last_reload_error = f"{type(error).__name__}: {error}"
            self._pointer = pointer
            return False
        self._bundle = bundle
        self._input_model = model
        self._pointer = pointer
        self.loads += 1
        self.last_reload_error = None
        return True

    # ---- 对外 ---------------------------------------------------------

    def load(self) -> bool:
        """启动时强制装一次；包还没发时也调用（之后靠 refresh 自愈）。"""
        with self._lock:
            return self._install(self._pointer_text())

    def refresh(self) -> bool:
        """指针变了就重新加载；返回这次是否真的换了模型。"""
        if self.fixed:
            return False
        pointer = self._pointer_text()
        with self._lock:
            if pointer is None or pointer == self._pointer:
                return False
            return self._install(pointer)

    def current(self) -> tuple[B.Bundle, type[BaseModel]] | None:
        with self._lock:
            if self._bundle is None or self._input_model is None:
                return None
            return self._bundle, self._input_model

    def stats(self) -> dict[str, int | str | bool | None]:
        with self._lock:
            return {
                "loads": self.loads,
                "reload_failures": self.reload_failures,
                "last_reload_error": self.last_reload_error,
                "hot_reload": not self.fixed,
            }
