# D-L1 模型包带运行时血统

- **目标**：让模型包自己说清「我是谁、我用什么环境训出来的」——`bundle.json` 里记 python / sklearn / pandas / numpy 四个版本（由 `bundle.runtime_now()` 从当前进程取，不由人填）；加载时先查血统再反序列化，scikit-learn 主次版本不同或 Python 主版本不同就拒绝加载，并说明为什么（pickle 不保证跨版本兼容）。
- **验收**：`python -m pytest tests/unit/test_bundle.py -q` 17 项全过（含 5 项运行时闸门）；`python -m bank_training.export --version 1.0.0 --promote` 打出的包里有 `runtime` 四项且 `checksum` 重算一致；把 `bundle.json` 的 `runtime.sklearn` 改成 `0.22.1` 后 `load_bundle` 抛 `BundleError` 且消息里同时出现两个版本；`GET /v1/model` 的 `describe()` 里有 `runtime`；`scripts/ci.py` 末行「质量门：全绿」。
- **不做**：不锁补丁版本（只查主次，补丁级差异允许）；不给没有 `runtime` 的老包做兼容后门（本项目能重导，直接拒绝）；不在包外另存一份 `requirements.txt` 让人自己比对——血统要跟包走，不能靠人对。