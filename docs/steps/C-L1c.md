# C-L1c 追踪协议与两个后端

- **目标**：把「往哪里记训练运行」收进一个协议——`Tracker` 六个方法（开运行、参数、标签、指标、单个指标、产物），两个实现（`memory` 在内存里，给测试用；`mlflow` 写 `mlflow.db`）；`mlflow` 这个库只允许在 `tracking/mlflow_backend.py` 里 import，由架构测试用 AST 守着；后端由 `configs/tracking.yaml` 选。
- **验收**：`python -m pytest tests/unit/test_tracking.py tests/architecture/test_layers.py -q` 全过（追踪层 19 项 + 架构 3 项）；`python -m bank_training.drift` 仍能打出新 run_id；`python -m bank_training.train` 五个模型的指标与改造前逐位相同（规则 0.6412/0.6510/0.7640、主模型 0.6498/0.6438/0.7416）；`scripts/ci.py` 末行「质量门：全绿」。
- **不做**：不做「只警告不拒绝」的宽松模式；不做第三个后端（HTTP 报表服务那种等有真实需求再说）；不把 `tracking` 段并进 `configs/base.yaml`（会让冻结指纹跟着工程设置变，理由见 ADR 0021）。