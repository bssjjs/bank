# C-L2b 新增算法只加一个文件

- **目标**：把「用哪种算法」从 `pipeline.py` 的 `if/elif` 里挪进注册表，每种算法一个文件；每种算法都过同一套一致性契约（可训练、可预测、两次训练逐位相同、与冻结参照一致）。
- **验收**：`python -m pytest tests/contract/test_algorithms.py -q` 14 项全过；`python -m bank_training.parity --check` 打「算法一致性通过：2 个算法」；`baselines/v1/algorithm_parity.json` 在 `baselines/SHA256SUMS` 里；`scripts/ci.py` 末行「质量门：全绿」。
- **不做**：不做框架无关的第二实现（numpy 重写逻辑回归）——理由写进 ADR 0019「参照独立于框架的边界」；不加第三种算法（LightGBM 那些下一轮再说）；不动 `configs/model.yaml` 里的超参数值。