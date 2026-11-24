# C-L1b 特征工程、模型规格与可解释的模型

## 目标

- 派生特征与预处理只有一个位置：`packages/core/src/bank/dataset.py::derive_features` 产派生列（缺失原样保留），
  `packages/core/src/bank/features.py::build_preprocessor` 做填补/标准化/独热——两者只在 Pipeline 里被调用。
- 模型一块一个文件：`packages/training/src/bank_training/algorithms/`（`@algorithm("lr")`、`@algorithm("rf")`），
  工厂 `pipeline.build_model` 只做分发；加算法不改流水线与解释。
- 解释与模型对得上：`bank/explain.py::contributions` 给出每一列贡献，**Σ贡献 + 截距 == logit**；
  `explain_one` 只是「排序 + 取前 k + 四舍五入」。
- 主模型必须可解释：`bank_training/export.py::explainable` 只放行线性模型，树模型当主模型直接拒绝上线。
- 超参数只在 `configs/model.yaml`，由 `ModelConfig`（`extra="forbid"`）强校验。

## 验收

- `pytest tests/unit/test_explain.py`（9 项）：贡献之和加截距等于 `decision_function`（`places=10`）、
  `sigmoid(decision)` 等于 `predict_proba`、贡献覆盖全部变换后列、按 |贡献| 降序、`explain_one` 是四舍五入后的前 5、
  树模型抛「解释只支持线性模型」、`EXPLAIN_METHOD == "linear_coef_x_scaled"`、导出门禁放行 `lr` 拒绝 `rf`、
  `main --version 9.9.9` 在 `rf` 配置下返回 1 且不写目录。
- `python -m bank_training.export --version 1.1.0` 正常导出；把 `configs/model.yaml` 的 `kind` 改成 `rf` 后
  同一条命令以「主模型 'rf' 无法逐列解释」失败（改完记得改回去）。
- `scripts/ci.py` 全绿。

## 不做

- 不给树模型做解释（No SHAP in serving，见 ADR 0022 的下限说明）；不为解释另建一套模型。
- 不改 `bank/features.py` 与 `bank/dataset.py::derive_features` 的路径与公式——旧模型包的 pickle 记着它们。
- 不把超参数写成候选网格（讲义用 `model.candidates.<名字>`；本项目只有一个主模型，参数写在 `model:` 一段，
  差异记在 `docs/adr/0005-模型选择.md` 与本文档）。