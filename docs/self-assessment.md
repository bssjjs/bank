# 自评表

> 本项目由五个人共同完成，分工见 `docs/contributions.md`。
>
> 按任务书的分项规则，逐条给出「证据在哪 + 怎么验证」。**证据只写仓库里真实存在、且能用一条命令复现的内容**；查无实据的条目标「缺」，不占位、不编。

## 一、基础项（60 分）

| 模块 | 要求 | 证据 | 验证命令 | 自评 |
| --- | --- | --- | --- | --- |
| M1 问题定义 | 五问、四口径、设计文档、非目标 | `docs/design.md`（五问 + 四口径表 + 主指标 AP + 基线 + ≥3 条不做）、`docs/problem.md`（一句话定义 + 成功标准 + 非目标）、`docs/spec.md`（17 列逐列判定、派生列、禁用列、切分、名额、清洗台账） | `uv run python -m bank.profile` | 达标 |
| M2 仓库骨架 | README、git 历史、`src/`、依赖锁 | 多次提交、三包 `src` 布局、`uv.lock`、`.python-version`、`final` tag | `git log --oneline`；`uv run python scripts/ci.py` | 达标 |
| M3 配置与可复现 | 单点配置、严格校验、指纹进产物 | `configs/base.yaml`（唯一口径来源）、`config.py` 强校验、`manifest.json` | `uv run python -m bank.freeze --check` | 达标，调参网格在 `tune.py` 内 |
| M4 数据准备 | 清洗、切分、特征目录、manifest | `packages/core/src/bank/clean.py` + `rules.py`（R1–R7）、`split.py`（时间切分）、`configs/features.yaml`（33 字段 + 8 禁用列） | `uv run python -m bank.build` | 达标 |
| M5 训练与评估 | 基线、指标、报告、结果保存 | `packages/training/src/bank_training/train.py`、`metrics.py`（13 指标）、`report.py`（Markdown + JSON 两种报告）、`baselines/` | `uv run python -m bank_training.train` | 部分：无独立封存测试集（ADR 0013 说明理由） |
| M6 模型解释 | 全局 + 单条 + 基准值 | 全局见 `packages/training/src/bank_training/global_explain.py`（平均 |贡献| 排序 + 置换重要度 + 蜂群图，结论并入评估报告）；单条见 `packages/core/src/bank/explain.py`（`base_value` + 贡献 + `base_probability`） | `uv run python -m bank_training.global_explain` | 达标 |
| M7 模型包与推理 | 元数据、校验和、版本闸门、推理 | `packages/core/src/bank/bundle.py`（`model_bundle.v1` + sha256 + `runtime` 血统闸门 + `explain_with_base`）、`packages/api/src/bank_api/`（`/v1/predict` 返回预测、等级、基准值与因素） | `uv run python -m bank_training.export --version 1.1.0` | 达标（推理入口是 HTTP 与函数，没有单独 `predict` CLI） |
| M8 测试与质量门 | 门全绿、覆盖率、守门、端到端冒烟 | `scripts/ci.py`（五道门）、435 项测试、覆盖率 86、`tests/architecture/test_quality_gate.py`、`tests/e2e/test_smoke_train_predict.py`（合成小数据 prepare → train → predict） | `uv run python scripts/ci.py`；`uv run python -m pytest tests/e2e -q` | 部分：e2e 已补，少量服务/契约测试仍读冻结集 |
| M9 智能体协作 | AGENTS、AI 日志、出错实例 | `AGENTS.md`、`docs/ai-log.md`（5 段关键提示词整理稿 + 3 个可按 commit 复核的真实实例；逐字聊天原文未随仓库保存） | `docs/ai-log.md` | 部分：提示词为整理稿 |

## 二、扩展项

> 扩展项要「测试进质量门 + 自评表写证据 + 能当场演示」三者同时成立才计分。下表逐项核对；凡实现但缺任一条的，据实标出。

| 扩展 | 实现证据 | 测试进质量门 | 当场演示 | 自评 |
| --- | --- | --- | --- | --- |
| 多模型 + 算法注册表 | `bank_training/algorithms/`（`linear.py`/`forest.py`）、`parity.py` | 是（`tests/contract/test_algorithms.py` 14 项） | `uv run python -m bank_training.parity --check` | 具备 |
| FastAPI 多路由 + 字段级 422 | `packages/api/src/bank_api/app.py`（六路由） | 是（`tests/service/test_api.py`、`tests/contract/test_openapi.py`） | `uv run python -m bank_api` 后 `curl` | 具备 |
| 注册表门禁 / 晋升 / 回退 / 审计 | `packages/core/src/bank/registry.py` | 是（`tests/unit/test_registry.py` 10 项） | `uv run python -m bank.registry show` | 具备 |
| PSI 漂移 + 注入演练正反对照 | `bank_training/drift.py`、`drill.py` | 是（`tests/unit/test_drift.py` 11 项、`test_drill.py` 7 项） | `uv run python -m bank_training.drill` | 具备 |
| mypy + import-linter + 覆盖率同门 | `scripts/ci.py` | 是（守门测试 15 项） | `uv run python scripts/ci.py` | 具备 |
| 步骤 tag 回放 | `scripts/replay_steps.py`、21 个 `step-*` tag | 是 | `uv run python scripts/replay_steps.py`（慢） | 具备 |
| ADR | `docs/adr/`（22 份） | 否（文档无测试） | 打开文档 | 部分（无测试门） |
| 前端页面（X10） | `apps/web/`（动态表单、结果卡、字段级错误、服务状态） | 否（页面无自动化测试） | `node apps/web/server.mjs` 后打开 5173 | 部分（无测试门） |

## 三、一致性说明

- 本表所有数字的出处：行数/正例/名额 → `uv run python -m bank.profile`；指标 → `uv run python -m bank_training.train` 与 `reports/evaluation_report.md`；测试数/覆盖率 → `uv run python scripts/ci.py` 末行。
- 若上述任一命令重新运行后数字与本表不一致，以命令输出为准，并回改本表。