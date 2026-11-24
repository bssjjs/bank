# docs/steps/：每一步的三行说明

每个文件只写三行：**目标**（这一步要什么）、**验收**（怎么算过）、**不做**（明确不碰什么）。
为什么要写：交付时"做了什么"容易说，"当初为什么只做这些"最容易含糊；三行写死，后面少吵。

- 这些文件是**从 git 历史回写**的：仓库开工时没有 `docs/steps/`，A1/A2 加固时补上，
  内容对照实际提交与当时的测试数，不是事后编的目标。
- 质量门（`scripts/ci.py`）诞生于 `step-M8`。M1、M2 两步早于它，所以没有对应 tag：
  当时只有 `unittest discover`，硬打 tag 会假装"每一步都能过质量门"。**如实记录比整齐重要。**
- 讲义步（C-L2a 评估报全、C-L2b 算法注册表与一致性参照、C-L2c 模型与规则取舍、C-L3a 校准与切片、
  D-L2 注册表与审计、E-L2 热加载与指标、F-L2 预测记录与回流、F6 告警与定时检查、
  F7-C-L3b 演练与调参）都有 tag，回放同样覆盖：这些步的提交里 `scripts/ci.py` 已存在，能独立过门。
- **讲义 B-L2 的四段没有单独 tag，对应关系在这里**：讲义把「从原始库重建」拆成
  B-L2-sources / B-L2-cleaning / B-L2-builder / B-L2-reconcile 四步；本项目这四段是在 M2 一个
  提交里一起落地的（`raw.py` 只读入口、`clean.py` 七条规则台账、`split.py` + `freeze.py` 的
  构建与冻结、`reconcile.py` 对账点 1），当时还没有 `scripts/ci.py`。硬补四个 tag 会把一次提交
  假装成四次独立交付，所以只在这里记映射，不补 tag——**宁可少一个 tag，不造假历史**。
- 回放：`python scripts/replay_steps.py` 会对每个 `step-*` tag 建独立工作目录，装环境、跑质量门。

## 讲义步骤 ⇄ 本仓库落点

| 讲义步 | 讲义要什么（一句话） | 本仓库落点 | tag |
| --- | --- | --- | --- |
| A1 / A2 | 地基守门、基线封条、考卷、组件卡、回放 | `scripts/ci.py`、`baselines/`、`tests/exam/`、`docs/cards/` | `step-A1`、`step-A2` |
| B-L1 | 冻结数据交给流水线，训练只吃冻结集 | `python -m bank.freeze` → `data/frozen/`，训练只读它 | 无（M2） |
| C-L1a | 名额策略、指标出口、决策配置 | `bank/policy.py` + `configs/decision.yaml` + `metric_specs.py` | `step-C-L1a` |
| C-L1b | 派生特征/预处理只在 Pipeline、一文件一算法、解释能加回分数 | `dataset.derive_features` + `features.build_preprocessor` + `algorithms/` + `explain.contributions` | `step-C-L1b` |
| C-L1c | 追踪协议 + memory/mlflow 两个后端，MLflow 只在一个文件里 import | `bank_training/tracking/` + `configs/tracking.yaml` | `step-C-L1c` |
| D-L1 | 模型包带 lineage/runtime，版本不一致拒绝加载 | `bundle.py`（`runtime` + `check_runtime`）+ `export.py` | `step-D-L1` |
| E-L1a | 应用层打分与严格逐条记录、NaN 拒绝 | `api/app.py` + `predictions.py`（v2 记录）+ 数值字段关掉 inf/NaN | 无单独 tag（随 `step-E-L2` / `step-F-L2`） |
| E-L1b | 服务壳与错误码 | `api/app.py` + ADR 0007 | `step-M5` |
| F-L1 | 上线后监控 | `bank/drift.py` + `bank_training/drift.py` + ADR 0008 | `step-M7` |
| UI-L1 | 页面 | 不做（M6 由负责人自己写） | — |
| B-L2 | 从原始库重建 + 对账点 | `raw.py` / `clean.py` / `split.py` + `freeze.py` / `reconcile.py` | 无（M2） |
| C-L2a | 评估报全（13 指标/两种口径/区间/报告） | `metrics.py` + `report.py` | `step-C-L2a` |
| C-L2b | 新算法只加一个文件 + 参照先冻结 | `algorithms/` + `parity.py` + `baselines/v1/algorithm_parity.json` | `step-C-L2b` |
| C-L2c | 模型 vs 规则、名额取舍表 | `compare.py` + ADR 0017 | `step-C-L2c` |
| C-L3a | 校准与切片 | `calibration.py` + `slices.py` + ADR 0012 | `step-C-L3a` |
| D-L2 / D-L3 | 门禁、注册表、别名、审计、封存集规则 | `registry.py` + ADR 0013（封存集只写规则，不切） | `step-D-L2` |
| E-L2 | 不重启换模型 + 服务指标 | `store.py` + `service_metrics.py` + `GET /metrics` | `step-E-L2` |
| F-L2 | 随机对照组 + 记录 v2 + 标签回流 | `predictions.py` + `backfill.py` + ADR 0014 | `step-F-L2` |
| F6 | 告警渠道 + 定时检查 | `alerts.py` + `watch.py` + `configs/watch.yaml` | `step-F6` |
| F7 | 注入已知漂移看监控响不响 | `drill.py` + ADR 0016 | `step-F7-C-L3b` |
| C-L3b | 只在训练期内层窗口调参 | `tune.py` + ADR 0016 | `step-F7-C-L3b` |
| A-L3 | 质量门不靠人记着跑 | `scripts/ci.py`，手工跑；按 AGENTS 第三条不建 `.github/` | 无（M8） |
| E-L3 | 解释讲清「为什么」、表单显示名 | 解释部分见 `explain.py`（`step-C-L1b`）；表单显示名随前端 | 部分 |
| UI-L3 | 前端进质量门、单条评估页、批量页、模型说明页 | 不做（M6 由负责人自己写） | — |

没有 tag 的几行不是遗漏，是当时还没有 `scripts/ci.py`，补 tag 会造假历史；映射写在这里，交付时照这张表说。
