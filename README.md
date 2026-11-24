# 银行电话营销：外呼名单排序

课程分组项目。**在打电话之前**，用那时能拿到的客户信息判断他会不会认购定期存款，再按每天的外呼名额排出今天该打给谁。模型分数不占分：架构、取舍、能跑通、数字有出处才是分。

## 现在能跑什么

```powershell
# 1) 环境（首次）
.venv\Scripts\uv.exe sync --all-groups

# 2) 口径层：从原始库构建数据集并自检
.venv\Scripts\python.exe -m bank.build

# 3) 画像：M1 文档里所有数字的唯一出处
.venv\Scripts\python.exe -m bank.profile

# 4) 冻结：按时序切分、算三级指纹、写 manifest（--check 只校验不重建）
.venv\Scripts\python.exe -m bank.freeze

# 4b) 对账点 1：从原始库重算一遍，与冻结账单逐项比（不一致退出码 1）
.venv\Scripts\python.exe -m bank.reconcile

# 5) 训练与评估：两条基线 + 主模型，指标与产物全部记进 mlflow
.venv\Scripts\python.exe -m bank_training.train

# 6) 对照实验 E1–E5（泄漏列、A/B 特征集、未知填众数、pdays 原列、随机切分）
.venv\Scripts\python.exe -m bank_training.experiments

# 7) 导出模型包：models/serving/<版本>/ + current.json（不覆盖旧版本）
.venv\Scripts\python.exe -m bank_training.export --version 1.0.0

# 8) 起服务：另一窗口执行，/docs 可打开，127.0.0.1:8000
.venv\Scripts\python.exe -m bank_api

# 9) 漂移监控：拿「线上」数据对比训练集，逐特征算 PSI
.venv\Scripts\python.exe -m bank_training.drift

# 10) 测试与质量门（交付前最后一关；pytest 240 项、覆盖率门槛 85）
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\ci.py

# 11) 基线封条：第一轮结果的文件指纹，改了数字必须复跑
.venv\Scripts\python.exe scripts\freeze_baseline.py --check

# 12) 回放：每个 step-* tag 各建一次干净环境、重建产物、跑质量门（慢）
.venv\Scripts\python.exe scripts\replay_steps.py
```

冻结的基线数字（`python -m bank.freeze` 输出，改数据或口径后必须重跑核对）：

| 事项 | 数字 |
| --- | --- |
| 原始表 | 45,211 行 × 17 列，正例 5,289（11.70%） |
| 建模窗口 | 2008-11-01 ~ 2010-11-17，21,093 行 / 正例 4,109（19.48%） |
| 训练集 | `< 2009-10-01`，17,537 行 / 2,327（13.27%） |
| 验证集 | `≥ 2009-10-01`，3,556 行 / 1,782（50.11%） |
| 每日名额 | 356 人（验证集 3,556 × 10%，向上取整） |
| 清洗 | 7 条规则命中 5 行，全是 R3 记账，无删行 |

评估报全：13 个互相自洽的指标（名额 8 + 排序 2 + 校准 3）、两种名额口径并列、200 次按行重抽的 95% 区间、Markdown 报告 `data/processed/evaluation_report.md` 随主模型 run 记进 mlflow。
校准与切片：报告里另有 10 箱可靠性表（平均分 0.2719 vs 基率 0.5011、ECE 0.2292）与五张切片表（职业 / 教育程度 / 联系方式 / 婚姻状态 / 年龄档）；名额按切片人数比例分配。**只报告不校准**——验证集不许拟合校准器（ADR 0012）。

主模型结果（`python -m bank_training.train` 输出，验证集 3,556 人 / 名额 356 人）：

| 模型 | AP | ROC-AUC | 名额内精确率 | 名额内召回率 | Brier |
| --- | --- | --- | --- | --- | --- |
| 规则基线（权重打分） | 0.6412 | 0.6510 | 0.7640 | 0.1526 | 0.4924 |
| 逻辑回归 + 特征集 A | 0.5959 | 0.6116 | 0.6180 | 0.1235 | 0.3104 |
| **逻辑回归 + 特征集 B（主模型）** | **0.6498** | 0.6438 | 0.7416 | 0.1481 | **0.2852** |
| 随机森林 + 特征集 A（对照） | 0.6229 | 0.6024 | 0.7500 | 0.1498 | 0.2875 |

主模型名额内打中 264 人；规则基线打中 272 人，但概率质量差一倍（Brier 0.49 vs 0.29），取舍见 `docs/adr/0005-模型选择.md`。

模型包（`python -m bank_training.export --version 1.0.0` 输出）：

| 事项 | 数字 |
| --- | --- |
| 版本与契约 | `1.0.0` / `model_bundle.v1` |
| 特征与必填输入 | 21 列 / 15 个原始字段 |
| 等级切点 | 优先 ≥ 0.6387，备选 ≥ 0.5730（验证集分数分位数，不是概率阈值） |
| 校验和 | `bundle.json` 与 `model.joblib` 各一份 sha256，加载时都对 |
| 发布门禁 | 特征 ≥ 10、AP ≥ 0.50、名额内精确率 ≥ 0.60、ROC-AUC > 0.5 |
| 回退 | 改 `models/serving/current.json` 指回旧版本，不删目录 |

设计取舍见 `docs/adr/0006-模型包与发布.md`。模型包 `models/` 不进 Git，每台机器用一条命令重新导出。

服务接口（`python -m bank_api` 起服务）：

| 路由 | 用途 | 说明 |
| --- | --- | --- |
| `GET /health` | 存活 | 进程活着就 200，不读模型包 |
| `GET /ready` | 就绪 | 模型包加载并校验通过才 200，否则 503 |
| `GET /v1/model` | 模型信息 | 版本、指标、15 个输入字段（类型/范围/枚举），前端据此生成表单 |
| `POST /v1/predict` | 单条预测 | 返回分数、等级、主要因素、用途提示；越界/枚举/缺字段都是 422 且 `loc` 指出字段 |
| `POST /v1/predict/batch` | 批量预测 | 上限 100 条（`configs/api.yaml`），整批共用一个请求号 |

每条预测追加一行到 `data/processed/predictions.jsonl`（`ts/request_id/model_version/score/level/control`，不写客户特征值）。契约与错误码见 `docs/adr/0007-服务契约与错误码.md`。

漂移监控（`python -m bank_training.drift`，基准=训练集 17,537 行，对照=验证集 3,556 行扮演「线上」批次）：

| 特征 | PSI | 判级 |
| --- | --- | --- |
| 联系月份 | 3.4028 | 严重 |
| 住房贷款 | 0.8087 | 严重 |
| 上次营销结果 | 0.6573 | 严重 |
| 职业 | 0.4583 | 严重 |
| 历史联系次数 | 0.3500 | 严重 |
| 年龄 | 0.2797 | 严重 |
| 联系方式 / 联系日期 / 账户余额 / 距上次联系天数 | 0.2318 / 0.2003 / 0.1944 / 0.1604 | 关注 |
| 个人贷款 / 教育程度 / 是否信贷违约 / 本次联系次数 / 婚姻状态 | 0.0881 / 0.0530 / 0.0282 / 0.0224 / 0.0084 | 稳定 |
| 分数 | 1.0851 | 严重 |

判级阈值：< 0.1 稳定，0.1–0.25 关注，≥ 0.25 严重；分箱边界只取基准集，缺失单独一箱，类别未见值合并「其他」。正例率同期 13.27% → 50.11%（+277.66%），与分数 PSI 同向，说明这是真实的时期漂移，不是数据错误。结果落 `data/processed/drift.json`（契约 `drift_report.v1`）并记一个 `m7-drift` run。回流与重训条件见 `docs/adr/0008-漂移监控与回流.md`。

## 目录

```
packages/core       口径规则、数据管道、特征目录、冻结、模型包、推理路径（谁都不依赖）
packages/training   训练、评估、规则基线、单条解释（只依赖 core）
packages/api        推理服务（只依赖 core）
apps/web            前端（M6）
configs/            base.yaml 口径数字；features.yaml 字段唯一出处（含禁用列）
data/raw|frozen|processed
models/serving/     模型包，每版一个目录，不覆盖
scripts/ci.py       五道门：ruff check / ruff format / mypy / lint-imports / pytest（覆盖率 85）
scripts/            另两个：freeze_baseline.py 基线封条、replay_steps.py 按 tag 回放
baselines/          第一轮结果的文件封条：v1/ 快照 + SHA256SUMS
tests/              unit / contract / architecture / service / exam（考卷）
docs/               problem / spec / data_card / pitfalls / adr / modules / steps / cards / notes
```

## 口径要点（答辩必问）

- **`通话时长` 禁用**：它只有通话结束才知道，单变量 AUC 0.8076，是最经典的泄漏列。
- **`距上次联系天数 = -1` 是哨兵**，不是天数；只派生「是否被联系过 / log / 档」三列入模。
- **「未知」是类别**，不填补、不删除。
- **按时间切分，不随机**：行序就是时间序（2008-05-05 ~ 2010-11-17，561 个日期，单调），随机切分会让模型提前看到未来。
- **正例率剧烈漂移**：2008 上半年 3.69% → 2010 上半年 53.22%。验证集正例率约 50%，模型概率会系统性偏低、Brier 很难看，这是漂移不是 bug。
- **名额制**：按每日外呼名额取分数最高的 356 人，阈值 0.5 只作对照；名额人数 = 验证集行数 × 比例，不是全表行数。

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/problem.md` | 一句话定义、成功标准、非目标 |
| `docs/spec.md` | 口径规格表：17 列逐列判定、派生列、禁用列、切分、名额、对照实验、清洗台账 |
| `docs/data_card.md` | 数据卡：来源许可、时间轴证据、漂移表、泄漏证据、限制 |
| `docs/pitfalls.md` | 9 条真踩过的坑 |
| `docs/model_card.md` | 模型卡：指标、校准与漂移、解释示例、7 条限制 |
| `docs/adr/` | 决策记录（数据集口径、时间切分、指纹与冻结、三包结构、模型选择、模型包与发布、服务契约与错误码、漂移与回流） |
| `docs/steps/` | 每个步骤三行：目标 / 验收 / 不做；文件名对应 `step-*` tag |
| `docs/cards/` | 组件卡九节：指纹规范化、配置加载、特征目录 |
| `baselines/` | 第一轮结果的快照与 `SHA256SUMS` 封条，`--check` 不通过就是有数字被改过 |
| `docs/notes/参考项目经验.md` | 两个参考项目哪些机制值得搬 |
| `notebooks/数据口径说明.md` | 早期口径分析与结论（§2.5 已更正为按时间切分） |

## 依赖

`uv.lock` 锁定全部依赖；Python 3.12。`.venv` 由 uv 管理（没有 pip，加依赖请用 `uv add`）。