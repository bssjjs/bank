# 架构设计文档

> 读这一页就能回答三个问题：系统由谁组成、五份契约各长什么样、一次预测和一次发布各自怎么走。
> 所有数字都能用第二节给出的命令跑出来；文档里的数字与命令输出不一致时，以命令为准。

## 1. 需求

### 1.1 业务需求

银行电话营销团队每天有固定外呼名额（本项目按验证集 10% 取 356 人）。**在打电话之前**，用那时能拿到的客户信息判断他会不会认购定期存款，按名额排出今天该打给谁。一句话定义与成功标准见 `docs/problem.md`。

### 1.2 课程要求（任务书九模块）

| 模块 | 要求 |
| --- | --- |
| M1 问题定义与口径 | 一句话定义、口径规格表、禁用列清单、数据卡、朴素版与规范版对照、踩坑 ≥5 条、成功标准与非目标 |
| M2 数据管道 | 单一只读入口、清洗规则 ≥3 条带命中行数、标签边界测试、时间切分与缓冲、内容指纹与冻结、manifest |
| M3 训练与评估 | 两条基线、Pipeline、名额规则、指标带口径、运行记录、单条解释、模型卡 |
| M4 模型包与版本 | 六项契约、统一加载与缺列拒绝、不覆盖旧版本、发布门禁、版本状态与回退 |
| M5 后端服务 | `/health` `/ready`、`/v1/predict`（字段级 422）、`/v1/model`、预测记录、只读模型包 |
| M6 前端 | 按 `/v1/model` 动态生成表单、结果卡、字段级错误、服务状态 |
| M7 监控与回流 | PSI（分箱边界取自训练集）、标签回流与对照组、重训条件 |
| M8 工程底座 | 分包与依赖方向、配置强校验、特征目录、注册表、质量门、测试 ≥10、AGENTS.md |
| M9 设计文档 | 本文档、ADR ≥6、模块卡 |

### 1.3 成功标准（可验收）

- 名额内精确率 ≥ 75%、召回率 ≥ 30%、AP ≥ 两条基线中较好者；ROC-AUC 与 Brier 只报告不设门槛（理由见 `docs/model_card.md`：时期漂移下 Brier 天然偏低）。
- 现状：名额内精确率 74.16%（差 0.84pp，如实记录）、召回 14.81%、AP 0.6498（高于规则基线 0.6412 与 LR-A 0.5959）。
- 演示线八步必须现场走通，见第 6 节。

## 2. 需求 → 机制对照表

| 需求 | 机制 | 代码位置 | 文档 | 验收命令 |
| --- | --- | --- | --- | --- |
| 单一原始入口、只读 | `sqlite3` URI `mode=ro` + `PRAGMA query_only=ON` | `packages/core/src/bank/raw.py` | ADR 0001 | `python -m bank.profile` |
| 清洗规则带命中行数 | 注册表 `REGISTRY` 7 条规则，逐条计 `rows`，致命/记账分开 | `clean.py` + `rules.py` | `docs/spec.md` §7 | `python -m bank.freeze` |
| 时间切分与缓冲 | 月份→日期重建、`<cut` 训练 / `≥cut` 验证、缓冲落月份边界 | `timeaxis.py` + `split.py` | ADR 0002 | `tests/unit/test_freeze.py` |
| 内容指纹与冻结 | 规范化 CSV 的 sha256 + 文件封条 `FROZEN.json` | `repro.py` + `freeze.py` | ADR 0003 | `python -m bank.freeze --check` |
| 评估报全（讲义 C-L2a） | 13 个指标同源自洽、两种名额口径并列、200 次按行重抽的 95% 区间、报告进训练运行 | `metrics.py` / `report.py` | ADR 0011 | `python -m bank_training.train` |
| 校准与切片（讲义 C-L3a） | 10 箱可靠性表 + 五张切片表；名额按切片人数比例分配、少于 100 人不报；校准只报告不拟合 | `calibration.py` / `slices.py` / `report.py` | ADR 0012 | `python -m bank_training.train` |
| 注册表、门禁与审计（讲义 D-L2 / D-L3） | 四状态迁移表、门禁先行、审计先行（`fsync`）、`aliases.production` 回退只改指针、封存集规则就位但不切 | `registry.py` / `export.py` | ADR 0013 | `python -m bank.registry show` |
| 故障演练与调参（讲义 F7 / C-L3b） | 注入已知漂移后监控必须响、没动的列不许误报、自适应分箱被反证；调参只在训练期内层时间窗里做，切点越界直接抛错 | `drill.py` / `tune.py` | ADR 0016 | `python -m bank_training.drill` / `tune` |
| 服务热加载与指标（讲义 E-L2） | `packages/api/src/bank_api/store.py`、`service_metrics.py` | `GET /metrics`、`docs/cards/E-L2-服务.md`、ADR 0015 |
| 预测记录 v2、对照组与回流（讲义 F-L2） | 记录带契约号、输入只留 `features_hash`、对照组按 request_id 哈希确定性抽 10%、标签只认对得上号的、对照组不进重训 | `predictions.py` / `app.py` / `backfill.py` | ADR 0014 | `python -m bank_training.backfill --labels ...` |
| 对账点 1（独立于冻结产物） | 从原始库内存重算 → 与 `manifest.json` 逐项比 22 项，不一致退出 1，不写盘 | `reconcile.py` | ADR 0010 | `python -m bank.reconcile` |
| 两条基线 | 规则加权打分 + 逻辑回归，同一验证集同一名额 | `baselines.py` + `train.py` | `docs/model_card.md` | `python -m bank_training.train` |
| 名额制决策 | `quota_n = ceil(n_valid × frac)`，验证集算名额 | `metrics.py` + `config.py` | ADR 0002 / 0005 | `python -m bank_training.train` |
| 运行记录 | 每次训练一个 mlflow run（params/tags/metrics/artifacts） | `train.py` `_log_run` | `docs/model_card.md` | `python -m bank_training.train` |
| 单条解释 | 线性贡献 = 系数 × 标准化后取值，取 |贡献| 前 5 | `packages/core/src/bank/explain.py` | ADR 0005 | `pytest tests/unit/test_training.py` |
| 模型包六项 | `bundle.json` + `model.joblib` + `current.json` | `bundle.py` | ADR 0006 | `python -m bank_training.export --version 1.0.0` |
| 缺列拒绝 | `Bundle.prepare` 先查 `required_columns` | `bundle.py` `MissingColumnsError` | ADR 0006 | `tests/unit/test_bundle.py` |
| 不覆盖旧版本 | 目标目录已存在直接报错 + staging 原子改名 | `bundle.write_bundle` | ADR 0006 | 同上 |
| 字段级 422 | 请求模型由模型包生成，`loc` 前置 `body` | `api/schemas.py` + `api/app.py` | ADR 0007 | `pytest tests/service` |
| 预测记录 | 每条一行 jsonl，不含客户特征值 | `api/app.py` `_append` | ADR 0007 | `pytest tests/service` |
| 漂移监控 | 等频 10 箱 PSI，边界只取基准，缺失单独一箱 | `bank/drift.py` + `bank_training/drift.py` | ADR 0008 | `python -m bank_training.drift` |
| 依赖方向 | 三包 + `.importlinter` 两条契约 + AST 测试 | `.importlinter` + `tests/architecture` | ADR 0004 | `scripts/ci.py` |
| 配置强校验 | pydantic `extra="forbid"` + 严格 YAML（重复键报错） | `config.py` `api/config.py` `model_config.py` | ADR 0004 | `pytest tests/unit/test_config.py` |
| 特征目录 | `features.yaml` 33 字段（含 8 禁用列），与 `schema.py` 对账 | `catalog.py` | ADR 0004 | `pytest tests/unit/test_catalog.py` |
| 注册表 | 清洗规则按编码登记，台账顺序可 diff | `rules.py` | ADR 0004 | `pytest tests/unit/test_rules.py` |
| 质量门 | 五道门一条命令，末行「质量门：全绿」 | `scripts/ci.py` | 本文件 §7 | `python scripts/ci.py` |
| 告警与定时检查 | 读漂移/回流/服务指标，按阈值分级，渠道可换；连续两轮越线才建议重训 | `bank/alerts.py` + `bank_training/watch.py` | ADR 0018 | `python -m bank_training.watch` |
| 文档与 ADR | 9 份 ADR + 9 张模块卡 + 本文件 | `docs/adr/` `docs/modules/` | — | — |

## 3. 模块图与分层

### 3.1 分层

```
┌──────────────────────── 入口层（人只碰这一层） ────────────────────────┐
│ bank.build  bank.profile  bank.freeze  bank_training.train             │
│ bank_training.experiments  bank_training.export  bank_training.drift   │
│ python -m bank_api（UVicorn 起服务）   scripts/ci.py（质量门）          │
└───────────────┬───────────────────────────────┬────────────────────────┘
                │ 训练期                         │ 服务期
        ┌───────▼────────┐              ┌───────▼────────┐
        │ bank_training  │              │    bank_api    │
        │ 训练/评估/导出  │              │  路由/校验/记录 │
        └───────┬────────┘              └───────┬────────┘
                │ 只依赖 core                    │ 只依赖 core
                └───────────┬────────────────────┘
                    ┌───────▼──────────────────────────┐
                    │  bank（packages/core）            │
                    │  口径规则 schema / 数据管道 dataset│
                    │  时间轴 timeaxis / 清洗 clean      │
                    │  切分 split / 指纹 repro           │
                    │  冻结 freeze / 特征目录 catalog    │
                    │  预处理与解释 features·explain     │
                    │  模型包 bundle / 漂移 drift        │
                    └───────┬──────────────────────────┘
                            │ 只读
        ┌───────────────────▼──────────────────────────┐
        │ data/bank-marketing.db（只读）               │
        │ data/frozen/*.parquet + manifest/FROZEN      │
        │ models/serving/<版本>/（运行期只读）          │
        │ mlflow.db / data/processed/*（产物，不入库）  │
        └──────────────────────────────────────────────┘
```

- `bank` 不 import `bank_training` / `bank_api`；`bank_training` 与 `bank_api` 互不 import。由 `.importlinter` 两条契约（`core-is-a-base`、`training-and-api-are-independent`）和 `tests/architecture/test_layers.py` 的 AST 检查双重守住。
- 服务期不 import 训练代码：预处理与解释路径在 M4 时从 training 搬进 core（`features.py`、`explain.py`），所以 `bank_api` 能只靠 `bank` 完成从原始字段到分数、等级、因素的全过程。
- 包内模块图：`freeze` → `raw` → `clean` → `dataset` → `split` → `repro`；`train` → `data` → `metrics`/`baselines`/`pipeline`；`export` → `bundle`；`app` → `schemas` → `bundle`。

### 3.2 一次请求数据流

```
HTTP JSON（15 个原始字段）
  → schemas.build_input_model 生成的 PredictInput 校验（类型/范围/枚举）
  → Bundle.prepare：dataset.derive_features（补派生列）→ select_features（按训练列序取 21 列）
  → Pipeline.predict_proba（同一份在训练时拟合的预处理器与系数）
  → score → Bundle.level（优先/备选/不建议）
  → explain.explain_one（top 5 因素）
  → predictions.jsonl 追加一行 + 响应
```

## 4. 五份契约

### 4.1 契约① `manifest.v1`（`data/frozen/manifest.json`）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `contract` | `"manifest.v1"` | 契约版本 |
| `builder` | `"bank.freeze/1"` | 谁生成的，换构建器必须换版本 |
| `config.path` / `config.sha256` | str / 64 hex | 口径配置路径与指纹（`configs/base.yaml` 的规范化 JSON） |
| `source.path` / `.table` / `.sha256` / `.bytes` | str / int | 原始库路径、表名、文件指纹、字节数 |
| `raw.rows` / `.cols` / `.positives` / `.pos_rate` | int / float | 原始表事实（45,211 / 17 / 5,289 / 0.117） |
| `raw.sequence_column` / `raw.columns` | str / list | 握手序号列名与原始 17 列清单 |
| `window.rows` / `.positives` / `.pos_rate` | int / float | 建模窗口事实（21,093 / 4,109 / 0.1948） |
| `window.date_min` / `.date_max` / `.start` / `.end` / `.cut` / `.buffer_days` / `.feature_set` | str / int | 实际日期范围、配置窗口、切点、缓冲、特征集 |
| `tables.train` / `tables.valid` | obj | 每张表 `rows/positives/pos_rate/date_min/date_max/path/content_sha256` |
| `capacity.rows` / `.positives` / `.pos_rate` / `.n_features` / `.quota_n` / `.quota_frac` | int / float | 容量与名额事实（21,093 / 4,109 / 0.1948 / 21 / 356 / 0.1） |
| `cleaning[]` | list | 7 条规则逐条 `{rule, rows, fatal, note}` |
| `columns` | list[33] | 数据集列序（前端与训练都按它对齐） |
| `fingerprint_rule` | str | 内容指纹算法说明（规范 CSV：LF 换行、`%.10g`、空串表示缺失） |

### 4.2 契约② 训练运行记录（mlflow run + `data/processed/runs.json`）

| 位置 | 内容 |
| --- | --- |
| params | `model`、`feature_set`、`c`、`max_iter`、`quota_frac`、`quota_n`、`n_features`、`seed` |
| tags | `stage`（train/experiment/bundle/drift）、`data_fingerprint`（训练集内容指纹前 16 位）、`config_fingerprint`、`window`、`cut`、`git_sha`、`version`（导出时） |
| metrics | `ap`、`roc_auc`、`precision_at_quota`、`recall_at_quota`、`lift_at_quota`、`brier`、`score_mean`、`base_rate`、`t05_*`（阈值 0.5 对照） |
| artifacts | `run.json`（本轮完整指标快照）、导出时附 `bundle.json` |
| 本地副本 | `data/processed/runs.json` 记录每轮 run_id；`data/processed/runs/<name>.json` 是同一份事实 |

追踪 URI `sqlite:///mlflow.db`，实验名 `bank-marketing`；`mlflow.db` 不进 Git，演示前现场重跑即能复现。

### 4.3 契约③ `model_bundle.v1`（`models/serving/<版本>/bundle.json`）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `contract` / `version` | `"model_bundle.v1"` / `x.y.z` | 契约与语义版本，不覆盖旧版本 |
| `created_at` | ISO8601 | 生成时间 |
| `model_file.name` / `.sha256` / `.size` | str / 64 hex / int | 模型文件名、校验和、字节数 |
| `feature_set` | str | 训练用特征集（B） |
| `features[]` | list[{name, type, order}] | 21 列及其类型、列序（1 起连续） |
| `inputs[]` | list[{name, type, display, unit, min, max, categories}] | 前端表单的 15 个必填输入（含范围与枚举） |
| `required_columns` | list[str] | 与 `inputs` 同序；服务端缺列即 422/缺列异常 |
| `catalog_fingerprint` | 64 hex | 生成时的特征目录指纹 |
| `decision` | obj | `mode="quota"`、`quota_frac`、`quota_n`、`level_cutoffs{优先,备选}`、`threshold`（仅对照）、`explain_top_k` |
| `metrics` | obj | 验证集指标快照（`n/positives/base_rate/ap/roc_auc/precision_at_quota/recall_at_quota/lift_at_quota/brier/score_mean/t05_*`） |
| `training` | obj | `run_id`、`experiment`、两份数据指纹、配置指纹、`git_sha`、`trained_at`、`rows_train`、`rows_valid`、`n_features` |
| `checksum` | {algorithm, value} | `sha256(canonical_json(除 checksum 外的全部字段))`，加载时校验 |

### 4.4 契约④ OpenAPI（`GET /openapi.json`，`/docs` 可视化）

| 字段 | 来源 | 约束 |
| --- | --- | --- |
| 每字段 `name` | `bundle.inputs[].name` | 15 个必填 |
| `type` | 目录类型 | numeric → `number`，类别 → `string` |
| `minimum` / `maximum` | 目录 `min/max` | 年龄 18–95、距上次联系天数 −1–871、余额 −8,019–102,127 等 |
| `enum` | 目录 `categories` | 运行时 `Literal` 内联，含「未知」「成功」等 |
| `description` | 目录 `display` | 中文说明，前端直接展示 |
| `required` | 全部输入 | 与 `bundle.required_columns` 逐项一致（契约测试守住） |
| 批量 `maxItems` | `configs/api.yaml` 的 `batch_limit` | 100 |
| 错误响应 | 422 | `loc` 形如 `["body","年龄"]`，`msg` 说明越界值 |

### 4.5 契约⑤ 预测记录（`data/processed/predictions.jsonl`）

| 字段 | 含义 |
| --- | --- |
| `ts` | ISO8601 带时区 |
| `request_id` | `bank-<uuid4 前 8 位>`，批量共用一个 |
| `model_version` | 出分的模型包版本 |
| `score` | 0–1 的排序分 |
| `level` | 优先 / 备选 / 不建议 |
| `control` | 是否对照组（`?control=true`，用于日后算增量） |

不写任何客户特征值；只写决策事实，便于回流关联与口径审计。

## 5. 目录树

```
bank/
├── AGENTS.md                 开工前必读：规矩、命令、当前进度
├── README.md                 能跑什么、基线数字、接口表
├── pyproject.toml            uv workspace 根（五个包/工具配置）
├── uv.lock                   依赖锁定
├── .importlinter             两条依赖契约
├── configs/
│   ├── base.yaml             口径数字（窗口、切点、名额比例、特征集、种子）
│   ├── model.yaml            模型超参（改它不动冻结指纹）
│   ├── api.yaml              批量上限、请求号前缀、预测记录路径
│   └── features.yaml         33 个字段的唯一出处（类型/范围/枚举/禁用理由）
├── packages/
│   ├── core/src/bank/        口径 + 管道 + 冻结 + 目录 + 预处理/解释 + 模型包 + 漂移
│   ├── training/src/bank_training/   训练、评估、基线、实验、导出、漂移入口
│   └── api/src/bank_api/     路由、请求模型、预测记录
├── scripts/ci.py             五道门
├── data/
│   ├── bank-marketing.db     原始库（只读，不入 Git）
│   ├── frozen/               train/valid 冻结集 + manifest + FROZEN（入 Git）
│   └── processed/            run 快照、drift.json、预测记录（产物，不入 Git）
├── models/serving/<版本>/    模型包（不入 Git，一条命令重导）
├── tests/{unit,contract,architecture,service}
├── notebooks/                分析存档与口径说明
└── docs/
    ├── problem.md spec.md data_card.md pitfalls.md model_card.md architecture.md
    ├── adr/0001–0010
    ├── modules/m1–m9
    └── notes/参考项目经验.md
```

## 6. 两条流程

### 6.1 一次预测

1. 客户端 `POST /v1/predict`，体为 15 个原始字段（示例见 `README.md`）。
2. `api/schemas.py` 的 `PredictInput`（由 `bundle.inputs` 生成）做类型/范围/枚举校验；失败 → 422，`loc=["body", 字段名]`。
3. `Bundle.prepare`：`dataset.derive_features` 补齐派生列（哨兵处理、分档、信息完整度、月份/日期周期编码），`select_features` 按 `bundle.features` 的顺序取 21 列；缺列 → `MissingColumnsError`。
4. `Pipeline.predict_proba`：预处理器（中位数/众数填补、标准化、有序编码、独热）与系数都是训练时拟合的那一份。
5. `Bundle.level(score)`：≥ 0.6387 → 优先；≥ 0.5730 → 备选；否则不建议。切点是验证集分数分位数，不是概率阈值。
6. `explain.explain_one`：取 |系数 × 标准化后取值| 前 5 个因素，附中文特征名与取值。
7. 追加一行 `predictions.jsonl`，返回 `{request_id, model_version, score, level, factors, advice}`。
8. 平均分（0.272）系统性低于验证集基率（0.501）是漂移所致，见 ADR 0008；`/v1/predict` 不返回校准后的概率。

### 6.2 一次发布

1. `python -m bank.freeze`：只读入口 → 时间轴 → 清洗台账 → 派生列 → 时间切分 → 容量检查 → 两级指纹 → 写 `train/valid/manifest/FROZEN`。第二步起可无脑重跑 `--check`。
2. `python -m bank_training.train`：读冻结集，跑规则基线 + LR-A/B + RF-A 四个 run，指标落 mlflow。
3. 改 `configs/model.yaml`（若调参）：只影响训练，不动冻结指纹。
4. `python -m bank_training.export --version <新版本>`：重训 → 发布门禁（特征 ≥10、AP ≥0.50、名额内精确率 ≥0.60、ROC-AUC > 0.5）→ staging 原子写版本目录 → `bundle.json` 校验和 → `load_bundle` 回读自检 → 成功后 `current.json` 指向新版本。
5. 验证：`GET /ready` 200、`GET /v1/model` 版本与指标正确、`POST /v1/predict` 抽查、`python -m bank_training.drift` 记录基线漂移。
6. 回退：把 `current.json` 改回旧版本号并重启服务；旧版本目录一直在，不删、不改。
7. 门禁失败或自检失败：版本目录不会被留下（staging 清理），`current.json` 不动，服务继续用旧版本。

## 7. 做了什么 / 没做什么

### 7.1 已实现

- M1 口径：`docs/problem.md`、`spec.md`、`data_card.md`、`pitfalls.md`（9 条）、ADR 0001/0002；`timeaxis.py` 与 `profile.py` 让每个数字可复现。
- M2 管道：`raw`（只读）、`clean` + `rules`（7 条规则、命中 5 行、无删行）、`split`（时间切分、缓冲 0 落月份边界）、`repro`（源/配置/内容三级指纹）、`freeze` + `--check`、`manifest.v1`。
- M3 训练评估：规则基线 + LR-A + LR-B（主）+ RF-A、名额制指标、mlflow 4 run + 5 实验、单条解释、`model_card.md`、ADR 0005。
- M4 模型包：`model_bundle.v1`、校验和、staging 原子写、不覆盖、`current.json`、发布门禁、ADR 0006。
- M5 服务：5 条路由、字段级 422、批量上限、预测记录、`/docs`、ADR 0007。
- M7 漂移：PSI（等频 10 箱、边界取基准、缺失单独一箱、未见类别合并）、`drift.json` 契约、`m7-drift` run、回流与三条重训条件、ADR 0008。
- M8 底座：三包 + import 契约 + AST 测试、四份 YAML 强校验、33 字段特征目录、清洗规则注册表、五道门、145 项测试、`AGENTS.md`、ADR 0004。
- M9 文档：本文档、8 份 ADR、9 张模块卡、贡献表。

### 7.2 没做（以及为什么）

| 没做 | 原因 | 若要做 |
| --- | --- | --- |
| M6 前端静态页 | 优先级排在最后，接口已就绪（`/v1/model` 返回 15 字段，可直接生成表单） | `apps/web` 一个 HTML + `fetch` 调 `/v1/model` 与 `/v1/predict`，零新依赖 |
| 自动化重训/调度 | 任务书要求「设计」而非实现；三条触发条件已写进 ADR 0008 | cron + `freeze→train→export` 脚本 + 门禁失败告警 |
| 在线 A/B 与影子流量 | 需要按请求分流与两套模型包，当前 `current.json` 单指针 | `traffic.json` 权重 + 两版本同时加载 |
| 数据库存预测记录 | jsonl 足够支撑回流 join，且不需要额外服务 | 换 SQLite 表 + 索引 `request_id` |
| 监控面板 | 任务书只要求「能跑出 PSI」 | Grafana/静态页读 `drift.json` |
| 客户级分组切分 | 数据里没有客户 ID，同一客户可能多次出现 | 上游补 ID 后改 `split.py` 的组切分 |
| 概率校准（Platt/isotonic） | 校准必须用验证集之外的集合，否则等于用验证集调参 | 留出第三个时间段做校准集 |
| FastAPI 自动 422 文案定制 | 手写校验已统一 `loc/msg`，再定制收益低 | 自定义 `RequestValidationError` 处理器 |

### 7.3 加分项（任务书要求 ≥3）

1. **mlflow 实验追踪**：每个训练/导出/漂移阶段都落 run，含参数、数据指纹、指标与产物，可现场查询并对照文档数字。
2. **可解释性**：单条预测返回 top 5 因素（线性贡献，零额外依赖），前端/接口可直接展示。
3. **PSI 漂移监控**：分箱边界冻结在训练集，缺失单独一箱，输出契约化 JSON 并记 run。
4. **内容指纹与冻结封条**：同一份数据在任意机器上重跑得到同一个 sha256。
5. **质量门与架构测试**：五道门一条命令，依赖方向有人守，改坏立刻红。
6. **清洗规则注册表**：加规则只写一个函数，台账顺序天然可 diff。

### 7.4 关键取舍索引

| 编号 | 决定 |
| --- | --- |
| ADR 0001 | 用本机库 45,211 行（每次联系一条），不重新导入 UCI 41,188 |
| ADR 0002 | 时间切分、窗口 A、缓冲 0 天、暂不切封存集 |
| ADR 0003 | 三级指纹；`FROZEN.json` 是封条，`--check` 比内容不比字节 |
| ADR 0004 | 三包与依赖方向；`bank` 为 base |
| ADR 0005 | 主模型 LR + 特征集 B + C=1.0；规则基线留存；不装 shap |
| ADR 0006 | 模型包六项契约、校验和、staging 原子写、`current.json` 回退 |
| ADR 0007 | 请求模型由模型包生成、`Literal` 内联枚举、422 的 `loc` 口径 |
| ADR 0008 | 漂移只看 15 输入 + score、阈值 0.1/0.25、回流与三条重训条件 |