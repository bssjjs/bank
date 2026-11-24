# AGENTS.md：给编程智能体的规矩

> 只写每次开工都要知道、而且**已经存在**的东西。每完成一步，在第六节追加这一步留下的规矩；不提前写后面步骤的规则。

## 一、这是什么项目

- 课程分组项目，任务书九个模块（M1 口径 → M9 文档），最终交「能现场演示的系统 + 设计文档 + 答辩」。**模型分数不占分**：架构、取舍、能跑通、数字有出处才是分。
- 业务问题：银行电话营销。**在打电话之前**，用当时能拿到的客户信息判断他会不会认购定期存款，再按每天的外呼名额排出今天该打给谁。
- 数据：UCI Bank Marketing（`data/bank-marketing.db`，单表 `银行营销数据`，只读，17 列）。45,211 行，正例 5,289（11.70%）。
- 代码现状：**三包（`packages/core|training|api`）+ 口径层 + 冻结层 + 特征目录 + 质量门**。`packages/core/src/bank/` 是唯一代码位置（`core` 谁都不依赖）：`schema.py` 放规则与常量，`dataset.py` 构建数据集并自检，`timeaxis.py` 重建时间轴，`profile.py` 出画像数字（M1 文档的唯一出处），`raw.py` 是唯一只读入口，`clean.py` 是清洗台账，`split.py` 按时间切分，`repro.py` 算指纹，`config.py` 强校验 `configs/base.yaml`，`freeze.py` 产出冻结集与 manifest，`yamlkit.py` 是两份 YAML 共用的严格读取器，`catalog.py` 读 `configs/features.yaml` 并与 `schema.py` 对账。`build.py` / `profile.py` / `freeze.py` 是三个可执行入口。`packages/core/src/bank/` 里另有 M4/M5 共用的推理路径：`features.py` 组预处理器、`explain.py` 单条解释、`bundle.py` 读写模型包（`bundle.json` + `model.joblib` + `current.json`）。`packages/training/src/bank_training/tracking/` 是追踪层：`base.py` 定义 `Tracker` 协议与注册表，`memory.py` / `mlflow_backend.py` 是两后端（**只有 `mlflow_backend.py` 能 import mlflow**，架构测试用 AST 扫着），`configs/tracking.yaml` 选后端。`packages/training/src/bank_training/` 是 M3 的训练层：`data.py` 读冻结集、`pipeline.py` 组预处理器与模型、`metrics.py` 算指标、`baselines.py` 规则基线、`explain.py` 单条解释、`model_config.py` 读 `configs/model.yaml`，`train.py` / `experiments.py` / `export.py` 是三个入口（导出模型包走 `export.py`）。`packages/api/src/bank_api/` 是 M5 的服务层：`config.py` 读 `configs/api.yaml`、`schemas.py` 按模型包生成请求模型与响应模型、`app.py` 建 FastAPI 应用（`create_app`）与 `main()`，`__main__.py` 让 `python -m bank_api` 一条命令起服务。`scripts/ci.py` 是五道门的质量门（门清单由公开函数 `gates()` 给出，`tests/architecture/test_quality_gate.py` 守着门槛不被调低）。`configs/decision.yaml` 是名额策略的配置（策略名 + 次优带宽度，`bank/policy.py` 强校验），`scripts/freeze_baseline.py` 给第一轮结果加文件封条（`baselines/`），`scripts/replay_steps.py` 按 `step-*` tag 回放质量门。服务层还有 `store.py`（按 `current.json` 热加载模型包）与 `service_metrics.py`（`service_metrics.v1` 计数）。`tests/` 有 425 项测试（unit + contract + service + architecture + exam 考卷）。分析结论在 `notebooks/`。
- 前端**还不存在**，别假设它存在，也别在文档里引用。已经存在的是：口径层、冻结层、特征目录（`configs/features.yaml` + `catalog.py`）、质量门（`scripts/ci.py`）、清洗规则注册表（`rules.py`）、训练层（`packages/training`）、模型包（`packages/core/src/bank/bundle.py`）、服务层（`packages/api`）。

### 五个人的角色

| 角色 | 模块 | 主要交付 |
| --- | --- | --- |
| 架构与质量 | M8、M9 | 三包与依赖方向、配置强校验、特征目录、注册表、质量门 `scripts/ci.py`；`architecture.md`、ADR、模块卡 |
| 数据 | M1、M2 | 一句话定义、口径规格表、禁用列、数据卡、踩坑清单；原始入口、清洗台账、标签、时间切分、冻结与指纹、manifest |
| 模型与评估 | M3 | 两条基线、Pipeline、名额规则、指标口径、运行记录、单条解释、模型卡 |
| 发布与服务 | M4、M5、M7 | ModelBundle、FastAPI 契约与 422、预测记录、PSI 漂移 |
| 前端 | M6 | 按 `/v1/model` 动态生成表单、结果卡、字段级错误、服务状态 |

## 二、怎么算完成

现在能跑的命令（二十条，不要自创第二十一条）：

```powershell
.venv\Scripts\python.exe -m bank.build
.venv\Scripts\python.exe -m bank.profile
.venv\Scripts\python.exe -m bank.freeze        # 加 --check 只校验不重建
.venv\Scripts\python.exe -m bank.reconcile     # 从原始库重算，与冻结账单逐项对账
.venv\Scripts\python.exe -m bank_training.train
.venv\Scripts\python.exe -m bank_training.experiments
.venv\Scripts\python.exe -m bank_training.export --version 1.0.0
.venv\Scripts\python.exe -m bank.registry show    # stage/promote/rollback/retire
.venv\Scripts\python.exe -m bank_api          # 起服务，127.0.0.1:8000
.venv\Scripts\python.exe -m bank_training.drift  # 对「线上」批次跑 PSI，默认对比验证集
.venv\Scripts\python.exe -m bank_training.backfill --labels data\processed\labels.csv
.venv\Scripts\python.exe -m bank_training.drill   # 注入已知漂移，看监控响不响
.venv\Scripts\python.exe -m bank_training.tune    # 只在训练期里切内层时间窗调参
.venv\Scripts\python.exe -m bank_training.compare  # 模型 vs 规则，名额 10/15/20% 各抓到多少人
.venv\Scripts\python.exe -m bank_training.watch   # 一次检查：漂移 / 回流 / 服务，有 alert 退出码 1
.venv\Scripts\python.exe -m bank_training.parity  # 算法一致性：--write 重算，--check 与 baselines/v1 比
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\ci.py
.venv\Scripts\python.exe scripts\freeze_baseline.py --check   # 基线封条，改了数字必须复跑
.venv\Scripts\python.exe scripts\replay_steps.py              # 慢：每个 step-* tag 各建一次环境再跑门（现在 21 个）
```

- `bank.build` 必须打印「口径层构建完成」，行数 45,211、正例 5,289，并写出 `data/model_dataset.parquet` 与 `data/feature_spec.json`。
- `bank.profile` 是文档数字的唯一出处：`docs/data_card.md`、`docs/spec.md` 里的行数、正例率、漂移表、口径冲突行数、泄漏 AUC 全部来自它。改数据或改口径后，先跑它，再改文档。
- `bank.freeze` 是 M2 的正式产物：读只读库 + `configs/base.yaml`，写出 `data/frozen/{train.parquet, valid.parquet, manifest.json, FROZEN.json}`。必须打印「冻结完成」以及 21,093/4,109、17,537/2,327、3,556/1,782、名额 356、7 条清洗规则命中 5 行（全是 R3 记账）。跑第二遍 `FROZEN.json` 的 sha256 必须不变。`--check` 只比对指纹，输出「冻结校验通过」。
- `bank.reconcile` 是对账点 1：**不读冻结的 parquet**，直接从原始库重切一遍，再和 `data/frozen/manifest.json` 的账单逐项比（源文件指纹、配置指纹、原始行数/正例/列清单、窗口、两表行数/正例/内容指纹/日期范围、名额、特征数、目录一致性）。必须打印「对账通过：从原始库重建的两表与冻结账单逐项一致」，退出码 0；任何一项不一致就退出 1，且不回写冻结数据。它和 `--check` 不同：`--check` 比「今天重跑 vs 冻结」，`reconcile` 比「账单 vs 重建」。
- 训练、评估、服务只读 `data/frozen/`，**不要再从原始库现算**；`data/processed/` 是临时产物，随时可删。
- `pytest tests` 必须全过（425 项），按目录分：unit 289 / contract 20 / service 23 / architecture 41 / exam 52。
  - unit 289：schema 14、时间轴 14、冻结层 25、特征目录 14、配置 11、规则注册表 5、训练 20（含 13 指标手算、区间、报告）、校准与切片 10、模型包 17（含 5 项运行时版本闸门）、漂移 11、对账 7、版本注册表 10、预测记录 12、标签回流 10、模型 vs 规则 8、告警与定时检查 33、追踪协议与两个后端 19、解释与模型的一致性 9（含主模型可解释门禁 2）、名额策略与阈值 19（分位点标定、A 口径两档、原始 float 往返、配置拼错键/策略名/越界值）、指标出口 7（13 个指标一处登记、方向、位数、报告表由登记表派生）。
  - contract 20：OpenAPI 的必填/范围/枚举/批量上限与模型包对账（6）；算法注册表与一致性参照（14，含注册表自检、两次训练逐位相同、`parity --check`、容差与人数判据）。
  - service 23：探针、字段级 422（含 NaN 不许炸）、批量上限、预测记录、无包时 503（干净检出时 `tests/support.py` 会先导出一个模型包）；另有热加载 4（指针换版本、坏指针保留旧模型、固定包关掉热加载）与服务指标 6（计数、延迟、等级、422 不计预测、不泄露特征值）。
  - unit 里另有故障演练 7（注入确定性与五项检查）与调参 6（内层切分边界、切点越界、选最大 AP）。
  - architecture 41：依赖方向与包布局 10、质量门守门 15、基线封条 9、回放 tag 4、追踪层三条（只有适配文件能 import mlflow、两个后端登记、配置选的后端必须已登记）。
  - exam 52：配置 18、目录 16、指纹 18，是人写的考卷，智能体只读、不许改题目。
- `scripts/ci.py` 是交付前最后一关：五道门（ruff check、ruff format --check、mypy、lint-imports、pytest）必须全绿，末行打印「质量门：全绿」。任何一道红都不算完成。
- 交付时贴命令与输出的最后几行原文；没跑过的命令不写进文档，没跑出来的数字不进报告。

## 三、绝对不要做

1. `通话时长` 永远不进特征清单（`EXCLUDE_COLS` 有断言守着）。要做泄漏对照实验，只能显式加列、单独跑，不得改 `model_features()`。
2. `距上次联系天数` 原列不当数值入模，只用派生三列（`是否被联系过` / `_log` / `_档`）；`-1` 是「从未联系」的哨兵，不是天数。
3. 「未知」不填补、不删除、不当缺失：它是类别（`联系方式=未知` 13,020 行 / 28.80%，认购率全表最低一档）。
4. 不改原始库 `data/bank-marketing.db`；不手改 `model_dataset.parquet` 与 `feature_spec.json`，它们只能由 `bank.build` 生成。
5. 不用验证集/封存集调参、选模型、定名额。报任何指标都要写清：哪个集合、多少人、名额几个人。
6. 不降低既有门槛：不改、不删、不跳过 `tests/` 里的断言；不写 `# type: ignore`、`# noqa` 让检查闭嘴。觉得断言有问题，停下来说明。
7. 不把口径常量复制到 notebook 或临时脚本里；唯一出处是 `packages/core/src/bank/schema.py`，别处只引用。
8. 不凭记忆写数字。报告里每个指标都要有本次命令输出对应——「报告数字与命令输出对不上」是任务书的一票否决项。
9. 不改 `baselines/`（`baselines/SHA256SUMS` 是第一轮结果的文件封条）。`scripts/freeze_baseline.py --check` 必须通过；确实换了数字要重写封条，必须先说明改了哪张表、为什么，得到负责人同意才 `--write`。
10. 不改 `tests/exam/` 的题目。考卷是人写的，你只改被测代码；考卷红了就是代码错了，不许连题目一起改绿。
11. 不手改 `models/serving/registry.json` 与 `audit.jsonl`。状态只能通过 `python -m bank.registry` 改，审计是只追加的事件流；绕过去改等于把「谁在线上、为什么」的证据抹掉。
12. 不创建 `.github/`（本课程手工跑 `scripts/ci.py`，见卷一 A1）。
13. **不照搬参考项目的文件或约定**。参考项目里的东西不等于本项目需要它；新增任何文件（`.gitattributes` 这种纯工程约定除外）都要能说出「本项目为什么需要」，说不出就不加。例子：`CLAUDE.md` 只有用 Claude Code 才有人读，本项目不用，2026-11-24 已删——别因为参考项目有就加回来。

## 四、怎么干活

1. 开工先读：本文件 → `notebooks/数据口径说明.md` → `configs/features.yaml` → `packages/core/src/bank/schema.py` → `tests/unit/test_freeze.py`。
2. 先给计划再动手：要新建/修改哪些文件、每个公开函数的签名、打算怎么验证。**等负责人确认**。
3. 改完跑第二节的命令（至少 `pytest tests` + `scripts/ci.py`），贴输出；不顺带重构无关代码。
4. 同一项检查修三轮还红就停下报告：贴失败输出和试过的办法。要改范围以外的文件，也先停。
5. 交付顺序：改了哪些文件 → `git status --short` 与 `git diff --stat` 原文 → 命令输出的末尾几行。

## 五、仓库约定

### 目录（任务书要求，代码已在 `packages/core`）

```
packages/core|training|api   三包：core 谁都不依赖，training 与 api 互不依赖
apps/web                     前端目录（M6 暂缓，技术栈待定，ADR 待写）
configs/                     base.yaml 放口径数字；features.yaml 是字段唯一出处（含禁用列）；
                             model.yaml 放超参数；api.yaml 放服务设置；watch.yaml 放告警阈值与渠道；
                             tracking.yaml 放追踪后端与实验名；decision.yaml 放名额策略与次优带宽度
data/raw|frozen|processed    raw 只读入口、processed 是产物、frozen 冻结后进 Git
models/serving/              模型包，每版一个目录，不覆盖
scripts/                     ci.py 质量门；freeze_baseline.py 基线封条；replay_steps.py 按 tag 回放
baselines/                   第一轮结果的文件封条：v1/ 快照 + SHA256SUMS
tests/unit|contract|architecture|service|exam
docs/                        problem/spec/data_card/pitfalls/architecture/model_card/demo
docs/adr/                    决策记录，0001 起编号，一份一个决定
docs/modules/                模块卡：目标 / 验收 / 不做
docs/steps/                  步骤卡：每步三行（目标/验收/不做），文件名对应 tag（含讲义步 C-L2a/F-L2 等）
docs/cards/                  组件卡：指纹 / 配置 / 特征目录 / 服务 / 告警，每张九节
docs/demo.md                 演示单：八步命令 + 期望输出 + 失败时查哪里（答辩用）
```

- 代码只在 `packages/core/src/bank`（导入名仍是 `bank`，由 uv workspace 以 editable 装进 `.venv`）。`packages/training` 与 `packages/api` 的导入名分别是 `bank_training`、`bank_api`。**不要在别处再放一份规则或另起一个 `src/`**；迁移的取舍写在 `docs/adr/0004-三包结构与依赖方向.md`。

### Git 边界（`.gitignore` 已按此设置）

- 不进 Git：`data/raw/`、`data/processed/`、`data/*.db`、`data/*.parquet`、`models/`、`mlflow.db`、`mlruns/`、`.venv/`、工具缓存。
- 进 Git：代码、`configs/`、`docs/`、`tests/`、`data/frozen/`（冻结集 + `FROZEN.json`）、`data/feature_spec.json`。
- 依赖锁 `uv.lock` 进 Git。加依赖用 `uv add` / `uv lock`，命令要带 `$env:UV_CACHE_DIR=".venv\.tmp\uv-cache"; $env:UV_LINK_MODE="copy"`（沙箱挡了默认缓存目录，不带会报拒绝访问）。uv 在 `.venv\Scripts\uv.exe`。

### 命名与语言

- 原始中文列名一律保留原样：业务方是银行，中文列名是契约的一部分。代码标识符、文件名、目录名用英文；注释、文档、提交信息用中文。
- 事件名/日志用英文「动词_名词」；日志里不写客户特征值。

### 提交

- 一条提交说清一件已经做完的事，中文；五个人每人都要有提交（任务书要求，答辩看 git 历史）。

## 六、当前进度与已知缺口

- 已完成 M1（口径与问题定义）：`docs/problem.md`（一句话定义、成功标准、非目标）、`docs/spec.md`（17 列逐列判定、派生列、禁用列、切分与名额口径、E1–E5 对照实验、清洗规则台账）、`docs/data_card.md`（来源许可、时间轴证据、漂移表、泄漏证据）、`docs/pitfalls.md`（9 条）、`docs/adr/0001`（数据集与行数口径）、`docs/adr/0002`（时间切分与窗口）、`packages/core/src/bank/timeaxis.py`、`packages/core/src/bank/profile.py`、`tests/unit/test_timeaxis.py`（14 项）。
- 已完成 M2（数据管道与冻结）：`configs/base.yaml`（唯一口径来源）+ `packages/core/src/bank/config.py` 强校验（拼错键/重复键/越界值直接报错）、`raw.py`（`mode=ro` + `query_only` 唯一只读入口）、`clean.py`（R1–R7 清洗台账，不删行）、`split.py`（时间切分与缓冲）、`repro.py`（源/配置/内容三级指纹）、`freeze.py`（`python -m bank.freeze` 产出 `data/frozen/` 四件套 + `--check`）、`docs/adr/0003`（内容指纹与冻结）、`tests/unit/test_freeze.py`（25 项）。单测合计 53 项。
- 已完成 M8 的工程底座（含清洗规则注册表 `rules.py`）：uv workspace 三包 + 依赖方向（`.importlinter` 两条契约 + 架构测试）、`configs/features.yaml` 33 个字段（含 8 个禁用列与禁用理由）+ `catalog.py` 与 `schema.py` 对账、`yamlkit.py` 严格 YAML、`scripts/ci.py` 五道门、`rules.py` 清洗规则注册表（7 条规则按编码排序登记，加规则只写一个函数）、测试目录按 `unit/contract/architecture/service` 分好。
- 已完成 M3（训练与评估）：`packages/training/src/bank_training/` 六个模块 + `train.py` / `experiments.py` 两个入口、`configs/model.yaml`（超参数单独一份，改它不动冻结指纹）、`docs/adr/0005`（模型选择，含 A/B 与 RF 证据）、`docs/model_card.md`（指标、校准、限制）、`docs/spec.md` 第 6 节 E1–E5 回填、`tests/unit/test_training.py`（11 项）。**主模型逻辑回归 + 特征集 B（21 列，去掉月份三列）+ C=1.0**，验证集 AP 0.6498 / 名额内精确 0.7416 / 召回 0.1481 / Brier 0.2852，名额 356 人里打中 264 人；规则基线 AP 0.6412 / 精确 0.7640 / Brier 0.4924；4 个 run 与 5 个实验都落在 `mlflow.db`（`data/processed/runs.json` 存 run_id）。mlflow 追踪 URI 是 `sqlite:///<仓库>/mlflow.db`，实验名 `bank-marketing`。
- 已完成 M4（模型包与版本）：`packages/core/src/bank/bundle.py`（`model_bundle.v1` 契约、`sha256(canonical_json)` 校验和、staging 原子写、不覆盖旧版本、`current.json` 指针）、`packages/core/src/bank/features.py` 与 `explain.py`（训练与服务共用的预处理与解释路径，所以 `bank_api` 不需要 import `bank_training`）、`packages/training/src/bank_training/export.py`（`python -m bank_training.export --version 1.0.0`，发布门禁 AP≥0.50、名额内精确率≥0.60、特征数≥10）、`docs/adr/0006-模型包与发布.md`、`tests/unit/test_bundle.py`（12 项）。当前线上版本 `models/serving/1.1.0`（1.0.0 降为 staging）：特征集 B 21 列、必填输入 15 列、等级切点 **优先≥0.57298 / 备选≥0.253228**（A 口径，见下一条 C-L1a）、验证集 AP 0.6498、名额内精确率 0.7416、mlflow run `7530c4281634419fa243470de1be3305`。模型包不进 Git，演示前要重跑 export。
- 已完成 M5（服务）：`packages/api/src/bank_api/` 四个模块 + `configs/api.yaml`（批量上限 100、请求号前缀 bank、预测记录路径）、`docs/adr/0007-服务契约与错误码.md`。六条路由（`/metrics` 见 E-L2 条目）：`/health` 永远 200、`/ready` 与 `/v1/model` 在包缺失时 503、`/v1/predict`（字段级 422，`loc` 形如 `["body","年龄"]`）、`/v1/predict/batch`（上限 100）。请求模型由模型包 15 个输入列动态生成，类别列用运行时 `Literal` 内联 `enum`；每条预测追加一行 v2 记录到 `data/processed/predictions.jsonl`，不写客户特征值（只留 `features_hash`）。`python -m bank_api` 起服务，`/docs` 可打开。
- 已完成 M7（漂移与回流）：`packages/core/src/bank/drift.py`（PSI：等频 10 箱、边界只取基准集、缺失单独一箱、类别未见值合并「其他」）、`packages/training/src/bank_training/drift.py`（`python -m bank_training.drift`，基准=训练集、对照默认验证集、落 `data/processed/drift.json` 与 mlflow run `m7-drift`）、`docs/adr/0008-漂移监控与回流.md`（阈值 0.1/0.25、只看 15 个原始输入 + score、标签回流与对照组、三条重训条件）、`tests/unit/test_drift.py`（11 项）。实测：联系月份 PSI 3.4028、住房贷款 0.8087、上次营销结果 0.6573、职业 0.4583、历史联系次数 0.3500、年龄 0.2797 为「严重」；score 1.0851；正例率 13.27% → 50.11%（+277.66%）。
- 已完成 C-L2a（评估报全）：`metrics.py` 13 个互相自洽的指标（名额 8 + 排序 2 + 校准 3）、`two_views` 两种名额口径并列、`bootstrap_ci` 200 次按行重抽的 95% 区间、`report.py` 生成 `data/processed/evaluation_report.md` 并作为产物记进主模型 run、`docs/adr/0011-评估报全.md`。主模型区间：精确率 [0.7051, 0.7838]、召回 [0.1403, 0.1561]、AP [0.6286, 0.6777]、ROC-AUC [0.6262, 0.6654]、Brier [0.2750, 0.2938]；ECE 0.2292、log_loss 0.8080。基线封条因新增 8 个指标键而用 `--write` 重写过一次（旧 15 个键逐项核对相等）。
- 已完成 C-L3a（校准与切片）：`calibration.py`（10 箱可靠性表、ECE、最偏一箱）、`slices.py`（职业/教育程度/联系方式/婚姻状态/年龄档五张表，名额按切片人数比例分配、少于 100 人不报）、报告新增「校准」「切片」两节、`docs/adr/0012-校准与切片.md`、`tests/unit/test_calibration.py`（10 项）。**校准只报告不拟合**：验证集不许拟合 Platt/Isotonic（那是拿验证集调参）。实测要点：`联系方式=未知` 253 人、名额 25 只打中 3 人（精确率 0.1200、AP 0.1468）——这一档不能靠模型排序，要单独处理；职业里最差是行政人员 0.6383，最好蓝领工人 0.8800。
- 已完成 A1（地基与守门，§卷一参考）：`.gitattributes`（LF/二进制声明）、`.editorconfig`、三个 `py.typed`、`.importlinter` 契约改名 `core-independent` / `training-api-independent`、`scripts/ci.py` 五道门 + 覆盖率门槛 85（门清单由 `gates()` 公开，`tests/architecture/test_quality_gate.py` 15 项守着不被调低）。161 项测试时全绿、覆盖率 86%。
- 已完成 A2（基线 + 考卷 + 卡片 + 回放）：`baselines/`（9 个快照 + `SHA256SUMS`，含算法一致性参照与 `bundle-1.1.0.json`，`scripts/freeze_baseline.py --write/--check`；快照 JSON 统一 UTF-8+LF 写出，否则 CRLF 会让封条在 clone 后对不上）、`tests/exam/` 三份考卷 52 项（配置 18 / 目录 16 / 指纹 18，人写的，智能体只读）、`docs/cards/` 五张组件卡、`docs/steps/` 23 张步骤卡、`scripts/replay_steps.py`（每个 `step-*` tag 各建一次干净 worktree：装环境 → 重建产物 → 跑门；步骤卡与 tag 守门测试取 tip，旧 tag 的 export 只在该 tag 自己支持 `--promote` 时才加该参数）、`tests/support.py::ensure_served_bundle`（缺模型包时先导出一个，服务与契约测试靠它才能在干净检出里跑）。
- 已完成 D-L2 / D-L3（注册表、门禁与审计）：`packages/core/src/bank/registry.py`（`model_registry.v1` 契约；四状态 `candidate → staging → production → retired` 与迁移表；**门禁先行**——门禁没过一步都不动；**审计先行**——先写 `audit.jsonl` 并 `fsync` 再改状态；`aliases.production` 回退只改指针不删目录；CLI `python -m bank.registry show|stage|promote|rollback|retire`），`export.py` 默认只登记为 `candidate`（要上线加 `--promote` 或单独 `promote`），`docs/adr/0013-注册表门禁与审计.md`，`tests/unit/test_registry.py`（10 项）。演练实测：`export --version 1.1.0`（candidate）→ `promote`（production，1.0.0 降 staging）→ `rollback 1.0.0`（production，1.1.0 降 staging），共 7 条审计事件。**封存集**这轮不切：只有 3,556 行验证集，再切一段会让名额内精确率的 95% 区间宽到没有判断力；规则写在 ADR 0013，代码里不留开箱入口。
- 已完成 F-L2（预测记录 v2、对照组与标签回流）：`packages/core/src/bank/predictions.py`（`prediction_record.v2`：`features_hash` 只留 sha256 不留特征值、`quota_frac/quota_n`、`is_control` 按 `sha256(seed:request_id)` 确定性抽 10%）、`app.py` 每条预测写 v2 记录（显式 `?control=` 可覆盖，批次共用一个标记）、`packages/training/src/bank_training/backfill.py`（`--labels` 两列 CSV；对不上号的 request_id 直接报错退出、重复标签拒绝、同一号多条只取最后一条；落 `labeled.jsonl` 与 `feedback.v1` 汇总，含 `training_allowed` = 有标签且非对照组）、`docs/adr/0014-预测记录v2与标签回流.md`、`tests/unit/test_predictions.py`（12 项）、`tests/unit/test_backfill.py`（10 项）。演练实测 12 条预测 / 10 条标签：覆盖率 83.3%、可重训 10、认购率 0.4000。
- 已完成 E-L2（热加载与服务指标）：`packages/api/src/bank_api/store.py`（`BundleStore`：每请求读 `current.json` 文本，变了才整包重载；坏包/坏指针保留旧模型、记 `reload_failures`，指针修好自动恢复）、`service_metrics.py`（`service_metrics.v1`）与 `GET /metrics`（请求数、`by_route`/`by_status`、预测数、`by_level`、平均/最大延迟、`model`、`reload`）、`docs/adr/0015-热加载与服务指标.md`、`docs/cards/E-L2-服务.md`、`tests/service/test_hot_reload.py`（4 项）、`tests/service/test_metrics.py`（6 项）。实测：服务进程不重启，`/ready` 从 1.0.0 → 1.2.0 → 1.0.0，`reload.loads` 1→2→3；`/metrics` 显示 `predictions=1`、`by_level={'不建议':1}`、`latency_ms_max=65.87`。路由从 5 条变 6 条（多 `/metrics`），契约测试同步更新。
- 已完成 F7 / F6 / C-L3b（演练与调参）：`packages/training/src/bank_training/drill.py`（`python -m bank_training.drill`：在验证集副本上注入「账户余额 ×1.5+5000」「职业一半改未知」「六成负例翻正例」，断言五项——数值 PSI 12.4432、类别 3.1173、标签 0.5011→0.8003（+59.7%）、对照婚姻状态 0.0084 不响、反证「两边各自分位数」0.0233 看不见漂移；固定种子 20261122）、`tune.py`（`python -m bank_training.tune`：训练期 `<2009-10-01` 再切拟合段 `<2009-04-01` 7,341 行与内层评估段 10,196 行，网格 C∈{0.3,1,3,10} 按内层 AP 选中 C=10（0.2741），但最高最低只差 0.0094（< 噪声线 0.01）→ 结论是不改 `configs/model.yaml`）、`docs/adr/0016-故障演练与调参.md`、`tests/unit/test_drill.py`（7 项）、`tests/unit/test_tune.py`（6 项）。
- 已完成 D-L1 的运行时部分（模型包血统与版本闸门）：`bundle.json` 新增 `runtime`（python / sklearn / pandas / numpy，由 `bundle.runtime_now()` 从当前进程取，不由人填），`load_bundle()` 在反序列化**之前**调 `check_runtime()`：sklearn 主次版本必须一致、Python 主版本必须一致，不一致抛 `BundleError` 并把两个版本都写进消息；`/v1/model` 的 `describe()` 一并返回 `runtime`。没有 `runtime` 的老包直接拒绝加载（本项目能重导，不做兼容后门）。`docs/adr/0020-模型包的运行时血统.md`；`tests/unit/test_bundle.py` 增 5 项（12→17）。本机 `models/serving/1.0.0` 已重导，run_id `4e184fb8ab4a41b4bec61ce1bccfb828`。
- 已完成 C-L2b（算法注册表与一致性参照）：`packages/training/src/bank_training/algorithms/` 下一种算法一个文件（`linear.py`→`lr`、`forest.py`→`rf`，用 `@algorithm("名字")` 登记，包在第一次用到时用 importlib 自动导入），`pipeline.build_model` 只做分发，`configs/model.yaml` 的 `kind` 改成按注册表校验（加算法不用改枚举），`python -m bank_training.parity --write/--check` 在冻结集上把每个算法跑一遍，冻结四项指标 + 逐条分数指纹，`baselines/v1/algorithm_parity.json` 由基线封条管；`docs/adr/0019-算法注册表与一致性参照.md`；`tests/contract/test_algorithms.py`（14 项）。`rf` 在特征集 B 上 AP 0.6455 / 名额内精确率 0.7640（与 A 的 0.6229 不是同一组）。
- 已完成 C-L2c（模型 vs 规则的名额取舍）：`packages/training/src/bank_training/compare.py`（`python -m bank_training.compare`：同一验证集上把逻辑回归与规则基线按 10%/15%/20% 三个名额各算一遍，输出 `data/processed/compare.md` 与 `.json` 并记一个 `c-l2c-compare` run）。实测——10%：模型 264 人 / 精确 0.7416，规则 272 人 / 0.7640；15%：403 / 0.7561 对 407 / 0.7636；20%：537 / 0.7553 对 550 / 0.7736。结论写进 ADR 0017：**模型没有比规则多抓人**（差 8/4/13 人，均在一个标准误内），但名额从 10% 放到 20% 精确率几乎不动（0.7553）而召回翻倍（0.1481→0.3013）；主模型仍是逻辑回归，理由是概率可校准（Brier 0.2852 对 0.4924）与单条可解释。`tests/unit/test_compare.py`（8 项）。
- 已完成 F6（告警渠道与定时检查）：`packages/core/src/bank/alerts.py`（告警级别、console/file/webhook 三种渠道，渠道失败降级成 warn 而不是崩）、`packages/training/src/bank_training/watch.py`（`python -m bank_training.watch`：读 `drift.json` 分级、读 `feedback.json` 看覆盖率、可选读 `/metrics`；重训建议要求同一特征 PSI≥0.25 **连续两次**，历史落 `data/processed/watch_history.jsonl`；有 alert 退出码 1）、`configs/watch.yaml`（关注线 0.1 / 告警线 0.25 / 认购率 50% / 覆盖率 0.7，强校验）、`docs/adr/0018-告警渠道与定时检查.md`、`tests/unit/test_watch.py`（17 项）。实测：第一轮 info 0 / warn 5 / alert 7（6 个特征越线 + 认购率 +277.7%），第二轮多出 `retrain/psi_two_runs`，退出码 1。
- 已完成 C-L1a（名额策略）：`packages/core/src/bank/policy.py` 把「名额比例」标定成两个**真实存在的分数**（验证集分位点，`method="higher"`，不插值），`configs/decision.yaml` 写策略名与次优带宽度并强校验（拼错键、策略名不认识、次优带宽度 ≤ 名额比例都当场报）。**口径选 A**（负责人 2026-11-24 拍板）：`优先` = 名额内全部（356 人，切点 0.57298），`备选` = 名额外的次优带（到 40% 分位，切点 0.253228），其余不建议。阈值**存原始 float 不四舍五入**——6 位小数会把边界上的人挤出去。`export.py` 用它冻结 `decision.level_cutoffs`，并加一道门禁：按阈值选中的人数与名额对不上就不许发布。模型包因此重导为 `1.1.0` 并重封基线（旧 `bundle-1.0.0.json` 快照删除，换成 `bundle-1.1.0.json`）。`docs/adr/0023-名额策略与阈值标定.md`、`docs/steps/C-L1a.md`、`tests/unit/test_policy.py`（19 项）。指标出口也是一处登记：`metric_specs.py` 写 13 个指标的中文名、方向（越高越好/越低越好）与小数位，报告表由它派生，`metrics.py` 导入时校验清单一致（`tests/unit/test_metric_specs.py` 7 项）。
- 全量复核：`pytest tests` **425 项全过**、覆盖率 86%、`scripts/ci.py` 末行「质量门：全绿」；`scripts/replay_steps.py` 实测 **全量 21 个 `step-*` tag 一次跑通**（2026-11-24，`python -u scripts\replay_steps.py *> .tmp\replay-all.log`，末行「全部通过（21 个 tag）」、退出码 0、日志里 0 处「未通过」）。此前也分过四个批次单跑，结论一致。回放要先把 `data/bank-marketing.db` 拷进工作目录（它不进 Git），这一步由脚本自己做。讲义各步与本仓库落点（含没有 tag 的 B-L1/B-L2/E-L1a/A-L3）的完整映射表写在 `docs/steps/README.md`。
- **未开始：M6 前端**，这是负责人的决定——他自己写，不要替他实现。设计卡在 `docs/modules/m6-前端页面.md`（目标/验收/不做），要实现时按任务书 M6：字段从 `GET /v1/model` 动态生成、结果卡、字段级错误、服务状态提示、只走 HTTP；接口已经就绪（`/v1/model` 返回 15 个输入字段，`/v1/predict` 30 秒内可接上）。
- **已关闭的口径问题：`联系方式` 不是隐私列**（负责人 2026-11-23 决定）。它存的是联系渠道类型（手机 / 固话 / 未知，13,020 行是「未知」），不是电话号码，不构成个人信息。因此它继续留在 21 个特征与 15 个必填输入里，分析与前端都照常使用。之前记过一条「待移除」的待办，已随这次决定作废——不要因为它听起来像隐私就去掉它，那样会让指标无故下降，还得改整条冻结/训练/导出/文档链。
- 已定的口径（ADR 0002 为准，别再问）：窗口 **A = 2008-11-01~2010-11-17**（21,093 行 / 4,109 正例）；训练 `<2009-10-01`（17,537 / 2,327 / 13.27%）；验证 `≥2009-10-01`（3,556 / 1,782 / 50.11%）；名额 **356** 人；主口径名额制、阈值 0.5 只作对照；主模型逻辑回归 + **特征集 B**（21 列，去掉月份序/sin/cos 三列，E2 证据见 ADR 0005）、切分按时间、不打乱不分层、缓冲 0 天。
- 两个会反复踩的事实，先记住：
  1. **行序就是时间序**：30 个月块从 2008-05 走到 2010-11，月份只回绕过两次（跨年），块内日号从不回退；重建日期 2008-05-05–2010-11-17、561 个日期、全局单调。`notebooks/数据口径说明.md` §2.5 写的「没有年份只能随机切分」是错的，改那份文档时按这个结论改。重建日期**必须先把握手序号（`rowid`）落进数据集**，不能依赖 `SELECT *` 的返回顺序。
  2. **正例率随时间剧烈漂移**：2008 上半年 3.69% → 2009 上半年 12.97% → 2010 上半年 53.22%；`本次联系次数` 均值 2.845→1.899，`pdays` 从未联系占比 100%→34.4%。按时间切分后，验证集正例率约 50%，而训练期只有 13%，模型概率会系统性偏低、Brier 会很难看——报校准指标时必须说明这一点，别把它当成 bug 去「修」。
- 数据本身的缺口：2008 年 9 月整段缺失、2008-10 只有 80 行、无客户 ID（不能按人分组切分）、标签通话结束即可得（缓冲期 0，已写进 ADR 0002）。