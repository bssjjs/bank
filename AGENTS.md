# AGENTS.md：给编程智能体的规矩

> 只写每次开工都要知道、而且**已经存在**的东西。每完成一步，在第六节追加这一步留下的规矩；不提前写后面步骤的规则。

## 一、这是什么项目

- 课程分组项目，任务书九个模块（M1 口径 → M9 文档），最终交「能现场演示的系统 + 设计文档 + 答辩」。**模型分数不占分**：架构、取舍、能跑通、数字有出处才是分。
- 业务问题：银行电话营销。**在打电话之前**，用当时能拿到的客户信息判断他会不会认购定期存款，再按每天的外呼名额排出今天该打给谁。
- 数据：UCI Bank Marketing（`data/bank-marketing.db`，单表 `银行营销数据`，只读，17 列）。45,211 行，正例 5,289（11.70%）。
- 代码现状：**三包（`packages/core|training|api`）+ 口径层 + 冻结层 + 特征目录 + 质量门**。`packages/core/src/bank/` 是唯一代码位置（`core` 谁都不依赖）：`schema.py` 放规则与常量，`dataset.py` 构建数据集并自检，`timeaxis.py` 重建时间轴，`profile.py` 出画像数字（M1 文档的唯一出处），`raw.py` 是唯一只读入口，`clean.py` 是清洗台账，`split.py` 按时间切分，`repro.py` 算指纹，`config.py` 强校验 `configs/base.yaml`，`freeze.py` 产出冻结集与 manifest，`yamlkit.py` 是两份 YAML 共用的严格读取器，`catalog.py` 读 `configs/features.yaml` 并与 `schema.py` 对账。`build.py` / `profile.py` / `freeze.py` 是三个可执行入口。`packages/core/src/bank/` 里另有 M4/M5 共用的推理路径：`features.py` 组预处理器、`explain.py` 单条解释、`bundle.py` 读写模型包（`bundle.json` + `model.joblib` + `current.json`）。`packages/training/src/bank_training/` 是 M3 的训练层：`data.py` 读冻结集、`pipeline.py` 组预处理器与模型、`metrics.py` 算指标、`baselines.py` 规则基线、`explain.py` 单条解释、`model_config.py` 读 `configs/model.yaml`，`train.py` / `experiments.py` / `export.py` 是三个入口（导出模型包走 `export.py`）。`packages/api/src/bank_api/` 是 M5 的服务层：`config.py` 读 `configs/api.yaml`、`schemas.py` 按模型包生成请求模型与响应模型、`app.py` 建 FastAPI 应用（`create_app`）与 `main()`，`__main__.py` 让 `python -m bank_api` 一条命令起服务。`scripts/ci.py` 是五道门的质量门（门清单由公开函数 `gates()` 给出，`tests/architecture/test_quality_gate.py` 守着门槛不被调低）。`scripts/freeze_baseline.py` 给第一轮结果加文件封条（`baselines/`），`scripts/replay_steps.py` 按 `step-*` tag 回放质量门。`tests/` 有 224 项测试（unit + contract + service + architecture + exam 考卷）。分析结论在 `notebooks/`。
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

现在能跑的命令（十二条，不要自创第十三条）：

```powershell
.venv\Scripts\python.exe -m bank.build
.venv\Scripts\python.exe -m bank.profile
.venv\Scripts\python.exe -m bank.freeze        # 加 --check 只校验不重建
.venv\Scripts\python.exe -m bank_training.train
.venv\Scripts\python.exe -m bank_training.experiments
.venv\Scripts\python.exe -m bank_training.export --version 1.0.0
.venv\Scripts\python.exe -m bank_api          # 起服务，127.0.0.1:8000
.venv\Scripts\python.exe -m bank_training.drift  # 对「线上」批次跑 PSI，默认对比验证集
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\ci.py
.venv\Scripts\python.exe scripts\freeze_baseline.py --check   # 基线封条，改了数字必须复跑
.venv\Scripts\python.exe scripts\replay_steps.py              # 慢：8 个 tag 各建一次环境再跑门
```

- `bank.build` 必须打印「口径层构建完成」，行数 45,211、正例 5,289，并写出 `data/model_dataset.parquet` 与 `data/feature_spec.json`。
- `bank.profile` 是文档数字的唯一出处：`docs/data_card.md`、`docs/spec.md` 里的行数、正例率、漂移表、口径冲突行数、泄漏 AUC 全部来自它。改数据或改口径后，先跑它，再改文档。
- `bank.freeze` 是 M2 的正式产物：读只读库 + `configs/base.yaml`，写出 `data/frozen/{train.parquet, valid.parquet, manifest.json, FROZEN.json}`。必须打印「冻结完成」以及 21,093/4,109、17,537/2,327、3,556/1,782、名额 356、7 条清洗规则命中 5 行（全是 R3 记账）。跑第二遍 `FROZEN.json` 的 sha256 必须不变。`--check` 只比对指纹，输出「冻结校验通过」。
- 训练、评估、服务只读 `data/frozen/`，**不要再从原始库现算**；`data/processed/` 是临时产物，随时可删。
- `pytest tests` 必须全过（224 项），按目录分：unit 117 / contract 6 / service 12 / architecture 37 / exam 52。
  - unit 117：schema 14、时间轴 14、冻结层 25、特征目录 14、配置 11、注册表 5、训练 11、模型包 12、漂移 11。
  - contract 6：OpenAPI 的必填/范围/枚举/批量上限与模型包对账。
  - service 12：探针、字段级 422、批量上限、预测记录、无包时 503（干净检出时 `tests/support.py` 会先导出一个模型包）。
  - architecture 37：依赖方向与包布局 10、质量门守门 15、基线封条 8、回放 tag 4。
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
11. 不创建 `.github/`（本课程手工跑 `scripts/ci.py`，见卷一 A1）。
12. **不照搬参考项目的文件或约定**。参考项目里的东西不等于本项目需要它；新增任何文件（`.gitattributes` 这种纯工程约定除外）都要能说出「本项目为什么需要」，说不出就不加。例子：`CLAUDE.md` 只有用 Claude Code 才有人读，本项目不用，2026-11-24 已删——别因为参考项目有就加回来。

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
configs/                     base.yaml 放口径数字；features.yaml 是字段唯一出处（含禁用列）
data/raw|frozen|processed    raw 只读入口、processed 是产物、frozen 冻结后进 Git
models/serving/              模型包，每版一个目录，不覆盖
scripts/                     ci.py 质量门；freeze_baseline.py 基线封条；replay_steps.py 按 tag 回放
baselines/                   第一轮结果的文件封条：v1/ 快照 + SHA256SUMS
tests/unit|contract|architecture|service|exam
docs/                        problem/spec/data_card/pitfalls/architecture/model_card
docs/adr/                    决策记录，0001 起编号，一份一个决定
docs/modules/                模块卡：目标 / 验收 / 不做
docs/steps/                  步骤卡：每步三行（目标/验收/不做），文件名对应 tag
docs/cards/                  组件卡：指纹 / 配置 / 特征目录，每张九节
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
- 已完成 M4（模型包与版本）：`packages/core/src/bank/bundle.py`（`model_bundle.v1` 契约、`sha256(canonical_json)` 校验和、staging 原子写、不覆盖旧版本、`current.json` 指针）、`packages/core/src/bank/features.py` 与 `explain.py`（训练与服务共用的预处理与解释路径，所以 `bank_api` 不需要 import `bank_training`）、`packages/training/src/bank_training/export.py`（`python -m bank_training.export --version 1.0.0`，发布门禁 AP≥0.50、名额内精确率≥0.60、特征数≥10）、`docs/adr/0006-模型包与发布.md`、`tests/unit/test_bundle.py`（12 项）。当前线上版本 `models/serving/1.0.0`：特征集 B 21 列、必填输入 15 列、等级切点 优先≥0.6387 / 备选≥0.5730、验证集 AP 0.6498、名额内精确率 0.7416、mlflow run `7d0aa8a9fa084615ade550de1055e790`。模型包不进 Git，演示前要重跑 export。
- 已完成 M5（服务）：`packages/api/src/bank_api/` 四个模块 + `configs/api.yaml`（批量上限 100、请求号前缀 bank、预测记录路径）、`docs/adr/0007-服务契约与错误码.md`。五条路由：`/health` 永远 200、`/ready` 与 `/v1/model` 在包缺失时 503、`/v1/predict`（字段级 422，`loc` 形如 `["body","年龄"]`）、`/v1/predict/batch`（上限 100）。请求模型由模型包 15 个输入列动态生成，类别列用运行时 `Literal` 内联 `enum`；每条预测追加一行 `{ts, request_id, model_version, score, level, control}` 到 `data/processed/predictions.jsonl`，不写客户特征值。`python -m bank_api` 起服务，`/docs` 可打开。
- 已完成 M7（漂移与回流）：`packages/core/src/bank/drift.py`（PSI：等频 10 箱、边界只取基准集、缺失单独一箱、类别未见值合并「其他」）、`packages/training/src/bank_training/drift.py`（`python -m bank_training.drift`，基准=训练集、对照默认验证集、落 `data/processed/drift.json` 与 mlflow run `m7-drift`）、`docs/adr/0008-漂移监控与回流.md`（阈值 0.1/0.25、只看 15 个原始输入 + score、标签回流与对照组、三条重训条件）、`tests/unit/test_drift.py`（11 项）。实测：联系月份 PSI 3.4028、住房贷款 0.8087、上次营销结果 0.6573、职业 0.4583、历史联系次数 0.3500、年龄 0.2797 为「严重」；score 1.0851；正例率 13.27% → 50.11%（+277.66%）。
- 已完成 A1（地基与守门，§卷一参考）：`.gitattributes`（LF/二进制声明）、`.editorconfig`、三个 `py.typed`、`.importlinter` 契约改名 `core-independent` / `training-api-independent`、`scripts/ci.py` 五道门 + 覆盖率门槛 85（门清单由 `gates()` 公开，`tests/architecture/test_quality_gate.py` 15 项守着不被调低）。161 项测试时全绿、覆盖率 86%。
- 已完成 A2（基线 + 考卷 + 卡片 + 回放）：`baselines/`（8 个快照 + `SHA256SUMS`，`scripts/freeze_baseline.py --write/--check`；快照 JSON 统一 UTF-8+LF 写出，否则 CRLF 会让封条在 clone 后对不上）、`tests/exam/` 三份考卷 52 项（配置 18 / 目录 16 / 指纹 18，人写的，智能体只读）、`docs/cards/` 三张卡、`docs/steps/` 十份步骤卡、`scripts/replay_steps.py`（8 个 `step-*` tag 各建一次干净 worktree：装环境 → 重建产物 → 跑门）、`tests/support.py::ensure_served_bundle`（缺模型包时先导出一个，服务与契约测试靠它才能在干净检出里跑）。
- 全量复核：`pytest tests` **224 项全过**、覆盖率 86%、`scripts/ci.py` 末行「质量门：全绿」；`scripts/replay_steps.py` 实测 **8 个 tag 全通过**（step-A1/A2/M3/M4/M5/M7/M8/M9，2026-11-23）。回放要先把 `data/bank-marketing.db` 拷进工作目录（它不进 Git），这一步由脚本自己做。
- **未开始：M6 前端**，这是负责人的决定——他自己写，不要替他实现。设计卡在 `docs/modules/m6-前端页面.md`（目标/验收/不做），要实现时按任务书 M6：字段从 `GET /v1/model` 动态生成、结果卡、字段级错误、服务状态提示、只走 HTTP；接口已经就绪（`/v1/model` 返回 15 个输入字段，`/v1/predict` 30 秒内可接上）。
- **已关闭的口径问题：`联系方式` 不是隐私列**（负责人 2026-11-23 决定）。它存的是联系渠道类型（手机 / 固话 / 未知，13,020 行是「未知」），不是电话号码，不构成个人信息。因此它继续留在 21 个特征与 15 个必填输入里，分析与前端都照常使用。之前记过一条「待移除」的待办，已随这次决定作废——不要因为它听起来像隐私就去掉它，那样会让指标无故下降，还得改整条冻结/训练/导出/文档链。
- 已定的口径（ADR 0002 为准，别再问）：窗口 **A = 2008-11-01~2010-11-17**（21,093 行 / 4,109 正例）；训练 `<2009-10-01`（17,537 / 2,327 / 13.27%）；验证 `≥2009-10-01`（3,556 / 1,782 / 50.11%）；名额 **356** 人；主口径名额制、阈值 0.5 只作对照；主模型逻辑回归 + **特征集 B**（21 列，去掉月份序/sin/cos 三列，E2 证据见 ADR 0005）、切分按时间、不打乱不分层、缓冲 0 天。
- 两个会反复踩的事实，先记住：
  1. **行序就是时间序**：30 个月块从 2008-05 走到 2010-11，月份只回绕过两次（跨年），块内日号从不回退；重建日期 2008-05-05–2010-11-17、561 个日期、全局单调。`notebooks/数据口径说明.md` §2.5 写的「没有年份只能随机切分」是错的，改那份文档时按这个结论改。重建日期**必须先把握手序号（`rowid`）落进数据集**，不能依赖 `SELECT *` 的返回顺序。
  2. **正例率随时间剧烈漂移**：2008 上半年 3.69% → 2009 上半年 12.97% → 2010 上半年 53.22%；`本次联系次数` 均值 2.845→1.899，`pdays` 从未联系占比 100%→34.4%。按时间切分后，验证集正例率约 50%，而训练期只有 13%，模型概率会系统性偏低、Brier 会很难看——报校准指标时必须说明这一点，别把它当成 bug 去「修」。
- 数据本身的缺口：2008 年 9 月整段缺失、2008-10 只有 80 行、无客户 ID（不能按人分组切分）、标签通话结束即可得（缓冲期 0，已写进 ADR 0002）。