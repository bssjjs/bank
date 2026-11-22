# AGENTS.md：给编程智能体的规矩

> 只写每次开工都要知道、而且**已经存在**的东西。每完成一步，在第六节追加这一步留下的规矩；不提前写后面步骤的规则。

## 一、这是什么项目

- 课程分组项目，任务书九个模块（M1 口径 → M9 文档），最终交「能现场演示的系统 + 设计文档 + 答辩」。**模型分数不占分**：架构、取舍、能跑通、数字有出处才是分。
- 业务问题：银行电话营销。**在打电话之前**，用当时能拿到的客户信息判断他会不会认购定期存款，再按每天的外呼名额排出今天该打给谁。
- 数据：UCI Bank Marketing（`data/bank-marketing.db`，单表 `银行营销数据`，只读，17 列）。45,211 行，正例 5,289（11.70%）。
- 代码现状：**三包（`packages/core|training|api`）+ 口径层 + 冻结层 + 特征目录 + 质量门**。`packages/core/src/bank/` 是唯一代码位置（`core` 谁都不依赖）：`schema.py` 放规则与常量，`dataset.py` 构建数据集并自检，`timeaxis.py` 重建时间轴，`profile.py` 出画像数字（M1 文档的唯一出处），`raw.py` 是唯一只读入口，`clean.py` 是清洗台账，`split.py` 按时间切分，`repro.py` 算指纹，`config.py` 强校验 `configs/base.yaml`，`freeze.py` 产出冻结集与 manifest，`yamlkit.py` 是两份 YAML 共用的严格读取器，`catalog.py` 读 `configs/features.yaml` 并与 `schema.py` 对账。`build.py` / `profile.py` / `freeze.py` 是三个可执行入口。`packages/training/src/bank_training/` 是 M3 的训练层：`data.py` 读冻结集、`pipeline.py` 组预处理器与模型、`metrics.py` 算指标、`baselines.py` 规则基线、`explain.py` 单条解释、`model_config.py` 读 `configs/model.yaml`，`train.py` / `experiments.py` 是两个入口。`packages/api`（`bank_api`）已建但**还没有业务代码**。`scripts/ci.py` 是五道门的质量门。`tests/` 有 104 项测试（unit + architecture）。分析结论在 `notebooks/`。
- 服务、前端**还不存在**，别假设它们存在，也别在文档里引用。已经存在的是：口径层、冻结层、特征目录（`configs/features.yaml` + `catalog.py`）、质量门（`scripts/ci.py`）、清洗规则注册表（`rules.py`）、训练层（`packages/training`）。

### 五个人的角色

| 角色 | 模块 | 主要交付 |
| --- | --- | --- |
| 架构与质量 | M8、M9 | 三包与依赖方向、配置强校验、特征目录、注册表、质量门 `scripts/ci.py`；`architecture.md`、ADR、模块卡 |
| 数据 | M1、M2 | 一句话定义、口径规格表、禁用列、数据卡、踩坑清单；原始入口、清洗台账、标签、时间切分、冻结与指纹、manifest |
| 模型与评估 | M3 | 两条基线、Pipeline、名额规则、指标口径、运行记录、单条解释、模型卡 |
| 发布与服务 | M4、M5、M7 | ModelBundle、FastAPI 契约与 422、预测记录、PSI 漂移 |
| 前端 | M6 | 按 `/v1/model` 动态生成表单、结果卡、字段级错误、服务状态 |

跨人的硬约定：**每人至少一份 ADR**；答辩随机点人，每人都要能独立把演示线八步跑通。

## 二、怎么算完成

现在能跑的命令（不要自创第七条）：

```powershell
.venv\Scripts\python.exe -m bank.build
.venv\Scripts\python.exe -m bank.profile
.venv\Scripts\python.exe -m bank.freeze        # 加 --check 只校验不重建
.venv\Scripts\python.exe -m bank_training.train
.venv\Scripts\python.exe -m bank_training.experiments
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\ci.py
```

- `bank.build` 必须打印「口径层构建完成」，行数 45,211、正例 5,289，并写出 `data/model_dataset.parquet` 与 `data/feature_spec.json`。
- `bank.profile` 是文档数字的唯一出处：`docs/data_card.md`、`docs/spec.md` 里的行数、正例率、漂移表、口径冲突行数、泄漏 AUC 全部来自它。改数据或改口径后，先跑它，再改文档。
- `bank.freeze` 是 M2 的正式产物：读只读库 + `configs/base.yaml`，写出 `data/frozen/{train.parquet, valid.parquet, manifest.json, FROZEN.json}`。必须打印「冻结完成」以及 21,093/4,109、17,537/2,327、3,556/1,782、名额 356、7 条清洗规则命中 5 行（全是 R3 记账）。跑第二遍 `FROZEN.json` 的 sha256 必须不变。`--check` 只比对指纹，输出「冻结校验通过」。
- 训练、评估、服务只读 `data/frozen/`，**不要再从原始库现算**；`data/processed/` 是临时产物，随时可删。
- `pytest tests` 必须全过（104 项）：14 项 schema、14 项时间轴（4 项拿真实库校验 561 个日期/30 个月块/缺 2008-09）、25 项冻结层（清洗规则、切分边界、指纹稳定、manifest 契约数字）、13 项特征目录（合法性 + 与 schema 对账）、13 项配置（拼错键/重复键/越界值/名额公式）、9 项架构（依赖方向、包布局、质量门存在）、5 项注册表（规则登记、顺序、致命标记）、11 项训练（指标口径、规则基线分值、Pipeline 可复现、解释排序、模型配置校验）。
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
apps/web                     前端（技术栈待定，见 ADR）
configs/                     base.yaml 放口径数字；features.yaml 是字段唯一出处（含禁用列）
data/raw|frozen|processed    raw 只读入口、processed 是产物、frozen 冻结后进 Git
models/serving/              模型包，每版一个目录，不覆盖
scripts/                     ci.py 一条命令的质量门
tests/unit|contract|architecture|service
docs/                        problem/spec/data_card/pitfalls/architecture/model_card
docs/adr/                    决策记录，0001 起编号，一份一个决定
docs/modules/                模块卡：目标 / 验收 / 不做
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
- 未开始：M4 模型包、M5 服务、M6 前端、M7 漂移、M9 文档。
- 已定的口径（ADR 0002 为准，别再问）：窗口 **A = 2008-11-01~2010-11-17**（21,093 行 / 4,109 正例）；训练 `<2009-10-01`（17,537 / 2,327 / 13.27%）；验证 `≥2009-10-01`（3,556 / 1,782 / 50.11%）；名额 **356** 人；主口径名额制、阈值 0.5 只作对照；主模型逻辑回归 + **特征集 B**（21 列，去掉月份序/sin/cos 三列，E2 证据见 ADR 0005）、切分按时间、不打乱不分层、缓冲 0 天。
- 两个会反复踩的事实，先记住：
  1. **行序就是时间序**：30 个月块从 2008-05 走到 2010-11，月份只回绕过两次（跨年），块内日号从不回退；重建日期 2008-05-05–2010-11-17、561 个日期、全局单调。`notebooks/数据口径说明.md` §2.5 写的「没有年份只能随机切分」是错的，改那份文档时按这个结论改。重建日期**必须先把握手序号（`rowid`）落进数据集**，不能依赖 `SELECT *` 的返回顺序。
  2. **正例率随时间剧烈漂移**：2008 上半年 3.69% → 2009 上半年 12.97% → 2010 上半年 53.22%；`本次联系次数` 均值 2.845→1.899，`pdays` 从未联系占比 100%→34.4%。按时间切分后，验证集正例率约 50%，而训练期只有 13%，模型概率会系统性偏低、Brier 会很难看——报校准指标时必须说明这一点，别把它当成 bug 去「修」。
- 数据本身的缺口：2008 年 9 月整段缺失、2008-10 只有 80 行、无客户 ID（不能按人分组切分）、标签通话结束即可得（缓冲期 0，已写进 ADR 0002）。