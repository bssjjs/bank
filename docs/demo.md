# 现场演示：任务书那八步，一条一条走

> 这份是**答辩当天的操作单**，不是设计文档。八步对应任务书「演示线」原文，每步只写：命令、该看到什么、看不到时查哪里。
> 所有数字都是 2026-11-24 在本仓库跑出来的；现场以本次输出为准，**报告里引用的数字必须来自本次输出**（任务书一票否决项）。
> 前置：`cd C:\Users\guojihuanjing\Desktop\bank`；一次准备 `$env:UV_CACHE_DIR=".venv\.tmp\uv-cache"; $env:UV_LINK_MODE="copy"`（装依赖才需要）。

## 0. 开场前 3 分钟：确认工作区干净、依赖在

```powershell
git status --short          # 应该为空（.tmp/ 与 data/processed/ 不跟踪）
uv run python -m pytest tests -q   # 432 passed 以内，几十秒
```

- 若 `models/serving/` 为空：演示第 2 步会重建，不用慌；服务第 3 步之前必须先跑第 2 步。

## 1. 一条命令从原始数据生成两表，打印行数与指纹；重跑指纹不变

```powershell
uv run python -m bank.freeze
uv run python -m bank.freeze --check
```

该看到（2026-11-24 实测）：

- `冻结完成`；窗口 `2008-11-01 ~ 2010-11-17`：**21,093 行 / 正例 4,109（19.48%）**
- 训练 `< 2009-10-01`：**17,537 行 / 2,327 正例（13.27%）**
- 验证 `≥ 2009-10-01`：**3,556 行 / 1,782 正例（50.11%）**
- **名额 356 人**；清洗 7 条规则命中 5 行（全是 R3 记账，致命 0）
- 内容指纹：`train` 前 16 位、`valid` 前 16 位；再跑 `--check` 输出「冻结校验通过」

要讲的一句：**指纹是规范化 CSV 的 sha256**（`repro.sha256_table`），换机器、换 pandas 只要内容一样就一样；`FROZEN.json` 是本机字节封条，所以 `--check` 比内容不比 parquet 字节（ADR 0003）。

看不到时：`data/bank-marketing.db` 是否在（不进 Git，换机器要拷回来）；`configs/base.yaml` 的 `features.set` 是否是 `B`。

## 2. 一条命令训练 + 一条命令导出模型包，打印基线与模型指标表、名额人数

```powershell
uv run python -m bank_training.train
uv run python -m bank_training.export --version 1.1.0 --promote
```

该看到（`train` 会打印 5 个 run 与一张表）：

| 模型 | AP | ROC-AUC | 名额内精确率 | 召回 | Brier | 名额内认购 |
| --- | --- | --- | --- | --- | --- | --- |
| 规则基线 | 0.6412 | 0.6510 | 0.7640 | 0.1526 | 0.4924 | 272 / 356 |
| 逻辑回归 A | 0.5959 | 0.6116 | 0.6180 | 0.1235 | 0.3104 | 220 / 356 |
| **逻辑回归 B（主）** | **0.6498** | **0.6438** | **0.7416** | **0.1481** | **0.2852** | **264 / 356** |
| 随机森林 A | 0.6229 | 0.6024 | 0.7500 | 0.1498 | 0.2875 | 267 / 356 |

- `export` 打印：特征 21 列、**必填输入 15 列**、名额 356、等级切点 **优先 ≥ 0.57298 / 备选 ≥ 0.253228**、回读一条样本的分数与等级、`run_id`、注册表状态（1.1.0 → production）、`current.json` 指向。
- 名额规则：`quota_n = ceil(3,556 × 0.10) = 356`；切点是验证集上真实存在的分数分位点（`method="higher"`，不插值、不四舍五入），导出时还有一道门禁：**按阈值选中的人数和名额差超过 1 人就拒绝发布**（ADR 0023）。

要讲的两句：**规则基线名额内多抓 8 人（272 vs 264）**，但 Brier 0.49 vs 0.29，概率质量差一倍；选逻辑回归还因为单条解释能加回分数（ADR 0005 / 0022）。

看不到时：`data/frozen/` 是否在（先跑第 1 步）；`models/serving/<版本>` 是否已存在（模型包**不覆盖**旧版本，换个版本号）。

## 3. 一条命令起服务：`/health` 200、`/docs` 可开

```powershell
uv run python -m bank_api      # 另开一个窗口
```

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/ready
Start-Process http://127.0.0.1:8000/docs
```

该看到：`/health` → `{"status":"ok","service":"bank-api"}`；`/ready` → 版本 1.1.0、`ready`；`/docs` 打开 Swagger，`PredictInput` 有 15 个字段、每个带中文说明、范围与枚举。

要讲的一句：请求模型**由模型包生成**（`schemas.build_input_model`），所以改字段只改 `configs/features.yaml` 与模型包，前端不用改（ADR 0007）。

## 4. 填一条样本，出分数、等级、主要因素、用途提示

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/predict -H "content-type: application/json" -d "{\"年龄\":46,\"职业\":\"退休\",\"婚姻状态\":\"已婚\",\"教育程度\":\"高等教育\",\"是否信贷违约\":\"否\",\"账户余额\":81204,\"住房贷款\":\"否\",\"个人贷款\":\"否\",\"联系方式\":\"固话\",\"联系日期\":12,\"联系月份\":\"三月\",\"本次联系次数\":2,\"距上次联系天数\":-1,\"历史联系次数\":3,\"上次营销结果\":\"成功\"}"
```

该看到：`score`、`level`（优先 / 备选 / 不建议）、`factors` 前 5 条（`账户余额 +1.395`、`上次营销结果=成功 +1.118`…）、`advice`、`request_id`、`model_version`。

- 前端页面在 `apps/web/`；另开一个终端跑 `node apps/web/server.mjs`，打开 `http://127.0.0.1:5173`，字段从 `GET /v1/model` 读，提交走 `POST /v1/predict`。现场不方便开页面时，用 `/docs` 的 Try it out 演示，同样算“填一条样本出结果”。

要讲的一句：因素 = 系数 × 标准化后取值，**全部因素之和 + 截距 == logit**，有测试逐行断言（ADR 0022）；所以界面上给的主要因素不是"近似解释"。

## 5. 越界值：在该字段上标错（后端 422）

把上一条的 `"年龄":46` 改成 `999` 再发一次：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/predict -H "content-type: application/json" -d "{\"年龄\":999, ...其余字段同上...}"
```

该看到：HTTP 422，`detail[0].loc = ["body","年龄"]`，`msg = "Input should be less than or equal to 95"`。

要讲的一句：`loc` 前面加 `"body"` 是刻意的（ADR 0007 补充），前端按 `loc[1]` 就能把错误挂到对应输入框；**错误详情只回 `loc/msg/type`**，不回显用户值——NaN 曾让 422 自己炸成 500，第 15 条坑记着。

再补一个 NaN 的例子（可选）：把某数值字段写成 `NaN`，仍然是 422，不是 500。

## 6. 预测记录多一行

```powershell
Get-Content data\processed\predictions.jsonl -Tail 1
```

该看到一行 `prediction_record.v2`：`ts / contract / request_id / model_version / score(6 位) / level / control / features_hash(64 位) / quota_frac / quota_n`。**没有客户特征值**，只有输入指纹。

要讲的一句：`control` 是**按 `sha256(seed:request_id)` 确定性抽 10%** 的对照组，同一请求号永远同一结论，不用存随机数；回流时对照组不进重训（ADR 0014）。

## 7. 对「线上」数据跑漂移，至少一个特征 PSI

```powershell
uv run python -m bank_training.drift
uv run python -m bank_training.watch --no-send
```

该看到（基准 = 训练集 17,537 行，对照 = 验证集 3,556 行）：

- **联系月份 PSI 3.4028**（严重）、住房贷款 0.8087、上次营销结果 0.6573、职业 0.4583、历史联系次数 0.3500、年龄 0.2797 均「严重」，共 6 个；
- `score` PSI **1.0851**；正例率 0.1327 → 0.5011（**+277.66%**）；
- `watch` 打印「检查完成：info 0 / warn 5 / alert 7」，退出码 **1**（有 alert）。

要讲的一句：**这不是数据错误，是真实的时期漂移**（日历推进 + 前几次营销的客户被筛过），所以模型概率系统性偏低、Brier 难看，我们报校准指标时如实写，不用验证集去拟合校准器（ADR 0008 / 0012）。

再补一句演练：`python -m bank_training.drill` 注入已知漂移（数值 PSI 12.4432、类别 3.1173、标签 +59.7%），同时断言没动的列（婚姻状态 0.0084）不报警——证明监控是"能响"的，不是摆设（ADR 0016）。

## 8. 质量门全绿

```powershell
uv run python scripts\ci.py
```

该看到：五道门依次 `[通过]`（ruff check、ruff format --check、mypy、lint-imports、pytest），末行 **`质量门：全绿`**，退出码 0；`pytest` 门里 **432 passed**、覆盖率 **86%**（门槛 85）。

```powershell
uv run python scripts\freeze_baseline.py --check   # 基线封条：9 个文件
uv run python scripts\replay_steps.py              # 可选：按 tag 回放，很慢
```

要讲的一句：门清单由 `gates()` 公开，`tests/architecture/test_quality_gate.py` 15 项守着它不被调低；**改门槛必须过测试**（ADR 0004）。

## 附：答辩六个必问，答在哪个文件

| 问题 | 一句话答案 | 出处 |
| --- | --- | --- |
| 标签窗口为什么这么长 | 标签在通话结束就可得，窗口 0 天；缓冲期 0 但切点落月份边界 | ADR 0002、`docs/problem.md` |
| 哪列被禁用凭什么 | `通话时长` 打完才知道（单变量 AUC 0.8076）；`pdays` 的 −1 是哨兵；标签列本身 | `docs/spec.md` 第 3 节、`configs/features.yaml` |
| 名额人数怎么算 | 验证集 3,556 × 10% = 356；切点是验证集分位点，不是概率阈值 | `docs/spec.md` 第 5 节、ADR 0023 |
| 缺一列请求在哪层拦住 | `api/schemas` 的动态请求模型 → 422；模型包里 `bundle.prepare` 再查 `required_columns` | ADR 0006 / 0007、`tests/service` |
| 换更好的模型上线几步、怎么退 | `export --version X` 登记 candidate → `registry promote X` 查门禁 → `current.json` 热加载；回退 `registry rollback 旧版本` 只改指针 | ADR 0013 / 0015 / 0019 |
| 上游改字段单位在哪一步发现 | 特征目录 `features.yaml` 的范围/类型 + `catalog` 与 `schema` 对账 → 冻结/训练/接口都从它读；再靠 PSI 与 `watch` 兜底 | ADR 0004、`configs/features.yaml` |

## 附：出问题时先看这几处

- **命令报「找不到模型包」**：先跑 `python -m bank_training.export --version 1.1.0 --promote`。
- **服务 `/ready` 503**：`models/serving/current.json` 是否指向存在的版本目录；`/metrics` 的 `reload.last_reload_error` 有原因。
- **`bank.freeze --check` 红**：配置或数据被改过（指纹对不上）；不要用 `--write` 覆盖，先查 `git diff configs/`。
- **`scripts/ci.py` 红**：看最后一道门的末 20 行；只跑 `pytest` 不等于过门。