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

# 5) 测试与质量门（交付前最后一关）
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\ci.py
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

## 目录

```
packages/core       口径规则、数据管道、特征目录、冻结（谁都不依赖）
packages/training   训练、评估、模型包（只依赖 core）
packages/api        推理服务（只依赖 core）
apps/web            前端（M6）
configs/            base.yaml 口径数字；features.yaml 字段唯一出处（含禁用列）
data/raw|frozen|processed
models/serving/     模型包，每版一个目录，不覆盖
scripts/ci.py       五道门：ruff check / ruff format / mypy / lint-imports / pytest
tests/              unit / contract / architecture / service
docs/               problem / spec / data_card / pitfalls / adr / modules / notes
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
| `docs/adr/` | 决策记录（数据集口径、时间切分、指纹与冻结、三包结构） |
| `docs/notes/参考项目经验.md` | 两个参考项目哪些机制值得搬 |
| `notebooks/数据口径说明.md` | 早期口径分析与结论（§2.5 已更正为按时间切分） |

## 依赖

`uv.lock` 锁定全部依赖；Python 3.12。`.venv` 由 uv 管理（没有 pip，加依赖请用 `uv add`）。