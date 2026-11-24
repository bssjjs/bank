# 服务卡：推理服务（E-L2 热加载与指标）

> 对应代码：`packages/api/src/bank_api/{app,store,service_metrics,schemas,config}.py`；
> 决策记录：`docs/adr/0007-服务契约与错误码.md`、`docs/adr/0015-热加载与服务指标.md`。

## 1. 是什么

把模型包暴露成 HTTP 服务：一条样本进、分数/等级/主要因素出；模型包换了不用重启。
它只读 `models/serving/` 下的文件，不 import 训练代码（由 import-linter 与架构测试守着）。

## 2. 谁用

- 前端（M6，负责人自己写）：`GET /v1/model` 拿字段清单与范围，`POST /v1/predict` 提交；
- 业务侧演示：答辩现场点页面或 `curl`；
- 运维视角：`GET /health`、`GET /ready`、`GET /metrics`。

## 3. 输入

- 请求体：15 个必填字段（由模型包的 `inputs` 决定，不是写死的），数值有 `minimum/maximum`，
  类别字段是 `enum`（运行时 `Literal` 内联，`/docs` 里看得见）；
- 查询参数：`control=true|false`（对照组标记，不填按请求号哈希抽 10%）；
- 配置：`configs/api.yaml`（批量上限、请求号前缀、预测记录路径）。

## 4. 输出

- `Prediction`：`request_id`、`model_version`、`score`（6 位小数）、`level`（优先/备选/不建议）、
  `factors`（前 5 个因素，带贡献值与原始取值）、`advice`（中文建议语）；
- `BatchResult`：一个批次共用一个 `request_id` 与一个对照组标记；
- `GET /metrics`：`service_metrics.v1`，见第 5 节；
- 副作用：每条预测追加一行 `data/processed/predictions.jsonl`（`prediction_record.v2`，不写特征值）。

## 5. 契约

| 契约 | 位置 | 关键字段 |
| --- | --- | --- |
| OpenAPI | `GET /openapi.json`（`/docs` 可读） | 字段名/类型/范围/枚举/必填、批量上限 |
| 模型信息 | `GET /v1/model` | `version`、`inputs`、`decision`、`metrics`、`training` |
| 预测记录 | `prediction_record.v2` | `ts`、`contract`、`request_id`、`model_version`、`score`、`level`、`control`、`features_hash`、`quota_frac`、`quota_n` |
| 服务指标 | `service_metrics.v1` | `uptime_s`、`requests`、`by_route`、`by_status`、`predictions`、`by_level`、`latency_ms_mean`、`latency_ms_max`、`model`、`reload` |

## 6. 错误边界

| 情况 | 状态码 | 表现 |
| --- | --- | --- |
| 字段越界/枚举错/缺字段 | 422 | `detail[].loc = ["body","年龄"]`，`msg` 是人话 |
| 批量为空 / 超过上限 | 422 | `loc = ["body"]`，消息写明上限与实收条数 |
| 没有模型包 | 503 | `/health` 仍 200；`/ready`、`/v1/model`、预测接口 503 |
| 指针文件坏 / 新包校验不过 | 200 + 旧模型 | `/metrics.reload.last_reload_error` 有值，`reload_failures` 加一 |
| 预测记录写不进去 | 500 | 不吞异常（记录是回流的前提，静默丢等于毁证据） |

## 7. 不做

- 不做鉴权、限流、多租户——课程项目，单机演示；
- 不做 Prometheus 文本格式与采集端；
- 不做多 worker 的全局指标聚合（进程内计数，重启清零）；
- 不做 A/B 的分流网关（对照组只做记录与统计，不改变谁被打电话）；
- 不做前端（M6 由负责人自己写，字段从 `/v1/model` 来）。

## 8. 怎么验收

```powershell
.venv\Scripts\python.exe -m pytest tests/service tests/contract -v
.venv\Scripts\python.exe -m bank_api          # 另开一个窗口
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
```

热加载：服务不停，另开一个窗口 `python -m bank_training.export --version 1.1.0 --promote`，
再 `curl /ready`，`model_version` 已变成 1.1.0，进程 PID 没变。

## 9. 为什么这样定

- 指针文本比对而不是文件 mtime：mtime 在不同文件系统上精度不一，且"内容没变但被 touch"会造成无谓重载；
- 保留旧版本而不是崩：换模型是常规操作，半截包不应该让整个服务不可用；
- 进程内指标而不是外部系统：没有采集端，最小可用的出口比没有出口强；
- 记录只写决策事实（分数、等级、对照组、输入指纹）：客户特征值留在请求里，不进日志。