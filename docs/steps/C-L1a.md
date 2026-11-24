# C-L1a：名额策略与阈值标定

**目标**：把「每天 356 个名额」变成一个可以发给服务端的分数阈值——名额边界（优先）与次优带边界（备选），阈值存原始 float、取真实分数、由配置指定策略并由发布门禁复核。

**验收**：
- `python -m pytest tests/unit/test_policy.py -q` 19 项全过（分位点标定、A 口径两档、原始 float 往返、配置拼错键/策略名/越界值）。
- `python -m bank_training.export --version <新版本>` 打印「名额策略 capacity_quantile：名额边界 X（选中 356 人），次优带边界 Y」，且 `bundle.json` 的 `decision.level_cutoffs` 是原始 float。
- 把验证集分数改成会撞边界的数据后 export 返回 1，门禁报「按阈值选中 N 人，名额是 356 人：阈值与名额对不上」。

**不做**：
- 不改名次与阈值的换算口径（A：优先 = 名额内全部；旧的三档口径已废，见 ADR 0023）。
- 不在服务端按批次重算分位点（一次只能看到一个客户，且不可复现）。
- 不把 `capacity` 复制进 `configs/decision.yaml`（唯一出处是 `base.yaml` 的 `quota.frac`）。
- 指标出口：`packages/training/src/bank_training/metric_specs.py` 一处登记 13 个指标的名字/中文名/方向/位数，
  `report.METRIC_ROWS` 由它派生，`metrics.py` 导入时校验清单一致。`pytest tests/unit/test_metric_specs.py`（7 项）

