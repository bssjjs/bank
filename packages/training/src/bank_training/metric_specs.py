"""把「指标出口」放到一个地方：名字、中文名、方向、小数位（讲义 C-L1a ②）。

为什么要有这张表：指标名会出现在报告、告警阈值、mlflow、模型卡里；如果每处各写一份
中文名和「越高越好/越低越好」，迟早出现「报告说越好越好、告警说越低越好」的矛盾。
这里登记一次，别处只能查。

`METRIC_SPECS` 与 `metrics.METRIC_NAMES` 必须一一对应，导入时就会校验（见 `assert_complete`）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    higher_is_better: bool
    digits: int = 4


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("precision_at_quota", "名额内精确率", True),
    MetricSpec("recall_at_quota", "名额内召回率", True),
    MetricSpec("f1_at_quota", "名额内 F1", True),
    MetricSpec("lift_at_quota", "名额内提升（相对基率）", True),
    MetricSpec("accuracy_at_quota", "名额内准确率", True),
    MetricSpec("tp_at_quota", "名额内打中的正例", True, 0),
    MetricSpec("fp_at_quota", "名额内多打的人", False, 0),
    MetricSpec("fn_at_quota", "名额外漏掉的正例", False, 0),
    MetricSpec("roc_auc", "ROC-AUC", True),
    MetricSpec("ap", "AP（平均精确率）", True),
    MetricSpec("brier", "Brier（越小越好）", False),
    MetricSpec("log_loss", "对数损失（越小越好）", False),
    MetricSpec("ece", "ECE（校准误差，越小越好）", False),
)

SPECS: dict[str, MetricSpec] = {spec.name: spec for spec in METRIC_SPECS}


def spec(name: str) -> MetricSpec:
    """按名字取登记项；没登记的名字直接报，不用猜。"""
    try:
        return SPECS[name]
    except KeyError:
        known = ", ".join(sorted(SPECS))
        raise KeyError(f"指标 {name!r} 没有登记：已登记的有 {known}") from None


def direction(name: str) -> str:
    return "越高越好" if spec(name).higher_is_better else "越低越好"


def format_value(name: str, value: float) -> str:
    """按登记的小数位格式化：人数是整数，比率 4 位。"""
    digits = spec(name).digits
    return f"{value:.{digits}f}"


def assert_complete(names: tuple[str, ...]) -> None:
    """指标清单与登记表必须一一对应，否则加指标时一定会漏掉某一处。"""
    registered = {spec.name for spec in METRIC_SPECS}
    missing = set(names) - registered
    extra = registered - set(names)
    if missing or extra:
        raise ValueError(
            f"指标登记表与 METRIC_NAMES 对不上：缺 {sorted(missing)}，多 {sorted(extra)}"
        )
