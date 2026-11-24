"""名额策略：把「每批名额比例」标定成阈值（讲义 C-L1a，ADR 0023 选定的 A 口径）。

服务对单个客户打分时，不知道同一批其他人的分数，所以名额比例必须先换成一组
**真实存在的分数**（验证集分数的分位点，`method="higher"`，不在两个人之间插值），
随模型包一起发布；之后任何一个分数都能翻译成等级。

两个阈值，都由验证集分数标定：

- `quota_threshold`——名额边界（`capacity` 分位）。`分数 >= 它` 的正好是名额内的人，
  这一档叫「优先」：今天的名额先给他们；
- `watch_threshold`——名额外的次优带（`medium_fraction` 分位）。`分数 >= 它` 但没进
  名额的，是「备选」：今天的名额用完了，明天从这些人里挑。

`medium_fraction > capacity` 时次优带才在名额外；写反了参数校验会当场拦住。

阈值**不四舍五入**：6 位小数会把边界上的人挤出去或多拉进来，讲义 C-L1a 明确要求存
原始 float。模型包里的 `decision.level_cutoffs` 就是这两个值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bank import schema as S
from bank.yamlkit import YamlError, load_mapping

LEVELS = ("优先", "备选", "不建议")
POLICY_NAMES = ("capacity_quantile",)
DECISION_CONFIG_PATH = S.ROOT / "configs" / "decision.yaml"


class PolicyError(ValueError):
    """使用者能看懂的错：参数写错、分数不合法。"""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionParams(_Strict):
    """`configs/decision.yaml` 的 `decision` 段。

    `capacity` 不在这里：名额比例的唯一出处是 `configs/base.yaml` 的 `quota.frac`，
    两个文件各写一份迟早会分叉。
    """

    policy: str
    medium_fraction: float = Field(gt=0.0, lt=1.0)


class DecisionConfig(_Strict):
    schema_version: int
    decision: DecisionParams


def load_decision_config(path: Path | None = None) -> DecisionConfig:
    """读决策配置；拼错键、写错范围、策略名不认识，都在这里报。"""
    target = path or DECISION_CONFIG_PATH
    try:
        raw = load_mapping(target, what="决策配置")
    except YamlError as exc:
        raise PolicyError(str(exc)) from exc
    try:
        config = DecisionConfig.model_validate(raw)
    except Exception as exc:  # pydantic 的 ValidationError 长得不好看，压成「键: 原因」
        raise PolicyError(f"决策配置校验失败：{exc}") from exc
    if config.schema_version != 1:
        raise PolicyError(f"决策配置版本 {config.schema_version} 不认识：当前只支持 1")
    if config.decision.policy not in POLICY_NAMES:
        raise PolicyError(
            f"未知决策策略 {config.decision.policy!r}：可用的有 {', '.join(POLICY_NAMES)}"
        )
    validate_params(0.10, config.decision.medium_fraction)
    return config


def _scores(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype="float64").reshape(-1)
    if scores.size == 0:
        raise PolicyError("没有分数：至少要一个人才能标定阈值")
    if not np.all(np.isfinite(scores)):
        raise PolicyError("分数里有 NaN 或 inf：先修好打分，再标定阈值")
    return scores


def _quantile(scores: np.ndarray, share: float) -> float:
    """分位点取真实存在的分数（higher），不插值。"""
    return float(np.quantile(scores, 1.0 - share, method="higher"))


@dataclass(frozen=True)
class DecisionPolicy:
    """标定一次、到处用：训练后写进模型包，服务按阈值判等级。"""

    kind: str
    capacity: float
    medium_fraction: float
    quota_threshold: float
    watch_threshold: float

    def level(self, score: float) -> str:
        return self.level_of(np.asarray([score], dtype="float64"))[0]

    def level_of(self, scores: Any) -> list[str]:
        values = _scores(scores)
        quota = values >= self.quota_threshold
        watch = values >= self.watch_threshold
        return [
            "优先" if a else ("备选" if b else "不建议") for a, b in zip(quota, watch, strict=True)
        ]

    def selected(self, scores: Any) -> np.ndarray:
        """名额内的人（布尔数组）：`分数 >= quota_threshold`。"""
        return _scores(scores) >= self.quota_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "capacity": self.capacity,
            "medium_fraction": self.medium_fraction,
            "quota_threshold": self.quota_threshold,
            "watch_threshold": self.watch_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionPolicy:
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise PolicyError(f"策略字段不认识：{sorted(unknown)}（允许的是 {sorted(expected)}）")
        missing = expected - set(data)
        if missing:
            raise PolicyError(f"策略字段缺失：{sorted(missing)}")
        if data["kind"] not in POLICY_NAMES:
            raise PolicyError(f"未知策略 {data['kind']!r}：可用的有 {', '.join(POLICY_NAMES)}")
        return cls(
            kind=str(data["kind"]),
            capacity=float(data["capacity"]),
            medium_fraction=float(data["medium_fraction"]),
            quota_threshold=float(data["quota_threshold"]),
            watch_threshold=float(data["watch_threshold"]),
        )


def validate_params(capacity: float, medium_fraction: float) -> None:
    """参数由策略自己校验：配置写错当场报，不等训练完才发现。"""
    if not 0.0 < capacity < 1.0:
        raise PolicyError(f"capacity 必须在 (0, 1) 内，收到 {capacity}")
    if not 0.0 < medium_fraction < 1.0:
        raise PolicyError(f"medium_fraction 必须在 (0, 1) 内，收到 {medium_fraction}")
    if medium_fraction <= capacity:
        raise PolicyError(
            f"medium_fraction 必须大于 capacity（次优带要在名额外面），收到 "
            f"capacity={capacity}、medium_fraction={medium_fraction}"
        )


def fit_capacity_quantile(
    scores: Any, *, capacity: float, medium_fraction: float
) -> DecisionPolicy:
    """在验证集分数上标定两个阈值。"""
    validate_params(capacity, medium_fraction)
    values = _scores(scores)
    return DecisionPolicy(
        kind="capacity_quantile",
        capacity=capacity,
        medium_fraction=medium_fraction,
        quota_threshold=_quantile(values, capacity),
        watch_threshold=_quantile(values, medium_fraction),
    )


def fit_policy(
    name: str, scores: Any, *, capacity: float, medium_fraction: float
) -> DecisionPolicy:
    if name not in POLICY_NAMES:
        raise PolicyError(f"未知策略 {name!r}：可用的有 {', '.join(POLICY_NAMES)}")
    return fit_capacity_quantile(scores, capacity=capacity, medium_fraction=medium_fraction)
