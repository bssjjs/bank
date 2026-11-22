"""特征目录：字段类型、范围、取值、中文名、是否禁用的**唯一出处**。

谁用它：
- 数据检查（R1-R7）看范围与取值白名单；
- 训练只用 `feature_names()` 返回的列；
- 接口（M5）按它生成校验规则与 422 文案；
- 页面（M6）按它渲染表单、下拉框。

因此这里不做"尽力而为"的宽容：目录与 `schema.py` 打架就抛错，谁也别猜。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from bank import repro
from bank import schema as S
from bank.yamlkit import YamlError, load_mapping

CATALOG_PATH = S.ROOT / "configs" / "features.yaml"

# 三条列必须标 forbidden：标错/漏标就是泄漏事故，启动即报错
REQUIRED_FORBIDDEN = (S.DURATION, S.PDAYS, S.TARGET_RAW)

FieldType = Literal["numeric", "binary", "categorical", "ordinal", "time", "date", "id"]
FieldRole = Literal["feature", "source"]


class CatalogError(RuntimeError):
    """目录与代码口径不一致、或目录自身不合法时抛这个。"""


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: FieldType
    display: str
    # feature = 直接进模型的列；source = 只用来派生（比如「联系月份」→「月份序」）
    role: FieldRole = "feature"
    derived: bool = False
    forbidden: bool = False
    reason: str | None = None
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    categories: tuple[str | int, ...] | None = None

    @model_validator(mode="after")
    def _sane(self) -> FieldSpec:
        if self.forbidden and not self.reason:
            raise ValueError(f"禁用列 {self.name} 必须写 reason：答辩要能一句话解释为什么禁用")
        if self.type in ("numeric", "time"):
            if self.min is None or self.max is None:
                raise ValueError(f"{self.name} 是 {self.type}，必须写 min 与 max")
            if self.min > self.max:
                raise ValueError(f"{self.name} 的 min 不能大于 max")
        if self.type in ("binary", "categorical", "ordinal"):
            if not self.categories:
                raise ValueError(f"{self.name} 是 {self.type}，必须列全 categories")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError(f"{self.name} 的 categories 有重复值")
        if self.type == "binary" and len(self.categories or ()) != 2:
            raise ValueError(f"{self.name} 是 binary，categories 必须正好两个")
        return self


class CatalogSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    fields: tuple[FieldSpec, ...]

    @model_validator(mode="after")
    def _sane(self) -> CatalogSpec:
        if self.schema_version != 1:
            raise ValueError(f"未知的特征目录版本 {self.schema_version}")
        names = [f.name for f in self.fields]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"字段名重复：{sorted(dup)}")
        missing = [n for n in REQUIRED_FORBIDDEN if n not in names]
        if missing:
            raise ValueError(f"禁用列必须出现在目录里：{missing}")
        for n in REQUIRED_FORBIDDEN:
            spec = next(f for f in self.fields if f.name == n)
            if not spec.forbidden:
                raise ValueError(f"{n} 必须标 forbidden: true")
        return self

    def by_name(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields}

    def field(self, name: str) -> FieldSpec:
        try:
            return self.by_name()[name]
        except KeyError:
            raise CatalogError(f"特征目录里没有字段 {name}") from None

    def feature_names(self) -> tuple[str, ...]:
        """可以进模型的列：非禁用、角色是 feature、不是时间轴/标签。"""
        skip = {"id", "date"}
        return tuple(
            f.name
            for f in self.fields
            if not f.forbidden and f.role == "feature" and f.type not in skip
        )

    def forbidden_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.forbidden)

    def fingerprint(self) -> str:
        return repro.sha256_json(self.model_dump(mode="json"))


def load_catalog(path: Path = CATALOG_PATH) -> CatalogSpec:
    try:
        raw = load_mapping(path, what="特征目录")
    except YamlError as exc:
        raise CatalogError(str(exc)) from exc
    try:
        return CatalogSpec.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"  {'.'.join(str(p) for p in e['loc']) or '<根>'}: {e['msg']}" for e in exc.errors()
        ]
        raise CatalogError("特征目录校验失败：\n" + "\n".join(lines)) from exc


def _feature_names_a() -> tuple[str, ...]:
    """惰性导入：目录是"规格层"，不该在 import 时把 pandas 数据层拖进来。"""
    from bank.dataset import model_features

    return tuple(model_features("A"))


def assert_agrees_with_schema(spec: CatalogSpec) -> None:
    """目录是唯一出处，`schema.py` 是转换实现——两边必须说同一件事。"""
    by = spec.by_name()
    problems: list[str] = []

    for col in S.NUMERIC:
        if col not in by:
            problems.append(f"schema 的数值列 {col} 不在目录里")
        elif by[col].forbidden:
            problems.append(f"{col} 在 schema 里是特征，目录却标了禁用")

    for col in S.CATEGORICAL:
        if col not in by:
            problems.append(f"schema 的类别列 {col} 不在目录里")
        elif by[col].type not in ("binary", "categorical"):
            problems.append(f"{col} 在目录里类型是 {by[col].type}，schema 当类别列用")

    for col in S.EXCLUDE_COLS:
        if col not in by or not by[col].forbidden:
            problems.append(f"schema 禁用了 {col}，目录必须标 forbidden")

    features = set(_feature_names_a())
    if set(spec.feature_names()) != features:
        missing = sorted(features - set(spec.feature_names()))
        extra = sorted(set(spec.feature_names()) - features)
        if missing:
            problems.append(f"特征集 A 用到但目录没标可用的列：{missing}")
        if extra:
            problems.append(f"目录说可用但特征集 A 没用到的列：{extra}")

    if problems:
        raise CatalogError("特征目录与 schema 不一致：\n" + "\n".join(f"  - {p}" for p in problems))
