"""ModelBundle：训练与线上之间唯一的那份契约。

一个模型包是一个目录（`models/serving/<版本>/`），里面有：

- `model.joblib`   拟合好的 sklearn `Pipeline`（预处理 + 逻辑回归），pickle 里的类全部来自 `bank`
- `bundle.json`    输入列与顺序、决策规则、训练来源、指标快照、校验和

`bundle.json` 的 `checksum` 是「除 checksum 外全部字段的规范化 JSON」的 sha256，
`model_file.sha256` 是模型文件字节的 sha256。两个都对上才算加载成功——文件被改过
一个字节都过不去。

**不覆盖旧版本**：写包时目标目录已存在就直接报错。回退靠 `current.json` 指回旧版本，
不删目录、不改文件（ADR 0006）。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.pipeline import Pipeline

from bank import dataset, explain, features
from bank import schema as S
from bank.repro import sha256_file, sha256_json

CONTRACT = "model_bundle.v1"
BUNDLE_JSON = "bundle.json"
MODEL_FILE = "model.joblib"
CURRENT_FILE = "current.json"
SERVING_DIR = S.ROOT / "models" / "serving"
LEVELS = ("优先", "备选", "不建议")
_VERSION_PARTS = 3


class BundleError(RuntimeError):
    """模型包本身有问题：契约不符、校验和不一致、目录被重复写。"""


class MissingColumnsError(BundleError):
    """请求缺列。服务层据此回 422，并指出缺的是哪几列。"""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"缺少字段：{'、'.join(missing)}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFile(_Strict):
    name: str
    sha256: str = Field(min_length=64, max_length=64)
    size: int = Field(gt=0)


class FeatureEntry(_Strict):
    name: str
    type: str
    order: int = Field(gt=0)


class InputField(_Strict):
    name: str
    type: str
    display: str
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    categories: tuple[str, ...] | None = None


class DecisionRule(_Strict):
    """名额制是主口径；阈值只作对照。等级用验证集分数切点冻结，便于单条请求判定。"""

    mode: Literal["quota"]
    quota_frac: float = Field(gt=0, le=1)
    quota_n: int = Field(gt=0)
    level_cutoffs: dict[str, float]
    threshold: float
    explain_top_k: int = Field(ge=1, le=50)

    @model_validator(mode="after")
    def _levels(self) -> DecisionRule:
        if set(self.level_cutoffs) != {"优先", "备选"}:
            raise ValueError("level_cutoffs 必须正好有「优先」「备选」两个切点")
        if not self.level_cutoffs["优先"] >= self.level_cutoffs["备选"]:
            raise ValueError("「优先」切点不能低于「备选」切点")
        return self


class Training(_Strict):
    run_id: str
    experiment: str
    data_fingerprint_train: str
    data_fingerprint_valid: str
    config_fingerprint: str
    git_sha: str
    trained_at: str
    rows_train: int = Field(gt=0)
    rows_valid: int = Field(gt=0)
    n_features: int = Field(gt=0)


class Checksum(_Strict):
    algorithm: Literal["sha256"]
    value: str = Field(min_length=64, max_length=64)


class BundleDoc(_Strict):
    contract: Literal["model_bundle.v1"]
    version: str
    created_at: str
    model_file: ModelFile
    feature_set: str
    features: tuple[FeatureEntry, ...]
    inputs: tuple[InputField, ...]
    required_columns: tuple[str, ...]
    catalog_fingerprint: str
    decision: DecisionRule
    metrics: dict[str, float]
    training: Training
    checksum: Checksum

    @model_validator(mode="after")
    def _coherent(self) -> BundleDoc:
        parts = self.version.split(".")
        if len(parts) != _VERSION_PARTS or not all(p.isdigit() for p in parts):
            raise ValueError(f"版本号必须是 x.y.z 形式，收到 {self.version!r}")
        orders = [entry.order for entry in self.features]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError("features.order 必须是从 1 开始的连续整数")
        if tuple(entry.name for entry in self.inputs) != self.required_columns:
            raise ValueError("required_columns 必须与 inputs 的顺序完全一致")
        if self.training.n_features != len(self.features):
            raise ValueError("training.n_features 与 features 条数不一致")
        return self


class Bundle:
    """加载好的模型包：`score` / `explain` 是服务与训练共用的唯一推理路径。"""

    def __init__(self, doc: BundleDoc, model: Pipeline, path: Path) -> None:
        self.doc = doc
        self.model = model
        self.path = path

    @property
    def version(self) -> str:
        return self.doc.version

    @property
    def feature_names(self) -> list[str]:
        return [entry.name for entry in self.doc.features]

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        """校验必填列 → 派生（与服务同一条路径）→ 按模型的列顺序取特征。"""
        missing = [name for name in self.doc.required_columns if name not in raw.columns]
        if missing:
            raise MissingColumnsError(missing)
        derived = dataset.derive_features(raw)
        return features.select_features(derived, self.feature_names)

    def score_frame(self, raw: pd.DataFrame) -> np.ndarray:
        frame = self.prepare(raw)
        proba = self.model.predict_proba(frame)
        return np.asarray(proba[:, 1], dtype="float64")

    def score(self, raw: pd.DataFrame) -> float:
        return float(self.score_frame(raw)[0])

    def explain(self, raw: pd.DataFrame) -> list[dict[str, object]]:
        frame = self.prepare(raw)
        return explain.explain_one(self.model, frame, top_k=self.doc.decision.explain_top_k)

    def level(self, score: float) -> str:
        cutoffs = self.doc.decision.level_cutoffs
        if score >= cutoffs["优先"]:
            return "优先"
        if score >= cutoffs["备选"]:
            return "备选"
        return "不建议"

    def describe(self) -> dict[str, Any]:
        """给 `/v1/model` 用：前端靠它生成表单。"""
        return {
            "version": self.doc.version,
            "contract": self.doc.contract,
            "feature_set": self.doc.feature_set,
            "created_at": self.doc.created_at,
            "inputs": [field.model_dump() for field in self.doc.inputs],
            "decision": self.doc.decision.model_dump(),
            "metrics": self.doc.metrics,
            "training": self.doc.training.model_dump(),
            "catalog_fingerprint": self.doc.catalog_fingerprint,
        }


def _payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != "checksum"}


def build_doc(
    *,
    version: str,
    feature_set: str,
    feature_names: list[str],
    feature_types: dict[str, str],
    inputs: list[dict[str, Any]],
    catalog_fingerprint: str,
    decision: dict[str, Any],
    metrics: dict[str, float],
    training: dict[str, Any],
    model_file: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """拼出 bundle.json 的字典并算校验和（写文件与测试共用）。"""
    unknown = [name for name in feature_names if name not in feature_types]
    if unknown:
        raise BundleError(f"这些特征没有登记类型：{unknown}；检查 configs/features.yaml")
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "version": version,
        "created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_file": model_file,
        "feature_set": feature_set,
        "features": [
            {"name": name, "type": feature_types[name], "order": index}
            for index, name in enumerate(feature_names, start=1)
        ],
        "inputs": inputs,
        "required_columns": [item["name"] for item in inputs],
        "catalog_fingerprint": catalog_fingerprint,
        "decision": decision,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "training": training,
    }
    payload["checksum"] = {"algorithm": "sha256", "value": sha256_json(payload)}
    BundleDoc.model_validate(payload)  # 先自己过一遍，坏包不落盘
    return payload


def write_bundle(
    *,
    version: str,
    model: Pipeline,
    feature_set: str,
    feature_names: list[str],
    feature_types: dict[str, str],
    inputs: list[dict[str, Any]],
    catalog_fingerprint: str,
    decision: dict[str, Any],
    metrics: dict[str, float],
    training: dict[str, Any],
    root: Path = SERVING_DIR,
) -> Path:
    """写一个版本目录；同名目录已存在就报错，绝不覆盖。

    先写进同级的临时目录，自检通过后再改名成正式版本目录：中途失败不会留下
    半个包，也不会占住版本号。
    """
    target = root / version
    if target.exists():
        raise BundleError(f"版本 {version} 已存在：{target}，模型包不覆盖旧版本，换个版本号")
    staging = root / f".staging-{version}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        model_path = staging / MODEL_FILE
        joblib.dump(model, model_path)
        doc = build_doc(
            version=version,
            feature_set=feature_set,
            feature_names=feature_names,
            feature_types=feature_types,
            inputs=inputs,
            catalog_fingerprint=catalog_fingerprint,
            decision=decision,
            metrics=metrics,
            training=training,
            model_file={
                "name": MODEL_FILE,
                "sha256": sha256_file(model_path),
                "size": model_path.stat().st_size,
            },
        )
        (staging / BUNDLE_JSON).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        load_bundle(staging)  # 写出去的包必须自己读得回来
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def load_bundle(path: Path) -> Bundle:
    """读并校验一个版本目录。`path` 可以是版本目录，也可以是 `models/serving`。"""
    directory = path if (path / BUNDLE_JSON).exists() else path / read_current(path)
    if not directory.is_dir():
        raise BundleError(f"找不到模型包目录：{directory}")

    doc_path = directory / BUNDLE_JSON
    if not doc_path.exists():
        raise BundleError(f"缺少 {BUNDLE_JSON}：{doc_path}")
    raw = json.loads(doc_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BundleError(f"{doc_path} 不是一个 JSON 对象")

    checksum = raw.get("checksum")
    if not isinstance(checksum, dict) or checksum.get("value") != sha256_json(_payload(raw)):
        raise BundleError("bundle.json 校验和对不上：文件被改过，拒绝加载")

    try:
        doc = BundleDoc.model_validate(raw)
    except Exception as error:  # pydantic 的报错太长，压成一句话
        raise BundleError(f"bundle.json 不符合契约：{error}") from error

    model_path = directory / doc.model_file.name
    if not model_path.exists():
        raise BundleError(f"缺少模型文件：{model_path}")
    if model_path.stat().st_size != doc.model_file.size:
        raise BundleError("模型文件大小与 bundle.json 不一致")
    if sha256_file(model_path) != doc.model_file.sha256:
        raise BundleError("模型文件校验和对不上：文件被改过，拒绝加载")

    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise BundleError(f"模型文件里不是 sklearn Pipeline，而是 {type(model).__name__}")
    return Bundle(doc, model, directory)


def read_current(root: Path = SERVING_DIR) -> str:
    """读 `current.json`：当前线上用的是哪个版本。"""
    pointer = root / CURRENT_FILE
    if not pointer.exists():
        raise BundleError(f"缺少 {CURRENT_FILE}：{pointer}；先跑 python -m bank_training.export")
    data = json.loads(pointer.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str):
        raise BundleError(f"{CURRENT_FILE} 里没有 version")
    return version


def write_current(version: str, root: Path = SERVING_DIR) -> Path:
    """发布/回退就是改这一行指针；不删目录、不改包内文件。"""
    if not (root / version / BUNDLE_JSON).exists():
        raise BundleError(f"版本 {version} 没有 bundle.json，不能指向它")
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / CURRENT_FILE
    pointer.write_text(
        json.dumps({"version": version}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return pointer
