"""一条命令训练、评估、记录：

    python -m bank_training.train

干四件事：
1. 从 `data/frozen/` 读训练集/验证集（不再碰原始库）；
2. 用同一份 `configs/features.yaml` 装配 Pipeline，跑规则基线、逻辑回归 A/B、树模型对照；
3. 按名额（验证集 10%，356 人）算指标并打印一张表——所有数字都来自这次运行；
4. 每个模型一个 mlflow run（参数、数据指纹、指标、产物），run_id 写进 `runs.json`。

验证集只用来报数，不回头改参数；要改就回训练集重训，再跑一遍这条命令。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import load_catalog
from bank.config import load_config
from bank.repro import git_sha

from bank_training.baselines import WEIGHTS, rule_score
from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate, threshold_metrics
from bank_training.model_config import load_model_config
from bank_training.pipeline import MODEL_KINDS, build_model

EXPERIMENT = "bank-marketing"
TRACKING_URI = f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}"
OUT_DIR = S.ROOT / "data" / "processed" / "runs"
RUNS_JSON = S.ROOT / "data" / "processed" / "runs.json"


def _write_metrics(name: str, metrics: dict[str, float], extra: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = {"metrics": metrics, **extra}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _log_run(
    *,
    name: str,
    params: dict[str, Any],
    tags: dict[str, Any],
    y_valid: pd.Series,
    score: np.ndarray,
    quota_n: int,
    artifacts: dict[str, str],
) -> tuple[str, dict[str, float]]:
    metrics = evaluate(y_valid, score, quota_n)
    metrics.update({f"t05_{k}": v for k, v in threshold_metrics(y_valid, score, 0.5).items()})
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**tags, "model_name": name})
        mlflow.log_params({k: _param(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        metrics_path = _write_metrics(name, metrics, {"params": params, "tags": tags})
        mlflow.log_artifact(str(metrics_path))
        for label, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=label)
        return run.info.run_id, metrics


def _param(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> int:
    cfg = load_config()
    mcfg = load_model_config()
    spec = load_catalog()
    train, valid, manifest = load_frozen()
    quota_n = int(manifest["capacity"]["quota_n"])
    feature_set = cfg.features.set

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    tags = {
        "data_fingerprint": manifest["tables"]["train"]["content_sha256"][:16],
        "config_fingerprint": manifest["config"]["sha256"][:16],
        "window": f"{manifest['window']['start']}~{manifest['window']['end']}",
        "cut": manifest["window"]["cut"],
        "git_sha": str(git_sha() or "dirty"),
    }
    base_params = {
        "seed": cfg.seed,
        "quota_frac": cfg.quota.frac,
        "quota_n": quota_n,
        "n_train": len(train),
        "n_valid": len(valid),
        "positives_train": int(train[S.TARGET].sum()),
        "positives_valid": int(valid[S.TARGET].sum()),
    }

    rows: list[dict[str, Any]] = []
    run_ids: dict[str, str] = {}

    score = rule_score(valid).to_numpy(dtype="float64")
    run_id, metrics = _log_run(
        name="m3-rule-baseline",
        params={**base_params, "model": "rule", "weights": WEIGHTS},
        tags=tags,
        y_valid=valid[S.TARGET],
        score=score,
        quota_n=quota_n,
        artifacts={},
    )
    run_ids["m3-rule-baseline"] = run_id
    rows.append({"name": "规则基线", **metrics})

    for kind, name, subset in (
        (mcfg.model.kind, f"m3-{mcfg.model.kind}-主模型", feature_set),
        ("lr", "m3-lr-A", "A"),
        ("lr", "m3-lr-B", "B"),
        ("rf", "m3-rf-A", "A"),
    ):
        if kind not in MODEL_KINDS:
            raise AssertionError(kind)
        x_train, y_train = split_xy(train, subset)
        x_valid, y_valid = split_xy(valid, subset)
        model = build_model(
            kind, spec, list(x_train.columns), c=mcfg.model.c, max_iter=mcfg.model.max_iter
        )
        model.fit(x_train, y_train)
        score = model.predict_proba(x_valid)[:, 1]
        model_path = OUT_DIR / f"{name}.joblib"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        run_id, metrics = _log_run(
            name=name,
            params={
                **base_params,
                "model": kind,
                "feature_set": subset,
                "n_features": len(x_train.columns),
                "c": mcfg.model.c,
                "max_iter": mcfg.model.max_iter,
            },
            tags=tags,
            y_valid=y_valid,
            score=score,
            quota_n=quota_n,
            artifacts={"model": str(model_path)},
        )
        run_ids[name] = run_id
        label = "主模型 LR-B" if name.endswith("主模型") else f"{kind.upper()}-{subset}"
        rows.append({"name": label, **metrics})

    RUNS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUNS_JSON.write_text(
        json.dumps({"quota_n": quota_n, "runs": run_ids}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    header = (
        f"{'模型':<10}{'AP':>8}{'ROC-AUC':>10}{'名额内精确':>12}"
        f"{'名额内召回':>12}{'Brier':>9}{'名额内认购':>11}"
    )
    print(
        f"验证集 {len(valid)} 人，正例 {int(valid[S.TARGET].sum())} 人；"
        f"名额 {quota_n} 人（{cfg.quota.frac:.0%}）"
    )
    print(header)
    for row in rows:
        tp = round(row["precision_at_quota"] * quota_n)
        print(
            f"{row['name']:<10}{row['ap']:>8.4f}{row['roc_auc']:>10.4f}"
            f"{row['precision_at_quota']:>12.4f}{row['recall_at_quota']:>12.4f}"
            f"{row['brier']:>9.4f}{tp:>11d}"
        )
    print(f"mlflow：{TRACKING_URI}（实验 {EXPERIMENT}，{len(run_ids)} 个 run）")
    print(f"run_id 已写入 {RUNS_JSON.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
