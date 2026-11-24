"""导出模型包：`python -m bank_training.export --version 1.0.0`

做五件事：
1. 读冻结集与 `configs/model.yaml`，用主口径（特征集、模型类型、C）重训一次；
2. 在验证集上算指标（只报数，不回头调参）；
3. 用验证集分数冻结两个等级切点：「优先」= 名额内前 1/3 的分数下限，「备选」= 名额边界；
4. 过发布门禁（指标下限 + 包自检），任何一条不过就拒绝发布；
5. 写 `models/serving/<版本>/`，再把 `current.json` 指向它——不覆盖旧版本。

模型包本身由 `bank.bundle` 写与读；服务端只认 `bundle.json` 的校验和与列清单，
不导入训练代码。
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
from bank import bundle as B
from bank import registry as REG
from bank import schema as S
from bank.catalog import load_catalog
from bank.config import load_config
from bank.repro import git_sha

from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate, threshold_metrics
from bank_training.model_config import load_model_config
from bank_training.pipeline import build_model

EXPERIMENT = "bank-marketing"
TRACKING_URI = f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}"
MIN_FEATURES = 10
GATE_AP = 0.50
GATE_PRECISION = 0.60


SERVING_EXCLUDED = (S.DURATION, S.TARGET_RAW)


def _inputs(spec: Any) -> list[dict[str, Any]]:
    """要用户填的原始列：非派生，且不是「通话后才知道」的时长与目标列。

    `距上次联系天数` 虽然在特征目录里标了 forbidden（-1 是哨兵，原列不入模），
    但模型算 `是否被联系过` / `_log` / `档` 都要它，所以它必须是必填输入——
    forbidden 约束的是「能不能原样喂给模型」，不是「能不能出现在表单上」。
    """
    return [
        {
            "name": field.name,
            "type": field.type,
            "display": field.display,
            "unit": field.unit,
            "min": field.min,
            "max": field.max,
            "categories": list(field.categories) if field.categories else None,
        }
        for field in spec.fields
        if not field.derived and field.name not in SERVING_EXCLUDED
    ]


def _level_cutoffs(scores: np.ndarray, quota_n: int) -> dict[str, float]:
    """把「名额内前 1/3」与「名额边界」两个分数冻进包里，单条请求据此判等级。"""
    ranked = np.sort(np.asarray(scores, dtype="float64"))[::-1]
    if len(ranked) < quota_n:
        raise ValueError(f"验证集 {len(ranked)} 人不够名额 {quota_n} 人")
    third = math.ceil(quota_n / 3)
    return {
        "优先": round(float(ranked[third - 1]), 6),
        "备选": round(float(ranked[quota_n - 1]), 6),
    }


def _gate(metrics: dict[str, float], n_features: int) -> list[str]:
    """发布门禁：指标达不到下限就不许上线。阈值写在代码里，不藏。"""
    failures: list[str] = []
    if n_features < MIN_FEATURES:
        failures.append(f"特征数 {n_features} < {MIN_FEATURES}")
    if metrics["ap"] < GATE_AP:
        failures.append(f"AP {metrics['ap']:.4f} < {GATE_AP}")
    if metrics["precision_at_quota"] < GATE_PRECISION:
        failures.append(f"名额内精确率 {metrics['precision_at_quota']:.4f} < {GATE_PRECISION}")
    if not metrics["roc_auc"] > 0.5:
        failures.append(f"ROC-AUC {metrics['roc_auc']:.4f} 不高于 0.5，排序无效")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 ModelBundle")
    parser.add_argument("--version", default="1.0.0", help="版本号 x.y.z，默认 1.0.0")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="登记进注册表后立刻发布（线上指针指向它）；不加只登记为 candidate",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    mcfg = load_model_config()
    spec = load_catalog()
    train, valid, manifest = load_frozen()

    feature_names = list(split_xy(train, cfg.features.set)[0].columns)
    x_train, y_train = split_xy(train, cfg.features.set)
    x_valid, y_valid = split_xy(valid, cfg.features.set)
    quota_n = int(manifest["capacity"]["quota_n"])

    model = build_model(
        mcfg.model.kind, spec, feature_names, c=mcfg.model.c, max_iter=mcfg.model.max_iter
    )
    model.fit(x_train, y_train)
    scores = np.asarray(model.predict_proba(x_valid)[:, 1], dtype="float64")

    metrics = evaluate(y_valid, scores, quota_n)
    metrics.update({f"t05_{k}": v for k, v in threshold_metrics(y_valid, scores, 0.5).items()})
    cutoffs = _level_cutoffs(scores, quota_n)

    failures = _gate(metrics, len(feature_names))
    if failures:
        for item in failures:
            print(f"[门禁未通过] {item}")
        return 1

    inputs = _inputs(spec)
    decision = {
        "mode": "quota",
        "quota_frac": cfg.quota.frac,
        "quota_n": quota_n,
        "level_cutoffs": cutoffs,
        "threshold": 0.5,
        "explain_top_k": 5,
    }
    training = {
        "run_id": "pending",
        "experiment": EXPERIMENT,
        "data_fingerprint_train": manifest["tables"]["train"]["content_sha256"],
        "data_fingerprint_valid": manifest["tables"]["valid"]["content_sha256"],
        "config_fingerprint": manifest["config"]["sha256"],
        "git_sha": str(git_sha() or "dirty"),
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "n_features": len(feature_names),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"m4-bundle-{args.version}") as run:
        run_id = run.info.run_id
        training["run_id"] = run_id
        mlflow.set_tags(
            {
                "stage": "bundle",
                "version": args.version,
                "feature_set": cfg.features.set,
                "data_fingerprint": training["data_fingerprint_train"][:16],
                "config_fingerprint": training["config_fingerprint"][:16],
            }
        )
        mlflow.log_params(
            {
                "version": args.version,
                "model": mcfg.model.kind,
                "feature_set": cfg.features.set,
                "n_features": len(feature_names),
                "c": mcfg.model.c,
                "max_iter": mcfg.model.max_iter,
                "quota_frac": cfg.quota.frac,
                "quota_n": quota_n,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        path = B.write_bundle(
            version=args.version,
            model=model,
            feature_set=cfg.features.set,
            feature_names=feature_names,
            feature_types={
                field.name: str(field.type)
                for field in spec.fields
                if field.name in set(feature_names)
            },
            inputs=inputs,
            catalog_fingerprint=spec.fingerprint(),
            decision=decision,
            metrics=metrics,
            training=training,
            runtime=B.runtime_now(),
        )
        mlflow.log_artifact(str(path / B.BUNDLE_JSON), artifact_path="bundle")

    registered = REG.register(
        args.version,
        run_id=run_id,
        metrics=metrics,
        gate=REG.GateRecord(passed=True),
        note=f"模型包 {args.version}，特征集 {cfg.features.set}",
    )
    registered_state = registered["versions"][args.version]["status"]
    if args.promote:
        registered = REG.promote(
            args.version, actor="bank_training.export", reason="导出后立即发布（--promote）"
        )
    current = REG.production() or "（未发布）"

    loaded = B.load_bundle(path)
    sample = valid.head(1)
    sample_score = loaded.score(sample)
    assert 0.0 <= sample_score <= 1.0

    print(f"模型包：{path}")
    print(f"版本 {loaded.version}，契约 {loaded.doc.contract}，特征集 {loaded.doc.feature_set}")
    print(f"特征 {len(loaded.feature_names)} 列，必填输入 {len(loaded.doc.required_columns)} 列")
    print(
        f"验证集 {int(metrics['n'])} 人 / 正例 {int(metrics['positives'])} 人 / 名额 {quota_n} 人："
        f"AP {metrics['ap']:.4f}，ROC-AUC {metrics['roc_auc']:.4f}，"
        f"名额内精确率 {metrics['precision_at_quota']:.4f}，召回率 {metrics['recall_at_quota']:.4f}"
    )
    print(f"等级切点：优先 ≥ {cutoffs['优先']:.4f}，备选 ≥ {cutoffs['备选']:.4f}")
    print(f"回读自检：随机一条样本得分 {sample_score:.4f}，等级 {loaded.level(sample_score)}")
    print(f"README 用的 run_id：{run_id}")
    print(
        f"注册表：{args.version} 状态 {registered['versions'][args.version]['status']}"
        f"（登记时 {registered_state}），线上 {current}"
    )
    if not args.promote:
        print("要上线：python -m bank.registry promote " + args.version + ' --reason "..."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
