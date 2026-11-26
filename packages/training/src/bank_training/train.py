"""一条命令训练、评估、记录：

    python -m bank_training.train

干四件事：
1. 从 `data/frozen/` 读训练集/验证集（不再碰原始库）；
2. 用同一份 `configs/features.yaml` 装配 Pipeline，跑规则基线、逻辑回归 A/B、树模型对照；
3. 按名额（验证集 10%，356 人）算指标并打印一张表——所有数字都来自这次运行；
4. 每个模型一个追踪 run（参数、数据指纹、指标、产物），run_id 写进 `runs.json`；后端默认 mlflow，
   由 `configs/tracking.yaml` 决定，测试里换成内存后端就不需要 mlflow。

验证集只用来报数，不回头改参数；要改就回训练集重训，再跑一遍这条命令。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import load_catalog
from bank.config import load_config
from bank.repro import git_sha

from bank_training import global_explain
from bank_training import metrics as metrics_mod
from bank_training import report as R
from bank_training.baselines import WEIGHTS, rule_score
from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate, threshold_metrics
from bank_training.model_config import load_model_config
from bank_training.pipeline import MODEL_KINDS, build_model
from bank_training.tracking import Tracker, get_tracker

OUT_DIR = S.ROOT / "data" / "processed" / "runs"
RUNS_JSON = S.ROOT / "data" / "processed" / "runs.json"
REPORTS_DIR = S.ROOT / "reports"


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
    tracker: Tracker,
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
    with tracker.start_run(name) as run_id:
        tracker.set_tags({**tags, "model_name": name})
        tracker.log_params(params)
        tracker.log_metrics({k: float(v) for k, v in metrics.items()})
        metrics_path = _write_metrics(name, metrics, {"params": params, "tags": tags})
        tracker.log_artifact(metrics_path)
        for label, path in artifacts.items():
            tracker.log_artifact(Path(path), artifact_path=label)
        return run_id, metrics


def _main_report(
    *,
    tracker: Tracker,
    main_name: str,
    label: str,
    y_valid: pd.Series,
    score: np.ndarray,
    quota_n: int,
    rows: list[dict[str, Any]],
    run_id: str,
    feature_set: str,
    quota_frac: float,
    frame: pd.DataFrame,
    extra_md: str = "",
) -> tuple[Path, dict[str, tuple[float, float]]]:
    """给主模型出一份 Markdown 评估报告：13 指标 + 区间 + 两种口径 + 校准 + 切片 + 对照。"""
    comparisons = [
        R.comparison(row["name"], {k: v for k, v in row.items() if k != "name"})
        for row in rows
        if row["name"] != label
    ]
    intervals = metrics_mod.bootstrap_ci(y_valid, score, quota_n)
    note_quota = f"名额 {quota_n} 人 = 验证集 {len(y_valid):,} 人 × {quota_frac:.0%}（base.yaml）"
    note_drift = (
        "验证集是 2009-10 以后的数据、正例率约 50%，训练期只有 13%——分数系统性偏低、"
        "Brier 与 ECE 会难看，这是时期漂移不是 bug（ADR 0005 与模型卡）。"
    )
    note_ci = (
        "95% 区间是 200 次按行重抽的分位区间，名额按比例跟着行一起缩放；"
        "它只说明这批数据的抖动幅度，不是上线后的保证。"
    )
    note_views = "两种口径选中人数不同：名额口径固定 356 人，阈值 0.5 口径随分数分布变。"
    note_slices = (
        "切片里的名额按该切片人数比例分配（k = 人数 × 名额比例，四舍五入），"
        "所以人数少的切片名额也少、数字更抖；少于 100 人的切片不报。"
    )
    built = R.build(
        title=f"评估报告：{main_name}（特征集 {feature_set}）",
        y_true=y_valid,
        score=score,
        quota_n=quota_n,
        comparisons=comparisons,
        intervals=intervals,
        notes=(note_quota, note_drift, note_ci, note_views, note_slices),
        frame=frame,
    )
    path = R.write(REPORTS_DIR / "evaluation_report.md", built)
    if extra_md:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n\n" + extra_md.lstrip())
    json_path = R.write_json(REPORTS_DIR / "evaluation_report.json", built)
    tracker.log_artifact(path, artifact_path="evaluation", run_id=run_id)
    tracker.log_artifact(json_path, artifact_path="evaluation", run_id=run_id)
    return path, intervals


def main() -> int:
    cfg = load_config()
    mcfg = load_model_config()
    spec = load_catalog()
    train, valid, manifest = load_frozen()
    quota_n = int(manifest["capacity"]["quota_n"])
    feature_set = cfg.features.set

    tracker = get_tracker()

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
    main_name = f"m3-{mcfg.model.kind}-主模型"
    main_y: pd.Series | None = None
    main_score: np.ndarray | None = None
    main_model: Any | None = None
    main_x_valid: pd.DataFrame | None = None

    score = rule_score(valid).to_numpy(dtype="float64")
    run_id, metrics = _log_run(
        tracker=tracker,
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
            tracker=tracker,
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
        if name == main_name:
            main_y = y_valid
            main_score = np.asarray(score, dtype="float64")
            main_model = model
            main_x_valid = x_valid
        label = "主模型 LR-B" if name == main_name else f"{kind.upper()}-{subset}"
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
    print(f"追踪：{tracker.name}，{len(run_ids)} 个 run（实验 bank-marketing）")

    if main_y is None or main_score is None or main_model is None or main_x_valid is None:
        raise AssertionError("主模型没有被训练：检查 configs/model.yaml 的 kind")

    # 全局解释先算出来，作为评估报告的一节；同一份 md 也作为独立产物挂到主模型 run。
    global_md, global_png = global_explain.run_for(
        model=main_model, x=main_x_valid, y=main_y, quota_n=quota_n
    )
    global_section = global_md.read_text(encoding="utf-8")

    report_path, intervals = _main_report(
        tracker=tracker,
        main_name=main_name,
        label="主模型 LR-B",
        y_valid=main_y,
        score=main_score,
        quota_n=quota_n,
        rows=rows,
        run_id=run_ids[main_name],
        feature_set=feature_set,
        quota_frac=cfg.quota.frac,
        frame=valid,
        extra_md=global_section,
    )
    print("主模型 95% 区间（200 次按行重抽）：")
    for name in ("precision_at_quota", "recall_at_quota", "ap", "roc_auc", "brier"):
        print(f"  {name:<20}{metrics_mod.format_ci(intervals[name])}")
    print(f"评估报告：{report_path.relative_to(S.ROOT)}（已记进追踪 run）")

    tracker.log_artifact(global_md, artifact_path="evaluation", run_id=run_ids[main_name])
    if global_png is not None:
        tracker.log_artifact(global_png, artifact_path="evaluation", run_id=run_ids[main_name])

    print(f"run_id 已写入 {RUNS_JSON.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
