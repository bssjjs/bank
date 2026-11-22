"""漂移巡检：`python -m bank_training.drift`。

基准 = 训练集（冻结产物），对照 = 「线上」批次。默认对照是验证集（它的时间在训练之后，
正例率 50% 对训练期的 13%，是一次真实的分布平移）；也可以 `--current` 传一个 parquet。

只看**模型包里的 15 个原始输入列**，不看派生列：派生列由原始列唯一决定，重复报一遍
没有信息量，反而让人以为漂移来源有两个。分数单独一项（`score`），正例率单独一项。

PSI 的分箱边界固定取自基准集，缺失单独一箱——这两条写在 `bank.drift` 的模块说明里。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from bank import catalog
from bank import drift as D
from bank import schema as S
from bank.bundle import SERVING_DIR, load_bundle

from bank_training.data import load_frozen
from bank_training.model_config import load_model_config
from bank_training.pipeline import build_model

OUT_JSON = S.ROOT / "data" / "processed" / "drift.json"
EXPERIMENT = "bank-marketing"
TRACKING_URI = f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}"
NUMERIC = ("numeric", "time", "date", "id")


def _current(path: Path | None, valid: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if path is None:
        return valid, "valid.parquet（验证集，时间在训练之后）"
    return pd.read_parquet(path), str(path)


def _score_drift(train: pd.DataFrame, current: pd.DataFrame) -> D.Drift:
    """同一个模型喂两个时期的特征：看分数分布整体有没有平移。"""
    params = load_model_config().model
    spec = catalog.load_catalog()
    features = [entry.name for entry in load_bundle(SERVING_DIR).doc.features]
    pipe = build_model(params.kind, spec, features, c=params.c, max_iter=params.max_iter)
    pipe.fit(train[features], train["label"].astype("int8"))
    return D.score_psi(
        pipe.predict_proba(train[features])[:, 1],
        pipe.predict_proba(current[features])[:, 1],
    )


def run(current_path: Path | None = None) -> dict[str, Any]:
    train, valid, manifest = load_frozen()
    current, current_label = _current(current_path, valid)
    bundle = load_bundle(SERVING_DIR)
    columns = [entry.name for entry in bundle.doc.inputs]
    numeric = {entry.name for entry in bundle.doc.inputs if entry.type in NUMERIC}

    rows = [item.__dict__ for item in D.report(train, current, columns, numeric_columns=numeric)]
    score = _score_drift(train, current)

    ref_rate = float(train["label"].mean())
    cur_rate = float(current["label"].mean()) if "label" in current.columns else float("nan")
    result: dict[str, Any] = {
        "contract": "drift_report.v1",
        "model_version": bundle.version,
        "reference": "data/frozen/train.parquet",
        "current": current_label,
        "reference_rows": len(train),
        "current_rows": len(current),
        "reference_fingerprint": manifest["tables"]["train"]["content_sha256"][:16],
        "features": rows,
        "score": score.__dict__,
        "label": {
            "reference_rate": round(ref_rate, 6),
            "current_rate": None if pd.isna(cur_rate) else round(cur_rate, 6),
            "relative_change_pct": None
            if pd.isna(cur_rate)
            else round(D.label_rate_shift(ref_rate, cur_rate), 2),
        },
        "thresholds": {"stable": D.STABLE, "watch": D.WATCH, "bins": D.BINS},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def _log(result: dict[str, Any]) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="m7-drift") as run_obj:
        mlflow.log_params(
            {
                "reference": result["reference"],
                "current": result["current"],
                "bins": result["thresholds"]["bins"],
                "reference_fingerprint": result["reference_fingerprint"],
                "model_version": result["model_version"],
            }
        )
        metrics = {
            f"psi_{item['feature']}": round(float(item["psi"]), 6) for item in result["features"]
        }
        metrics["psi_score"] = round(float(result["score"]["psi"]), 6)
        metrics["label_reference_rate"] = float(result["label"]["reference_rate"])
        if result["label"]["current_rate"] is not None:
            metrics["label_current_rate"] = float(result["label"]["current_rate"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(OUT_JSON))
        return str(run_obj.info.run_id)


def summarize(result: dict[str, Any], run_id: str | None = None) -> None:
    print("漂移巡检完成")
    severe = [item["feature"] for item in result["features"] if item["level"] == "严重"]
    print(
        f"基准：{result['reference']}（{result['reference_rows']} 行，"
        f"指纹 {result['reference_fingerprint']}…）"
    )
    print(f"对照：{result['current']}（{result['current_rows']} 行）")
    print(
        f"模型包：{result['model_version']}；分箱 {result['thresholds']['bins']} 个等频箱，"
        f"边界取自基准集，缺失单独一箱"
    )
    print("")
    print(f"{'特征':<22}{'类型':<12}{'PSI':>8}{'判定':>8}")
    for item in result["features"]:
        print(f"{item['feature']:<22}{item['kind']:<12}{item['psi']:>8.4f}{item['level']:>8}")
    print(f"{'score':<22}{'score':<12}{result['score']['psi']:>8.4f}{result['score']['level']:>8}")
    label = result["label"]
    change = label["relative_change_pct"]
    suffix = "（无标签）" if change is None else f"（变化 {change:+.2f}%）"
    print("")
    print(f"正例率：基准 {label['reference_rate']:.4f}，对照 {label['current_rate']}{suffix}")
    print(f"严重漂移的特征 {len(severe)} 个：{'、'.join(severe) if severe else '无'}")
    if run_id:
        print(f"mlflow run：{run_id}（实验 {EXPERIMENT}）")
    print(f"报告：{OUT_JSON.relative_to(S.ROOT).as_posix()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对「线上」批次跑 PSI 漂移巡检")
    parser.add_argument("--current", type=Path, default=None, help="对照数据 parquet，默认验证集")
    parser.add_argument("--no-mlflow", action="store_true", help="只打印，不记 mlflow")
    args = parser.parse_args(argv)
    result = run(args.current)
    run_id = None if args.no_mlflow else _log(result)
    summarize(result, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
