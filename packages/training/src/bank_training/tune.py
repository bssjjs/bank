"""C-L3b 调参：只在训练期里切内层时间窗，验证集一次都不看。

为什么单独做一条命令：调参最容易被"顺手拿验证集试一下"污染，那样报出来的指标就不再是
"没见过的数据"上的成绩（任务书一票否决项）。这里把纪律写进代码：

1. 训练期是 `重建日期 < configs/base.yaml: split.cut`（2009-10-01）那 17,537 行；
2. 再在内层按 `INNER_CUT = 2009-04-01` 切成"拟合段"和"评估段"，两段都在训练期里；
3. 只在这两段上跑网格（C 的四个值），按内层 AP 挑一个，打印表格；
4. 断言评估段最后一天 **早于** `split.cut`——验证集一行都没进来，代码里挡住，不靠自觉。

推荐值不自动写进 `configs/model.yaml`：改超参数是人的决定，改完要重跑 train 与 export，
让新数字重新出现在报告、模型包与封条里。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from bank import catalog
from bank import config as cfg_mod
from bank import schema as S
from bank import timeaxis as T

from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate
from bank_training.model_config import load_model_config
from bank_training.pipeline import build_model

CONTRACT = "tune_report.v1"
INNER_CUT = "2009-04-01"
GRID = (0.3, 1.0, 3.0, 10.0)
NOISE = 0.01  # 内层 AP 差在这个量级以内算噪声，别为它改配置
OUT_PATH = S.ROOT / "data" / "processed" / "tune.json"
EXPERIMENT = "bank-marketing"


def inner_split(train: pd.DataFrame, cfg: cfg_mod.Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把训练期再切成拟合段与评估段；越界直接抛错，不静默裁。"""
    cut = pd.Timestamp(cfg.split.cut)
    inner = pd.Timestamp(INNER_CUT)
    if not inner < cut:
        raise ValueError("INNER_CUT 必须早于 split.cut")
    dates = pd.to_datetime(train[T.DATE])
    fit = train[dates < inner]
    holdout = train[(dates >= inner) & (dates < cut)]
    if fit.empty or holdout.empty:
        raise ValueError("内层切分后有一段是空的")
    if pd.to_datetime(holdout[T.DATE]).max() >= cut:
        raise ValueError("内层评估段越过了切点，验证集漏进来了")
    return fit, holdout


def grid(
    fit: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    feature_set: str,
    spec: catalog.CatalogSpec,
    quota_frac: float,
    max_iter: int,
) -> list[dict[str, Any]]:
    x_fit, y_fit = split_xy(fit, feature_set)
    x_hold, y_hold = split_xy(holdout, feature_set)
    quota_n = max(1, round(len(y_hold) * quota_frac))
    rows: list[dict[str, Any]] = []
    for c in GRID:
        model = build_model("lr", spec, list(x_fit.columns), c=c, max_iter=max_iter)
        model.fit(x_fit, y_fit)
        score = model.predict_proba(x_hold)[:, 1]
        metrics = evaluate(y_hold.to_numpy(), score, quota_n)
        rows.append(
            {
                "c": c,
                "ap": round(float(metrics["ap"]), 6),
                "roc_auc": round(float(metrics["roc_auc"]), 6),
                "precision_at_quota": round(float(metrics["precision_at_quota"]), 6),
                "n_fit": len(fit),
                "n_holdout": len(holdout),
                "quota_n": quota_n,
            }
        )
    return rows


def _span(frame: pd.DataFrame) -> str:
    dates = pd.to_datetime(frame[T.DATE])
    return f"{dates.min().date()} ~ {dates.max().date()}"


def best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("网格是空的")
    return max(rows, key=lambda row: float(row["ap"]))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _log(rows: list[dict[str, Any]], chosen: dict[str, Any], fit_rows: int) -> list[str]:
    try:
        import mlflow
    except ImportError:
        return []
    mlflow.set_tracking_uri(f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    ids: list[str] = []
    for row in rows:
        with mlflow.start_run(run_name=f"c-l3b-tune-c{row['c']}") as run:
            mlflow.log_params(
                {"c": row["c"], "inner_cut": INNER_CUT, "n_fit": fit_rows, "contract": CONTRACT}
            )
            mlflow.log_metrics(
                {
                    "ap": float(row["ap"]),
                    "roc_auc": float(row["roc_auc"]),
                    "precision_at_quota": float(row["precision_at_quota"]),
                }
            )
            mlflow.set_tag("stage", "tune")
            ids.append(run.info.run_id)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-L3b 调参：只用训练期里的内层时间窗")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    cfg = cfg_mod.load_config()
    params = load_model_config().model
    spec = catalog.load_catalog()
    train, _valid, _manifest = load_frozen()
    fit, holdout = inner_split(train, cfg)

    fit_span = _span(fit)
    hold_span = _span(holdout)
    print(f"调参窗口（只用训练期）：拟合 {len(fit):,} 行（{fit_span}）")
    print(f"　　　　　　　　　　　　内层评估 {len(holdout):,} 行（{hold_span}）")
    print("验证集（≥ split.cut）一次都没读。")

    rows = grid(
        fit,
        holdout,
        feature_set=cfg.features.set,
        spec=spec,
        quota_frac=cfg.quota.frac,
        max_iter=params.max_iter,
    )
    chosen = best(rows)
    print(f"{'C':>5} {'AP':>8} {'ROC-AUC':>8} {'名额内精确':>10}")
    for row in rows:
        mark = " ←选中" if row is chosen else ""
        print(
            f"{row['c']:>5} {row['ap']:>8.4f} {row['roc_auc']:>8.4f}"
            f" {row['precision_at_quota']:>10.4f}{mark}"
        )

    spread = max(float(row["ap"]) for row in rows) - min(float(row["ap"]) for row in rows)
    note = (
        "四个 C 的内层 AP 差在噪声量级内（{spread:.4f} < {noise}），不必为它改配置；"
        "若要改，先说明依据，再重跑 train 与 export。"
        if spread < NOISE
        else "内层 AP 有明显差异（{spread:.4f}），建议采用选中值，改配置后重跑 train 与 export。"
    ).format(spread=spread, noise=NOISE)

    ids = [] if args.no_mlflow else _log(rows, chosen, len(fit))
    _write_json(
        Path(args.out),
        {
            "contract": CONTRACT,
            "inner_cut": INNER_CUT,
            "split_cut": str(cfg.split.cut),
            "feature_set": cfg.features.set,
            "grid": rows,
            "chosen": chosen,
            "spread": round(spread, 6),
            "advice": note,
            "mlflow_run_ids": ids,
        },
    )
    print(f"推荐 C={chosen['c']}（内层 AP {chosen['ap']:.4f}）")
    print(note)
    if ids:
        print(f"mlflow runs：{len(ids)} 个（c-l3b-tune-*）")
    print(f"写出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
