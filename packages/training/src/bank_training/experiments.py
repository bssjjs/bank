"""E1–E5 对照实验：一次只改一处，看指标怎么动。

    python -m bank_training.experiments

E1 泄漏对照：把 `通话时长` 加进特征
E2 特征集对照：A（含月份序/月份 sin/cos，24 列）vs B（21 列）
E3 「未知」当缺失：三列未知值填训练集众数
E4 哨兵对照：`距上次联系天数` 原列当数值，替掉派生三列
E5 切分对照：按时间切 vs 随机 80/20 切

数字回填 `docs/spec.md` 第 6 节；解释写进 `docs/adr/0005-模型选择.md`。
这些实验只用来解释"为什么这么定"，不反过来拿验证集调参。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from bank import schema as S
from bank.catalog import load_catalog
from bank.config import load_config

from bank_training.data import load_frozen, split_xy
from bank_training.metrics import evaluate
from bank_training.pipeline import build_model
from bank_training.tracking import get_tracker
from bank_training.train import OUT_DIR

E1 = "E1-通话时长泄漏"
E2 = "E2-特征集A对B"
E3 = "E3-未知填众数"
E4 = "E4-pdays原列"
E5 = "E5-随机切分"


def _log(name: str, params: dict[str, object], metrics: dict[str, float]) -> str:
    tracker = get_tracker()
    with tracker.start_run(name) as run_id:
        tracker.set_tags({"experiment": name})
        tracker.log_params(
            {
                k: v
                if isinstance(v, (str, int, float, bool))
                else json.dumps(v, ensure_ascii=False)
                for k, v in params.items()
            }
        )
        tracker.log_metrics({k: float(v) for k, v in metrics.items()})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"exp-{name}.json"
        path.write_text(
            json.dumps(
                {"params": params, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        tracker.log_artifact(str(path))
        return run_id


def _fit_score(
    frame_train: pd.DataFrame, frame_valid: pd.DataFrame, columns: list[str], *, kind: str = "lr"
) -> np.ndarray:
    model = build_model(kind, load_catalog(), columns)
    model.fit(frame_train[columns], frame_train[S.TARGET])
    return np.asarray(model.predict_proba(frame_valid[columns])[:, 1], dtype="float64")


def main() -> int:
    cfg = load_config()
    train, valid, manifest = load_frozen()
    quota_n = int(manifest["capacity"]["quota_n"])
    params = {"quota_n": quota_n, "n_valid": len(valid), "seed": cfg.seed}
    y_valid = valid[S.TARGET]
    spec = load_catalog()
    features = [c for c in spec.feature_names() if c in train.columns]
    out: dict[str, object] = {"quota_n": quota_n}
    table: list[tuple[str, dict[str, float]]] = []

    e1 = evaluate(y_valid, _fit_score(train, valid, [*features, S.DURATION]), quota_n)
    out["E1"] = {"run_id": _log(E1, {**params, "added": S.DURATION}, e1), **e1}
    table.append((E1, e1))

    e2: dict[str, dict[str, float]] = {}
    for subset in ("A", "B"):
        x_train, y_train = split_xy(train, subset)
        x_valid, y_valid_sub = split_xy(valid, subset)
        model = build_model("lr", spec, list(x_train.columns))
        model.fit(x_train, y_train)
        e2[subset] = evaluate(y_valid_sub, model.predict_proba(x_valid)[:, 1], quota_n)
    out["E2"] = {
        "run_id": _log(E2, {**params, "comparison": "A vs B"}, e2["B"]),
        "A": e2["A"],
        "B": e2["B"],
    }
    table.append((f"{E2}-A", e2["A"]))
    table.append((f"{E2}-B", e2["B"]))

    train_filled = train.copy()
    valid_filled = valid.copy()
    for column in S.UNKNOWN_COLS:
        if column not in train_filled.columns:
            continue
        mode = train_filled.loc[train_filled[column] != S.UNKNOWN, column].mode().iloc[0]
        train_filled[column] = train_filled[column].replace(S.UNKNOWN, mode)
        valid_filled[column] = valid_filled[column].replace(S.UNKNOWN, mode)
    e3 = evaluate(y_valid, _fit_score(train_filled, valid_filled, features), quota_n)
    out["E3"] = {"run_id": _log(E3, {**params, "replace": "未知→训练集众数"}, e3), **e3}
    table.append((E3, e3))

    derived = [S.CONTACTED, S.PDAYS_LOG, S.PDAYS_BUCKET]
    e4_columns = [*[c for c in features if c not in derived], S.PDAYS]
    e4 = evaluate(y_valid, _fit_score(train, valid, e4_columns), quota_n)
    out["E4"] = {"run_id": _log(E4, {**params, "replaced": "pdays 三派生列→原列"}, e4), **e4}
    table.append((E4, e4))

    frame = pd.concat([train, valid], ignore_index=True)
    x_all, y_all = split_xy(frame, "B")
    order = np.random.default_rng(S.SEED).permutation(len(frame))
    cut = round(len(frame) * 0.8)
    fit_idx, ev_idx = order[:cut], order[cut:]
    model = build_model("lr", spec, list(x_all.columns))
    model.fit(x_all.iloc[fit_idx], y_all.iloc[fit_idx])
    e5_quota = round(len(ev_idx) * cfg.quota.frac)
    e5 = evaluate(
        y_all.iloc[ev_idx].to_numpy(dtype="int8"),
        model.predict_proba(x_all.iloc[ev_idx])[:, 1],
        e5_quota,
    )
    out["E5"] = {
        "run_id": _log(E5, {**params, "split": "random 80/20", "quota_n": e5_quota}, e5),
        **e5,
    }
    table.append((E5, e5))

    print(f"{'实验':<18}{'AP':>8}{'ROC-AUC':>10}{'名额内精确':>12}{'名额内召回':>12}{'Brier':>9}")
    for label, metrics in table:
        print(
            f"{label:<18}{metrics['ap']:>8.4f}{metrics['roc_auc']:>10.4f}"
            f"{metrics['precision_at_quota']:>12.4f}{metrics['recall_at_quota']:>12.4f}{metrics['brier']:>9.4f}"
        )
    path = S.ROOT / "data" / "processed" / "experiments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"明细写入 {path.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
