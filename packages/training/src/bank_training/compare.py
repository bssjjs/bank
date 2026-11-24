"""模型与规则基线放在同一张桌上比：名额放宽到 15%、20% 各能多抓到几个阳性。

讲义 C-L2c 问的是「模型比规则强多少、名额要不要放宽」，所以这里固定三件事：

1. 同一份验证集、同一个名额比例，模型与规则用**同一套指标**算；
2. 规则基线用 `baselines.rule_score` 的原分值，**不为对比好看去调权重**
   （要调权重得另开 ADR，规则基线照量表打分的理由写在 ADR 0017）；
3. 名额比例是实验变量：10%（当前配置）、15%、20%，看多打电话能多抓到多少人、
   精确率掉多少——这是业务方决定名额宽窄时唯一该看的表。

输出 `data/processed/compare.{json,md}`；`compare.md` 里带结论段，数字全部由本次运行算出。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import numpy as np
from bank import catalog
from bank import schema as S
from bank.config import load_config
from bank.repro import git_sha

from bank_training import baselines as B
from bank_training import data as D
from bank_training import metrics as M
from bank_training import model_config as MC
from bank_training import pipeline as P

CONTRACT = "compare_report.v1"
FRACTIONS = (0.10, 0.15, 0.20)
OUT_DIR = S.ROOT / "data" / "processed"
JSON_PATH = OUT_DIR / "compare.json"
MD_PATH = OUT_DIR / "compare.md"
EXPERIMENT = "bank-marketing"
TRACKING_URI = f"sqlite:///{(S.ROOT / 'mlflow.db').as_posix()}"
MODEL_NAME = "逻辑回归"
RULE_NAME = "规则基线"


def quota_rows(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    fractions: tuple[float, ...] = FRACTIONS,
) -> list[dict[str, Any]]:
    """每个名额比例下，每个打分方式抓到多少阳性。"""
    y = np.asarray(y_true).astype("int8")
    n = len(y)
    positives = int(y.sum())
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        quota_n = max(1, min(n, round(n * fraction)))
        for name, score in scores.items():
            got = M.confusion(y, np.asarray(score, dtype="float64"), quota_n)
            tp = int(got["tp"])
            picked = int(quota_n)
            rows.append(
                {
                    "fraction": float(fraction),
                    "quota_n": picked,
                    "method": name,
                    "tp": tp,
                    "precision": tp / picked if picked else 0.0,
                    "recall": tp / positives if positives else 0.0,
                    "base_rate": positives / n if n else 0.0,
                }
            )
    return rows


def gains(
    rows: list[dict[str, Any]], fractions: tuple[float, ...] = FRACTIONS
) -> list[dict[str, Any]]:
    """同一名额比例下，模型比规则多抓到几个人、精确率差多少。"""
    out: list[dict[str, Any]] = []
    for fraction in fractions:
        pair = {
            row["method"]: row for row in rows if abs(float(row["fraction"]) - fraction) < 1e-12
        }
        if MODEL_NAME not in pair or RULE_NAME not in pair:
            continue
        model = pair[MODEL_NAME]
        rule = pair[RULE_NAME]
        out.append(
            {
                "fraction": float(fraction),
                "quota_n": int(model["quota_n"]),
                "extra_positives": int(model["tp"]) - int(rule["tp"]),
                "precision_gap": float(model["precision"]) - float(rule["precision"]),
            }
        )
    return out


def render(rows: list[dict[str, Any]]) -> str:
    """Markdown：对比表 + 名额取舍表 + 结论。"""
    lines = [
        "# 模型 vs 规则基线：同一验证集上的名额取舍",
        "",
        "固定同一份验证集（不重抽、不调权重），只改两件事：谁打分、名额放多宽。",
        "",
        "## 对比表",
        "",
        "| 名额比例 | 名额人数 | 打分方式 | 抓到阳性 | 名额内精确率 | 名额内召回率 |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {:.0%} | {:d} | {} | {:d} | {:.4f} | {:.4f} |".format(
                row["fraction"],
                int(row["quota_n"]),
                row["method"],
                int(row["tp"]),
                float(row["precision"]),
                float(row["recall"]),
            )
        )
    pairs = gains(rows, tuple(sorted({float(row["fraction"]) for row in rows})))
    lines += [
        "",
        "## 名额取舍：多打电话换多少人",
        "",
        "| 名额比例 | 名额人数 | 模型比规则多抓 | 精确率差 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for gain in pairs:
        lines.append(
            "| {:.0%} | {:d} | {:+d} | {:+.4f} |".format(
                gain["fraction"],
                gain["quota_n"],
                gain["extra_positives"],
                gain["precision_gap"],
            )
        )
    if not pairs:
        lines.append("| — | — | — | — |")
        lines += ["", "（表里没有同时出现模型与规则，算不出差值。）", ""]
        return "\n".join(lines)
    strict = pairs[0]
    relaxed = pairs[-1]
    lines += [
        "",
        "## 结论",
        "",
        "- 名额 {:.0%} → {:.0%}：人数 {:d} → {:d}，模型比规则多抓的 {:d} 人 → {:d} 人。".format(
            strict["fraction"],
            relaxed["fraction"],
            strict["quota_n"],
            relaxed["quota_n"],
            strict["extra_positives"],
            relaxed["extra_positives"],
        ),
        "- 精确率差 {:+.4f} → {:+.4f}：放宽名额既不会自动让模型赢，"
        "也不会自动让它输，要看这一列。".format(strict["precision_gap"], relaxed["precision_gap"]),
        "- 规则基线按自己的量程打分，没有为对比好看调权重；调权重得另开 ADR。",
        "",
    ]
    return "\n".join(lines)


def _log(payload: dict[str, Any]) -> str:
    """把这张表记成一个 mlflow run，方便和其他运行对上号。"""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="c-l2c-compare") as run:
        mlflow.log_params(
            {
                "fractions": ",".join(f"{fraction:.2f}" for fraction in FRACTIONS),
                "model": MODEL_NAME,
                "rule": RULE_NAME,
            }
        )
        mlflow.set_tags({"git_sha": git_sha() or "", "contract": CONTRACT})
        for row in payload["table"]:
            key = "f{:.0f}_{}".format(row["fraction"] * 100, row["method"])
            mlflow.log_metric(f"tp_{key}", int(row["tp"]))
            mlflow.log_metric(f"precision_{key}", float(row["precision"]))
        mlflow.log_artifact(str(MD_PATH), artifact_path="compare")
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模型 vs 规则的名额取舍表")
    parser.add_argument("--no-mlflow", action="store_true", help="不写 mlflow")
    args = parser.parse_args(argv)

    cfg = load_config()
    spec = catalog.load_catalog()
    params = MC.load_model_config()
    train, valid, _manifest = D.load_frozen()
    x_train, y_train = D.split_xy(train, cfg.features.set)
    x_valid, y_valid = D.split_xy(valid, cfg.features.set)
    if params.model.kind not in P.MODEL_KINDS:
        raise ValueError(f"模型种类必须是 {P.MODEL_KINDS}，配置里是 {params.model.kind}")
    model = P.build_model(
        params.model.kind,
        spec,
        list(x_train.columns),
        c=params.model.c,
        max_iter=params.model.max_iter,
    )
    model.fit(x_train, y_train)
    scores = {
        MODEL_NAME: np.asarray(model.predict_proba(x_valid)[:, 1], dtype="float64"),
        RULE_NAME: np.asarray(B.rule_score(valid), dtype="float64"),
    }
    rows = quota_rows(np.asarray(y_valid), scores)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(render(rows), encoding="utf-8", newline="\n")
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "feature_set": cfg.features.set,
        "window": {
            "train_rows": len(y_train),
            "valid_rows": len(y_valid),
            "valid_positives": int(np.asarray(y_valid).sum()),
        },
        "fractions": list(FRACTIONS),
        "table": rows,
        "gains": gains(rows),
    }
    if not args.no_mlflow:
        payload["mlflow_run_id"] = _log(payload)
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for row in rows:
        print(
            "{:>4.0%} 名额 {:>4d} 人  {:<4} 抓到 {:>4d}  精确 {:.4f}  召回 {:.4f}".format(
                row["fraction"],
                int(row["quota_n"]),
                row["method"],
                int(row["tp"]),
                float(row["precision"]),
                float(row["recall"]),
            )
        )
    print(f"写出：{MD_PATH} 与 {JSON_PATH}")
    print("对比完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
