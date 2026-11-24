"""算法一致性：每个登记过的算法都在同一份冻结数据上跑一遍，指标与分数指纹要对得上参照。

为什么要有它（讲义 C-L2b）：`algorithms/` 让"加一种算法"变成加一个文件，但加得太容易就会
有人改坏了共享的预处理还看不出来。这里把每个算法的四项指标（AP、ROC-AUC、名额内精确率、
名额内召回率）与**逐条分数的指纹**冻结成参照；以后谁动了预处理器或算法，`--check` 会当场
报出是哪个算法、哪一项变了。

    python -m bank_training.parity            # 重算并写出 data/processed/algorithm_parity.json
    python -m bank_training.parity --check    # 与 baselines/v1 的参照比（容差 1e-6）

参照文件跟基线快照一起进 `baselines/`（封条由 `scripts/freeze_baseline.py` 维护）。容差 1e-6
是讲义的要求：同一个算法换台机器跑，浮点误差远小于它；真改了逻辑，差的一定比它大。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bank import catalog, repro
from bank import config as config_mod
from bank import schema as S

from bank_training import algorithms, data
from bank_training import metrics as M
from bank_training import model_config as MC

CONTRACT = "algorithm_parity.v1"
TOLERANCE = 1e-6
OUT_PATH = S.ROOT / "data" / "processed" / "algorithm_parity.json"
REF_PATH = S.ROOT / "baselines" / "v1" / "algorithm_parity.json"

_FIELDS = ("ap", "roc_auc", "precision_at_quota", "recall_at_quota")
_COUNTS = ("n_selected",)


def _fingerprint(scores: np.ndarray) -> str:
    """逐条分数的指纹：`%.10g` 拼成文本再 sha256（与内容指纹同一套小数口径）。"""
    text = ",".join(f"{float(v):.10g}" for v in scores)
    return repro.sha256_text(text)


def measure(
    name: str,
    *,
    frame: pd.DataFrame,
    valid: pd.DataFrame,
    feature_set: str,
    quota_n: int,
    spec: catalog.CatalogSpec,
    c: float,
    max_iter: int,
) -> dict[str, Any]:
    """训练一个算法、在验证集上打分，返回它的指标与分数指纹。"""
    x_train, y_train = data.split_xy(frame, feature_set)
    x_valid, y_valid = data.split_xy(valid, feature_set)
    pipeline = algorithms.build(name, spec, list(x_train.columns), c=c, max_iter=max_iter)
    pipeline.fit(x_train, y_train)
    scores = np.asarray(pipeline.predict_proba(x_valid)[:, 1], dtype="float64")
    got = M.evaluate(y_valid, scores, quota_n)
    return {
        "n": int(got["n"]),
        "positives": int(got["positives"]),
        "quota_n": int(got["quota_n"]),
        "n_selected": int(got["quota_n"]),
        "ap": float(got["ap"]),
        "roc_auc": float(got["roc_auc"]),
        "precision_at_quota": float(got["precision_at_quota"]),
        "recall_at_quota": float(got["recall_at_quota"]),
        "scores_sha256": _fingerprint(scores),
    }


def run() -> dict[str, Any]:
    """在冻结集上把每个登记过的算法跑一遍，返回完整报告。"""
    cfg = config_mod.load_config()
    params = MC.load_model_config()
    spec = catalog.load_catalog()
    train, valid, manifest = data.load_frozen()
    quota_n = int(manifest["capacity"]["quota_n"])
    report: dict[str, Any] = {
        "contract": CONTRACT,
        "feature_set": cfg.features.set,
        "c": float(params.model.c),
        "max_iter": int(params.model.max_iter),
        "seed": int(S.SEED),
        "data_fingerprint_train": manifest["tables"]["train"]["content_sha256"][:16],
        "algorithms": {},
    }
    for name in algorithms.available():
        report["algorithms"][name] = measure(
            name,
            frame=train,
            valid=valid,
            feature_set=cfg.features.set,
            quota_n=quota_n,
            spec=spec,
            c=params.model.c,
            max_iter=params.model.max_iter,
        )
    return report


def write(path: Path = OUT_PATH, report: dict[str, Any] | None = None) -> Path:
    payload = run() if report is None else report
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _differences(got: dict[str, Any], ref: dict[str, Any], tolerance: float) -> list[str]:
    problems: list[str] = []
    for name, expected in ref["algorithms"].items():
        actual = got["algorithms"].get(name)
        if actual is None:
            problems.append(f"缺算法 {name}：登记表里有，这次没跑出来")
            continue
        for field in _COUNTS:
            if int(actual[field]) != int(expected[field]):
                problems.append(f"{name}.{field}: 期望 {expected[field]}，实际 {actual[field]}")
        for field in _FIELDS:
            delta = abs(float(actual[field]) - float(expected[field]))
            if delta > tolerance:
                problems.append(
                    f"{name}.{field}: 期望 {expected[field]:.6f}，"
                    f"实际 {actual[field]:.6f}（差 {delta:.2e}）"
                )
        if actual["scores_sha256"] != expected["scores_sha256"]:
            problems.append(f"{name}.scores_sha256: 逐条分数变了")
    for name in sorted(set(got["algorithms"]) - set(ref["algorithms"])):
        problems.append(f"多出算法 {name}：登记了但没进参照，先跑 --write 再说明为什么")
    return problems


def check(ref: Path = REF_PATH, tolerance: float = TOLERANCE) -> int:
    if not ref.is_file():
        print(f"缺 {ref.relative_to(S.ROOT).as_posix()}：先跑 --write 并冻结基线", file=sys.stderr)
        return 1
    expected = json.loads(ref.read_text(encoding="utf-8"))
    got = run()
    problems = _differences(got, expected, tolerance)
    if not problems:
        count = len(expected["algorithms"])
        print(f"算法一致性通过：{count} 个算法，指标与分数指纹都与参照一致")
        return 0
    for line in problems:
        print(f"[不一致] {line}")
    print("算法一致性未通过：先看是哪个算法变了，别改参照把红变绿")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="算法一致性与指标参照")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="重算并写出 algorithm_parity.json")
    group.add_argument("--check", action="store_true", help="与 baselines/v1 的参照比")
    args = parser.parse_args(argv)
    if args.write:
        path = write()
        print(f"算法一致性已写出：{path.relative_to(S.ROOT).as_posix()}")
        return 0
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
