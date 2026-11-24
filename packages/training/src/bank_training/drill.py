"""F7 故障演练：往「线上」批次注入已知的坏数据，看监控能不能发现。

监控不是"有就行"，要能证明它**真的会响**。这里不跑完整服务，直接对冻结集做五件事：

1. 把 `账户余额` 整体放大（数值漂移）→ `psi_numeric` 必须报「严重」；
2. 把一半的 `职业` 改成「未知」（类别漂移）→ `psi_categorical` 必须报「严重」；
3. 把六成未认购改成正例（标签漂移）→ 正例率相对变化必须过 ±50% 这条重训线；
4. 对照：没动过的 `婚姻状态` 不许被误报（PSI < 0.1）；
5. 反证：同一份注入数据，如果两边各算各的分位数（自适应分箱），PSI 会被抹平到 0.1 以下
   ——这条证明「分箱边界只取基准集」不是多余的讲究。

任何一项没按预期响，演练就算失败（退出码 1）。这种"自己给自己考试"的检查比一句
「我们装了监控」有用得多。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bank import drift as D
from bank import schema as S

from bank_training.data import load_frozen
from bank_training.tracking import get_tracker

CONTRACT = "drill_report.v1"
DRILL_SEED = 20261122
SHIFT_FACTOR = 1.5
SHIFT_OFFSET = 5000.0
CATEGORY_SHARE = 0.5
LABEL_SHARE = 0.6
LABEL_LINE = 50.0  # 正例率相对变化（%），与 ADR 0008 的重训条件一致
OUT_PATH = S.ROOT / "data" / "processed" / "drill.json"
EXPERIMENT = "bank-marketing"


@dataclass(frozen=True)
class Check:
    name: str
    expect: str
    observed: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expect": self.expect,
            "observed": self.observed,
            "passed": self.passed,
        }


# ---- 注入 ---------------------------------------------------------------


def shift_numeric(
    frame: pd.DataFrame, column: str, *, factor: float = SHIFT_FACTOR, offset: float = SHIFT_OFFSET
) -> pd.DataFrame:
    """整体放大再抬高：确定性的，重跑结果一样。"""
    result = frame.copy()
    result[column] = result[column].astype("float64") * factor + offset
    return result


def shift_category(
    frame: pd.DataFrame, column: str, *, share: float = CATEGORY_SHARE, seed: int = DRILL_SEED
) -> pd.DataFrame:
    """按固定种子选一部分行，把类别改成「未知」。"""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share 必须在 [0, 1]")
    result = frame.copy()
    rng = np.random.default_rng(seed)
    picked = rng.random(len(result)) < share
    result.loc[picked, column] = S.UNKNOWN
    return result


def flip_negatives(
    frame: pd.DataFrame, *, share: float = LABEL_SHARE, seed: int = DRILL_SEED
) -> pd.DataFrame:
    """把一部分未认购改成认购：模拟「这批客户突然变得很愿意买」。"""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share 必须在 [0, 1]")
    result = frame.copy()
    negatives = result.index[result["label"] == 0]
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(negatives), size=round(len(negatives) * share), replace=False)
    result.loc[negatives[picked], "label"] = 1
    return result


# ---- 反证 ---------------------------------------------------------------


def adaptive_psi_numeric(reference: pd.Series, current: pd.Series, bins: int = D.BINS) -> float:
    """故意用错口径：两边各算各的分位数，看 PSI 会不会被抹平。"""
    ref_edges = D.quantile_edges(reference, bins)
    cur_edges = D.quantile_edges(current, bins)
    ref_shares = D.shares(reference, ref_edges)
    cur_shares = D.shares(current, cur_edges)
    return D.psi_from_shares(ref_shares, cur_shares)


# ---- 检查 ---------------------------------------------------------------


def run(train: pd.DataFrame, valid: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []

    shifted = shift_numeric(valid, S.ACCOUNT)
    numeric = D.psi_numeric(train[S.ACCOUNT], shifted[S.ACCOUNT])
    checks.append(
        Check(
            name=f"数值漂移：{S.ACCOUNT} ×{SHIFT_FACTOR}+{SHIFT_OFFSET:.0f}",
            expect=f"PSI ≥ {D.WATCH}（严重）",
            observed=f"PSI {numeric.psi:.4f}（{numeric.level}）",
            passed=numeric.psi >= D.WATCH,
        )
    )

    recoded = shift_category(valid, S.OCCUPATION)
    categorical = D.psi_categorical(train[S.OCCUPATION], recoded[S.OCCUPATION])
    checks.append(
        Check(
            name=f"类别漂移：{S.OCCUPATION} 一半改成「未知」",
            expect=f"PSI ≥ {D.WATCH}（严重）",
            observed=f"PSI {categorical.psi:.4f}（{categorical.level}）",
            passed=categorical.psi >= D.WATCH,
        )
    )

    flipped = flip_negatives(valid)
    before = float(valid["label"].mean())
    after = float(flipped["label"].mean())
    shift = D.label_rate_shift(before, after)
    checks.append(
        Check(
            name="标签漂移：六成未认购改成认购",
            expect=f"正例率相对变化 ≥ {LABEL_LINE:.0f}%",
            observed=f"{before:.4f} → {after:.4f}（{shift:+.1f}%）",
            passed=shift >= LABEL_LINE,
        )
    )

    control = D.psi_categorical(train[S.MARITAL], valid[S.MARITAL])
    checks.append(
        Check(
            name=f"对照：没动过的 {S.MARITAL}",
            expect=f"PSI < {D.STABLE}（稳定，不许误报）",
            observed=f"PSI {control.psi:.4f}（{control.level}）",
            passed=control.psi < D.STABLE,
        )
    )

    adaptive = adaptive_psi_numeric(train[S.ACCOUNT], shifted[S.ACCOUNT])
    checks.append(
        Check(
            name="反证：同一份注入数据若两边各自定箱",
            expect=f"PSI < {D.STABLE}（漂移被自适应分箱抹平）",
            observed=f"PSI {adaptive:.4f}",
            passed=adaptive < D.STABLE,
        )
    )
    return checks


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _log(checks: list[Check], rows: int) -> str | None:
    tracker = get_tracker()
    with tracker.start_run("f7-drill") as run_id:
        tracker.log_params({"seed": DRILL_SEED, "rows": rows, "contract": CONTRACT})
        for index, check in enumerate(checks):
            tracker.log_metric(f"check_{index}", 1.0 if check.passed else 0.0)
        tracker.set_tag("stage", "drill")
        return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F7 故障演练：注入坏数据，看监控会不会响")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    train, valid, _manifest = load_frozen()
    checks = run(train, valid)
    passed = all(check.passed for check in checks)

    print(f"演练：往「线上」批次注入 {len(checks)} 类故障（固定种子 {DRILL_SEED}）")
    print(f"{'预期':<28} {'实测':<24} 结果")
    for check in checks:
        mark = "响" if check.passed else "没响"
        print(f"[{mark}] {check.name}")
        print(f"       预期 {check.expect}；实测 {check.observed}")

    run_id = None if args.no_mlflow else _log(checks, len(valid))
    _write_json(
        Path(args.out),
        {
            "contract": CONTRACT,
            "seed": DRILL_SEED,
            "rows": {"reference": len(train), "current": len(valid)},
            "checks": [check.as_dict() for check in checks],
            "passed": passed,
            "mlflow_run_id": run_id,
        },
    )
    if run_id:
        print(f"mlflow run：{run_id}")
    print(f"写出：{args.out}")
    if not passed:
        failed = [check.name for check in checks if not check.passed]
        print("演练失败：以下检查没按预期响应 → " + "；".join(failed))
        return 1
    print(f"演练通过：{len(checks)} 项全部按预期（监控响 3 次、对照没误报、反证成立）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
