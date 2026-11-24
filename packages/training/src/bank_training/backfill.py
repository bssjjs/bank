"""标签回流：把「后来到底认购了没有」贴回预测记录，产出可用的回流样本。

`python -m bank_training.backfill --labels data/processed/labels.csv`

标签文件两列：`request_id,label`（label 写 `1/0` 或 `是/否`）。这里做四件事：

1. **只认能对上号的**：`request_id` 不在预测记录里就报错退出（说明标签来自别的地方，不能悄悄丢）；
2. **不重复记账**：同一个 `request_id` 出现两次直接拒绝；模型重试导致
   同一号有多条预测时只取最后一条；
3. **对照组单独统计**：`control=True` 的样本不进重训集合（ADR 0008）；
4. 落两份文件：`data/processed/labeled.jsonl`（逐条）与
   `data/processed/feedback.json`（汇总，契约 `feedback.v1`）。

汇总里 `training_allowed` 才是"能拿去重训"的样本数，`coverage` 是标签覆盖率——
这两个数决定了 M7 里那三条重训条件算不算数。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from bank import predictions as P
from bank import schema as S

CONTRACT = "feedback.v1"
LOG_PATH = S.ROOT / "data" / "processed" / "predictions.jsonl"
LABELED_PATH = S.ROOT / "data" / "processed" / "labeled.jsonl"
FEEDBACK_PATH = S.ROOT / "data" / "processed" / "feedback.json"

_YES = {"1", "是", "true", "True", "yes"}
_NO = {"0", "否", "false", "False", "no"}


def read_labels(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"标签文件不存在：{path}")
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"request_id", "label"} - set(reader.fieldnames):
            raise ValueError(f"{path} 必须有两列：request_id,label；实际是 {reader.fieldnames}")
        for index, row in enumerate(reader, start=2):
            rid = (row.get("request_id") or "").strip()
            raw = (row.get("label") or "").strip()
            if not rid:
                raise ValueError(f"{path} 第 {index} 行没有 request_id")
            if raw in _YES:
                label = 1
            elif raw in _NO:
                label = 0
            else:
                raise ValueError(f"{path} 第 {index} 行 label 只能是 1/0（或 是/否），收到 {raw!r}")
            rows.append((rid, label))
    return rows


def join(labels: list[tuple[str, int]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predict_of = P.by_request_id(records)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rid, label in labels:
        if rid in seen:
            raise ValueError(f"request_id {rid} 在标签文件里出现了两次")
        seen.add(rid)
        entry = predict_of.get(rid)
        if entry is None:
            raise ValueError(f"request_id {rid} 不在预测记录里：标签来源对不上，拒绝静默丢弃")
        out.append(
            {
                "request_id": rid,
                "label": label,
                "score": entry.get("score"),
                "level": entry.get("level"),
                "model_version": entry.get("model_version"),
                "control": bool(entry.get("control")),
                "predicted_at": entry.get("ts"),
            }
        )
    return out


def feedback(
    labeled: list[dict[str, Any]], records: list[dict[str, Any]], labels: int
) -> dict[str, Any]:
    treatment = [row for row in labeled if not row["control"]]
    control = [row for row in labeled if row["control"]]

    def rate(rows: list[dict[str, Any]]) -> float:
        return round(sum(int(row["label"]) for row in rows) / len(rows), 6) if rows else 0.0

    return {
        "contract": CONTRACT,
        "predictions": len(records),
        "labeled": len(labeled),
        "labels_in_file": labels,
        "coverage": round(len(labeled) / len(records), 6) if records else 0.0,
        "training_allowed": len(treatment),
        "positive_rate_all": rate(labeled),
        "positive_rate_treatment": rate(treatment),
        "positive_rate_control": rate(control),
        "control_labeled": len(control),
        "treatment_scored": sorted(float(row["score"] or 0.0) for row in treatment),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把标签回流到预测记录")
    parser.add_argument("--labels", required=True, help="标签 CSV：request_id,label")
    parser.add_argument("--log", default=str(LOG_PATH), help="预测记录 jsonl")
    parser.add_argument("--out", default=str(LABELED_PATH), help="逐条回流样本输出")
    parser.add_argument("--feedback", default=str(FEEDBACK_PATH), help="汇总输出")
    args = parser.parse_args(argv)

    records = P.read(Path(args.log))
    labels = read_labels(Path(args.labels))
    try:
        labeled = join(labels, records)
    except ValueError as exc:
        print(f"回流失败：{exc}")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8", newline="\n") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = feedback(labeled, records, len(labels))
    scored = summary.pop("treatment_scored")
    _write_json(Path(args.feedback), summary)

    print(
        f"回流完成：预测记录 {summary['predictions']} 条，贴上标签 {summary['labeled']} 条"
        f"（覆盖率 {summary['coverage']:.1%}）"
    )
    allowed = summary["training_allowed"]
    held_out = summary["control_labeled"]
    print(f"可重训样本 {allowed} 条（对照组 {held_out} 条不进重训）")
    print(
        f"认购率：全部 {summary['positive_rate_all']:.4f}，"
        f"处理组 {summary['positive_rate_treatment']:.4f}，"
        f"对照组 {summary['positive_rate_control']:.4f}"
    )
    if scored:
        print(f"处理组分数范围 {scored[0]:.4f} ~ {scored[-1]:.4f}")
    print(f"写出：{args.out}")
    print(f"写出：{args.feedback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
