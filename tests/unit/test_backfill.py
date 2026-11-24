"""标签回流的单测：只认对得上号的标签、对照组不进重训、汇总数字能对上。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any

from bank import predictions as P
from bank import schema as S
from bank_training import backfill as BF


def _tmp_dir() -> Path:
    root = S.ROOT / ".tmp" / "backfill" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def _records() -> list[dict[str, Any]]:
    return [
        {
            "request_id": "bank-1",
            "score": 0.9,
            "level": "优先",
            "model_version": "1.0.0",
            "control": False,
            "ts": "2026-11-01T09:00:00+08:00",
        },
        {
            "request_id": "bank-2",
            "score": 0.4,
            "level": "不建议",
            "model_version": "1.0.0",
            "control": True,
            "ts": "2026-11-01T09:01:00+08:00",
        },
        {
            "request_id": "bank-3",
            "score": 0.6,
            "level": "备选",
            "model_version": "1.0.0",
            "control": False,
            "ts": "2026-11-01T09:02:00+08:00",
        },
    ]


class JoinTest(unittest.TestCase):
    def test_join_keeps_prediction_facts(self) -> None:
        rows = BF.join([("bank-1", 1), ("bank-3", 0)], _records())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["score"], 0.9)
        self.assertEqual(rows[0]["level"], "优先")
        self.assertFalse(rows[0]["control"])
        self.assertEqual(rows[1]["label"], 0)

    def test_unknown_request_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            BF.join([("bank-404", 1)], _records())
        self.assertIn("bank-404", str(ctx.exception))

    def test_duplicate_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BF.join([("bank-1", 1), ("bank-1", 0)], _records())


class FeedbackTest(unittest.TestCase):
    def test_stats_are_consistent(self) -> None:
        labeled = BF.join([("bank-1", 1), ("bank-2", 0), ("bank-3", 1)], _records())
        summary = BF.feedback(labeled, _records(), labels=3)
        self.assertEqual(summary["contract"], BF.CONTRACT)
        self.assertEqual(summary["predictions"], 3)
        self.assertEqual(summary["labeled"], 3)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["control_labeled"], 1)
        self.assertEqual(summary["training_allowed"], 2)
        self.assertEqual(summary["positive_rate_all"], 0.666667)
        self.assertEqual(summary["positive_rate_treatment"], 1.0)
        self.assertEqual(summary["positive_rate_control"], 0.0)
        self.assertEqual(summary["treatment_scored"], [0.6, 0.9])

    def test_empty_records_do_not_divide_by_zero(self) -> None:
        summary = BF.feedback([], [], labels=0)
        self.assertEqual(summary["coverage"], 0.0)
        self.assertEqual(summary["positive_rate_all"], 0.0)


class LabelsFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _tmp_dir()
        self.addCleanup(shutil.rmtree, self.root, True)

    def _write(self, body: str) -> Path:
        path = self.root / "labels.csv"
        path.write_text(body, encoding="utf-8", newline="\n")
        return path

    def test_chinese_yes_no(self) -> None:
        rows = BF.read_labels(self._write("request_id,label\nbank-1,是\nbank-2,否\n"))
        self.assertEqual(rows, [("bank-1", 1), ("bank-2", 0)])

    def test_bad_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BF.read_labels(self._write("request_id,label\nbank-1,maybe\n"))

    def test_missing_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BF.read_labels(self._write("id,label\nbank-1,1\n"))

    def test_main_writes_files_and_summary(self) -> None:
        log = self.root / "predictions.jsonl"
        for entry in _records():
            P.append(log, entry)
        labels = self._write("request_id,label\nbank-1,1\nbank-2,0\n")
        out = self.root / "labeled.jsonl"
        feedback = self.root / "feedback.json"
        code = BF.main(
            [
                "--labels",
                str(labels),
                "--log",
                str(log),
                "--out",
                str(out),
                "--feedback",
                str(feedback),
            ]
        )
        self.assertEqual(code, 0)
        written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(written), 2)
        summary = json.loads(feedback.read_text(encoding="utf-8"))
        self.assertEqual(summary["labeled"], 2)
        self.assertEqual(summary["control_labeled"], 1)
        self.assertEqual(summary["training_allowed"], 1)

    def test_main_returns_one_on_unknown_id(self) -> None:
        log = self.root / "predictions.jsonl"
        for entry in _records():
            P.append(log, entry)
        labels = self._write("request_id,label\nbank-999,1\n")
        code = BF.main(
            [
                "--labels",
                str(labels),
                "--log",
                str(log),
                "--out",
                str(self.root / "o.jsonl"),
                "--feedback",
                str(self.root / "f.json"),
            ]
        )
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "o.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
