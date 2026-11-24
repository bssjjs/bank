"""预测记录 v2 的单测：确定性对照组、输入指纹、追加与回读。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import predictions as P
from bank import schema as S


def _tmp_dir() -> Path:
    root = S.ROOT / ".tmp" / "predictions" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


class ControlTest(unittest.TestCase):
    def test_same_request_id_gets_same_answer(self) -> None:
        first = [P.is_control(f"bank-{i:06d}") for i in range(50)]
        second = [P.is_control(f"bank-{i:06d}") for i in range(50)]
        self.assertEqual(first, second)

    def test_rate_zero_and_one(self) -> None:
        self.assertFalse(any(P.is_control(f"bank-{i}", rate=0.0) for i in range(200)))
        self.assertTrue(all(P.is_control(f"bank-{i}", rate=1.0) for i in range(200)))

    def test_default_rate_is_about_ten_percent(self) -> None:
        hits = sum(P.is_control(f"bank-{i:05d}") for i in range(3000))
        self.assertTrue(0.08 < hits / 3000 < 0.12, hits / 3000)

    def test_bad_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            P.is_control("bank-x", rate=1.5)


class FeatureHashTest(unittest.TestCase):
    def test_order_does_not_matter(self) -> None:
        a = {"年龄": 46, "职业": "退休"}
        b = {"职业": "退休", "年龄": 46}
        self.assertEqual(P.features_hash(a), P.features_hash(b))

    def test_value_matters(self) -> None:
        self.assertNotEqual(P.features_hash({"年龄": 46}), P.features_hash({"年龄": 47}))

    def test_hash_is_hex_and_hides_values(self) -> None:
        digest = P.features_hash({"年龄": 46, "职业": "退休"})
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in digest))
        self.assertNotIn("退休", digest)


class RecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _tmp_dir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.path = self.root / "predictions.jsonl"

    def _entry(self, rid: str = "bank-abc") -> dict[str, object]:
        return P.record(
            request_id=rid,
            model_version="1.0.0",
            score=0.5123456789,
            level="备选",
            control=False,
            payload={"年龄": 46, "职业": "退休"},
            quota_frac=0.1,
            quota_n=356,
        )

    def test_record_has_contract_and_no_raw_features(self) -> None:
        entry = self._entry()
        self.assertEqual(entry["contract"], P.CONTRACT)
        self.assertEqual(entry["score"], 0.512346)
        self.assertEqual(entry["quota_n"], 356)
        self.assertEqual(entry["features_hash"], P.features_hash({"年龄": 46, "职业": "退休"}))
        text = json.dumps(entry, ensure_ascii=False)
        self.assertNotIn("职业", text)
        self.assertNotIn("退休", text)

    def test_append_then_read(self) -> None:
        P.append(self.path, self._entry("bank-1"))
        P.append(self.path, self._entry("bank-2"))
        rows = P.read(self.path)
        self.assertEqual([row["request_id"] for row in rows], ["bank-1", "bank-2"])

    def test_read_missing_file_is_empty(self) -> None:
        self.assertEqual(P.read(self.path), [])

    def test_latest_record_wins(self) -> None:
        P.append(self.path, self._entry("bank-1"))
        later = self._entry("bank-1")
        later["score"] = 0.9
        P.append(self.path, later)
        latest = P.by_request_id(P.read(self.path))
        self.assertEqual(latest["bank-1"]["score"], 0.9)
        self.assertEqual(len(P.read(self.path)), 2)

    def test_broken_line_reports_line_number(self) -> None:
        self.path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            P.read(self.path)
        self.assertIn("第 2 行", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
