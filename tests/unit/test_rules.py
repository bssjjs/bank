"""清洗规则注册表：规则必须"登记过才能跑"，否则台账与文档会对不上。"""

from __future__ import annotations

import unittest

import pandas as pd
from bank import clean as CL
from bank import rules as R


class RegistryTest(unittest.TestCase):
    def test_seven_rules_registered_in_order(self) -> None:
        codes = [r.code for r in R.all_rules()]
        self.assertEqual(
            codes,
            [
                "R1_标签取值",
                "R2_pdays非法负值",
                "R3_pdays与上次结果冲突",
                "R4_月份无法识别",
                "R5_日号越界",
                "R6_真空值",
                "R7_通话时长为负",
            ],
        )

    def test_fatal_flags(self) -> None:
        by_code = {r.code: r for r in R.all_rules()}
        self.assertFalse(by_code["R3_pdays与上次结果冲突"].fatal)
        for code in ("R1_标签取值", "R2_pdays非法负值", "R4_月份无法识别", "R6_真空值"):
            self.assertTrue(by_code[code].fatal, code)

    def test_duplicate_registration_rejected(self) -> None:
        with self.assertRaises(ValueError):

            @R.rule("R1_标签取值", "重复", fatal=True, note="")
            def _dup(df: pd.DataFrame) -> pd.Series:
                return df["x"] > 0

    def test_every_rule_has_note(self) -> None:
        for spec in R.all_rules():
            self.assertTrue(spec.note, spec.code)

    def test_run_checks_follows_registry(self) -> None:
        from tests.unit.test_freeze import raw_frame

        findings = CL.run_checks(raw_frame())
        self.assertEqual([f.rule for f in findings], [r.code for r in R.all_rules()])


if __name__ == "__main__":
    unittest.main()
