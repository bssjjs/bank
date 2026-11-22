"""M2 冻结层的单测：清洗台账、切分边界、指纹稳定、manifest 契约。

要点：
- 清洗规则只做"记账"，不删行；致命规则命中必须抛 ``CleaningError``。
- 切点是"切点当天归验证集"，一秒都不能含糊。
- 同一个数据集算两次指纹必须一致；列顺序变了指纹必须变。
- manifest 里的行数/正例/名额是与 ADR 0002 钉死的，改口径必须同时改这里。
"""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd
from bank import clean, freeze, repro, split
from bank import config as C
from bank import schema as S
from bank import timeaxis as T

POS = S.POSITIVE
NEG = S.NEGATIVE


def raw_frame(**overrides: object) -> pd.DataFrame:
    """两行合法的原始记录，17 列齐全；测试只改自己关心的列。"""
    base = {
        "年龄": [40, 41],
        "职业": ["技术员", "教师"],
        "婚姻状态": ["已婚", "未婚"],
        "教育程度": ["中学", "高等教育"],
        "是否信贷违约": ["否", "否"],
        "账户余额": [1000, 2000],
        "住房贷款": ["是", "否"],
        "个人贷款": ["否", "否"],
        "联系方式": ["手机", "座机"],
        "联系日期": [15, 16],
        "联系月份": ["十一月", "十一月"],
        "通话时长": [200, 300],
        "本次联系次数": [1, 1],
        "距上次联系天数": [-1, -1],
        "历史联系次数": [0, 0],
        "上次营销结果": ["未知", "未知"],
        "是否认购": [NEG, POS],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def rule_rows(findings: list[clean.Finding], prefix: str) -> int:
    for f in findings:
        if f.rule.startswith(prefix):
            return f.rows
    raise AssertionError(f"没有找到规则 {prefix}，台账不完整")


class CleaningRulesTest(unittest.TestCase):
    def test_all_rules_reported_in_fixed_order(self) -> None:
        findings = clean.run_checks(raw_frame())
        names = [f.rule for f in findings]
        self.assertEqual(
            names,
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
        self.assertEqual(clean.assert_clean(findings), None)
        self.assertTrue(all(f.rows == 0 for f in findings))

    def test_bad_label_is_fatal(self) -> None:
        findings = clean.run_checks(raw_frame(**{"是否认购": [NEG, "可能"]}))
        self.assertEqual(rule_rows(findings, "R1"), 1)
        with self.assertRaises(clean.CleaningError):
            clean.assert_clean(findings)

    def test_negative_pdays_other_than_sentinel_is_fatal(self) -> None:
        self.assertEqual(
            rule_rows(clean.run_checks(raw_frame(**{"距上次联系天数": [-1, -5]})), "R2"), 1
        )
        self.assertEqual(
            rule_rows(clean.run_checks(raw_frame(**{"距上次联系天数": [-1, 30]})), "R2"), 0
        )

    def test_pdays_conflict_is_recorded_not_fatal(self) -> None:
        findings = clean.run_checks(
            raw_frame(**{"距上次联系天数": [-1, 100], "上次营销结果": ["未知", "未知"]})
        )
        self.assertEqual(rule_rows(findings, "R3"), 1)
        self.assertFalse(next(f for f in findings if f.rule.startswith("R3")).fatal)
        clean.assert_clean(findings)  # 记账类不中止

    def test_unknown_month_is_fatal(self) -> None:
        findings = clean.run_checks(raw_frame(**{"联系月份": ["十一月", "十三月"]}))
        self.assertEqual(rule_rows(findings, "R4"), 1)
        with self.assertRaises(clean.CleaningError):
            clean.assert_clean(findings)

    def test_day_out_of_range_is_fatal(self) -> None:
        findings = clean.run_checks(raw_frame(**{"联系日期": [15, 32]}))
        self.assertEqual(rule_rows(findings, "R5"), 1)

    def test_null_and_negative_duration(self) -> None:
        findings = clean.run_checks(raw_frame(**{"职业": ["技术员", None], "通话时长": [200, -1]}))
        self.assertEqual(rule_rows(findings, "R6"), 1)
        self.assertEqual(rule_rows(findings, "R7"), 1)

    def test_summary_counts_fatal_and_recorded(self) -> None:
        findings = clean.run_checks(
            raw_frame(**{"距上次联系天数": [-1, 100], "上次营销结果": ["未知", "成功"]})
        )
        findings.append(clean.Finding("R9_测试用", 3, False, "仅用于验证汇总"))
        text = clean.summarize(findings)
        self.assertIn("命中 3 行（致命 0 行，记账 3 行）", text[0])


class SplitBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = C.load_config()

    def frame(self, dates: list[str], labels: list[int]) -> pd.DataFrame:
        return pd.DataFrame({T.DATE: pd.to_datetime(dates), S.TARGET: labels})

    def test_cut_day_belongs_to_valid(self) -> None:
        df = self.frame(["2009-09-30", "2009-10-01"], [1, 0])
        train, valid, facts = split.split_and_check(df, self.cfg)
        self.assertEqual(list(train[T.DATE]), [pd.Timestamp("2009-09-30")])
        self.assertEqual(list(valid[T.DATE]), [pd.Timestamp("2009-10-01")])
        self.assertEqual(facts["train"]["positives"], 1)
        self.assertEqual(facts["valid"]["positives"], 0)

    def test_out_of_window_rows_are_trimmed(self) -> None:
        df = self.frame(["2008-10-31", "2008-11-04", "2010-11-17", "2010-11-18"], [0, 1, 0, 1])
        train, valid, facts = split.split_and_check(df, self.cfg)
        self.assertEqual(len(train), 1)  # 只留 2008-11-04
        self.assertEqual(len(valid), 1)  # 只留 2010-11-17
        self.assertEqual(facts["window"]["rows"], 2)
        self.assertEqual(facts["window"]["date_min"], "2008-11-04")
        self.assertEqual(facts["window"]["date_max"], "2010-11-17")

    def test_train_valid_do_not_overlap(self) -> None:
        df = self.frame(["2009-09-01", "2009-10-01"], [1, 0])
        train, valid, _ = split.split_and_check(df, self.cfg)
        self.assertFalse(set(train.index) & set(valid.index))

    def test_quota_uses_valid_rows_as_denominator(self) -> None:
        # 踩坑清单第 9 条：名额 = 验证集行数 × 比例，不是全表
        self.assertEqual(split.quota_n(3556, self.cfg), 356)
        self.assertEqual(split.quota_n(10, self.cfg), 1)

    def test_missing_date_column_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            split.split_and_check(pd.DataFrame({S.TARGET: [0]}), self.cfg)


class FingerprintTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.25, 0.5]})

    def test_same_data_same_fingerprint(self) -> None:
        self.assertEqual(repro.sha256_table(self.frame()), repro.sha256_table(self.frame()))

    def test_column_order_changes_fingerprint(self) -> None:
        forward = repro.sha256_table(self.frame())
        backward = repro.sha256_table(self.frame()[["b", "a"]])
        self.assertNotEqual(forward, backward)

    def test_value_change_changes_fingerprint(self) -> None:
        other = self.frame()
        other.loc[0, "a"] = 99
        self.assertNotEqual(repro.sha256_table(self.frame()), repro.sha256_table(other))

    def test_canonical_json_is_key_order_independent(self) -> None:
        left = repro.sha256_json({"a": 1, "b": [1, 2]})
        right = repro.sha256_json({"b": [1, 2], "a": 1})
        self.assertEqual(left, right)


class FrozenArtifactTest(unittest.TestCase):
    """跑一次完整冻结，把 ADR 0002 定下的数字钉在测试里。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = C.load_config()
        cls.built = freeze.build(cls.cfg)

    def test_manifest_contract_fields(self) -> None:
        m = self.built.manifest
        self.assertEqual(m["contract"], "manifest.v1")
        self.assertEqual(m["builder"], "bank.freeze/1")
        self.assertEqual(m["source"]["table"], S.RAW_TABLE)
        self.assertEqual(m["raw"]["cols"], 17)
        self.assertEqual(m["raw"]["rows"], 45211)
        for key in ("window", "tables", "capacity", "cleaning", "columns", "fingerprint_rule"):
            self.assertIn(key, m)
        self.assertIn(T.DATE, m["columns"])
        self.assertIn(S.TARGET, m["columns"])

    def test_counts_match_adr_0002(self) -> None:
        m = self.built.manifest
        self.assertEqual((m["window"]["rows"], m["window"]["positives"]), (21093, 4109))
        self.assertEqual(
            (m["tables"]["train"]["rows"], m["tables"]["train"]["positives"]), (17537, 2327)
        )
        self.assertEqual(
            (m["tables"]["valid"]["rows"], m["tables"]["valid"]["positives"]), (3556, 1782)
        )
        self.assertEqual(m["window"]["cut"], "2009-10-01")
        self.assertEqual(m["window"]["buffer_days"], 0)

    def test_quota_is_ten_percent_of_valid(self) -> None:
        self.assertEqual(self.built.manifest["capacity"]["quota_n"], 356)

    def test_frozen_matches_manifest(self) -> None:
        f = self.built.frozen
        m = self.built.manifest
        self.assertEqual(f["contract"], "frozen.v1")
        for name in ("train", "valid"):
            self.assertEqual(
                f["tables"][name]["content_sha256"], m["tables"][name]["content_sha256"]
            )
            self.assertEqual(f["tables"][name]["rows"], m["tables"][name]["rows"])
        self.assertEqual(f["source"]["sha256"], m["source"]["sha256"])
        self.assertEqual(f["config"]["sha256"], m["config"]["sha256"])

    def test_rebuild_is_reproducible(self) -> None:
        again = freeze.build(self.cfg)
        self.assertEqual(again.manifest, self.built.manifest)
        self.assertEqual(again.frozen, self.built.frozen)

    def test_train_and_valid_are_disjoint_in_time(self) -> None:
        train_dates = self.built.train[T.DATE]
        valid_dates = self.built.valid[T.DATE]
        self.assertLess(train_dates.max(), valid_dates.min())
        self.assertLess(train_dates.max(), pd.Timestamp(self.cfg.split.cut))

    def test_check_passes_on_frozen_files(self) -> None:
        self.assertEqual(freeze.check(self.cfg), 0)

    def test_window_too_small_is_rejected(self) -> None:
        tiny = C.WindowCfg(start=date(2008, 11, 1), end=date(2008, 11, 30))
        cfg = self.cfg.model_copy(update={"window": tiny})
        with self.assertRaises(ValueError):
            freeze.build(cfg)


if __name__ == "__main__":
    unittest.main()
