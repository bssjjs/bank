"""考卷：特征目录（A4）。人出题，智能体只负责考过。

目录是"字段类型的唯一出处"：数据检查、训练、接口、页面都读它。所以题目集中在
"改坏它会不会被发现"：
- 禁用列少了、漏了理由、被偷偷改成可用；
- 数值列没范围、bin 类别不只两个、字段重名、类型拼错；
- 目录与 `schema.py` 说的不是同一件事。

这套题的价值在于：它是**变异测试的人工版**——每道题就是一次"把代码改坏"，
考不过说明守门是假的。
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from bank.catalog import (
    CATALOG_PATH,
    CatalogError,
    assert_agrees_with_schema,
    load_catalog,
)
from bank.dataset import model_features

ROOT = Path(__file__).resolve().parents[2]
BASE = CATALOG_PATH.read_text(encoding="utf-8")
SCRATCH = ROOT / ".tmp" / "exam"

AGE = (
    "{name: 年龄, type: numeric, display: 年龄, unit: 岁, "
    "min: 18, max: 95, derived: false, forbidden: false}"
)
CREDIT = (
    "{name: 是否信贷违约, type: binary, display: 是否有信贷违约, "
    "derived: false, forbidden: false, categories: [是, 否]}"
)
CONTACT = (
    "{name: 联系方式, type: categorical, display: 联系方式, "
    "derived: false, forbidden: false, categories: [手机, 固话, 未知]}"
)
DURATION_REASON = (
    "    reason: 结果泄漏——只有通话结束才知道；单变量 AUC 0.8076，比很多完整模型都高\n"
)
LABEL_BLOCK = """  - name: 是否认购
    type: binary
    display: 是否认购定期存款（标签）
    categories: [是, 否]
    derived: false
    forbidden: true
    reason: 标签本身，只在训练时读
"""


def variant(*replacements: tuple[str, str]) -> Path:
    text = BASE
    for old, new in replacements:
        if old not in text:
            raise AssertionError(f"features.yaml 里找不到要替换的片段：{old[:60]!r}")
        text = text.replace(old, new, 1)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"{uuid.uuid4().hex}.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


class ShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_catalog()

    def test_field_counts(self) -> None:
        self.assertEqual(len(self.spec.fields), 33)
        self.assertEqual(len(self.spec.feature_names()), 24)
        self.assertEqual(len(self.spec.forbidden_names()), 8)

    def test_features_are_exactly_schema_feature_set_a(self) -> None:
        self.assertEqual(set(self.spec.feature_names()), set(model_features("A")))

    def test_agrees_with_schema(self) -> None:
        assert_agrees_with_schema(self.spec)  # 不抛异常就算过

    def test_required_forbidden_columns_are_all_there(self) -> None:
        for name in ("通话时长", "距上次联系天数", "是否认购"):
            self.assertIn(name, self.spec.forbidden_names())

    def test_every_forbidden_column_explains_why(self) -> None:
        for field in self.spec.fields:
            if field.forbidden:
                self.assertTrue(field.reason, f"{field.name} 标了禁用却没写理由")

    def test_every_field_has_chinese_display_name(self) -> None:
        for field in self.spec.fields:
            self.assertTrue(field.display.strip(), f"{field.name} 没有中文名")

    def test_fingerprint_is_stable_64_hex(self) -> None:
        first = self.spec.fingerprint()
        self.assertEqual(len(first), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in first))
        self.assertEqual(load_catalog().fingerprint(), first)


class MutationTest(unittest.TestCase):
    """每一条都是"把目录改坏一处"，必须被拦下。"""

    def test_min_greater_than_max_is_rejected(self) -> None:
        path = variant((AGE, AGE.replace("min: 18, max: 95", "min: 118, max: 95")))
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        self.assertIn("min 不能大于 max", str(caught.exception))

    def test_forbidden_without_reason_is_rejected(self) -> None:
        path = variant((DURATION_REASON, ""))
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        self.assertIn("reason", str(caught.exception))

    def test_binary_with_three_values_is_rejected(self) -> None:
        path = variant(
            (CREDIT, CREDIT.replace("categories: [是, 否]", "categories: [是, 否, 未知]"))
        )
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        self.assertIn("正好两个", str(caught.exception))

    def test_unknown_type_is_rejected(self) -> None:
        path = variant((AGE, AGE.replace("type: numeric", "type: money")))
        with self.assertRaises(CatalogError):
            load_catalog(path)

    def test_duplicate_field_name_is_rejected(self) -> None:
        path = variant(("{name: 婚姻状态,", "{name: 职业,"))
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        self.assertIn("字段名重复", str(caught.exception))

    def test_missing_required_forbidden_column_is_rejected(self) -> None:
        path = variant((LABEL_BLOCK, ""))
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        message = str(caught.exception)
        self.assertIn("必须出现在目录里", message)
        self.assertIn("是否认购", message)

    def test_forbidden_flag_removed_on_label_is_rejected(self) -> None:
        path = variant((LABEL_BLOCK, LABEL_BLOCK.replace("forbidden: true", "forbidden: false")))
        with self.assertRaises(CatalogError) as caught:
            load_catalog(path)
        self.assertIn("必须标 forbidden", str(caught.exception))

    def test_feature_downgraded_to_source_breaks_parity(self) -> None:
        # 联系方式 是特征集 A 的一员；把它改成 source，目录与 schema 就对不上了
        path = variant(
            (CONTACT, CONTACT.replace("derived: false,", "role: source, derived: false,"))
        )
        spec = load_catalog(path)
        with self.assertRaises(CatalogError) as caught:
            assert_agrees_with_schema(spec)
        self.assertIn("联系方式", str(caught.exception))

    def test_changing_a_range_changes_fingerprint(self) -> None:
        base = load_catalog().fingerprint()
        other = load_catalog(variant((AGE, AGE.replace("max: 95", "max: 96")))).fingerprint()
        self.assertNotEqual(base, other)


if __name__ == "__main__":
    unittest.main()
