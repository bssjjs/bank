"""特征目录：它是唯一出处，所以"目录本身合法"和"目录与代码一致"都要测。"""

from __future__ import annotations

import unittest
from pathlib import Path

from bank import catalog as C
from bank import schema as S


class CatalogLoadTest(unittest.TestCase):
    def test_counts(self) -> None:
        spec = C.load_catalog()
        self.assertEqual(len(spec.fields), 33)
        self.assertEqual(len(spec.feature_names()), 24)
        self.assertEqual(len(spec.forbidden_names()), 8)

    def test_required_forbidden_present(self) -> None:
        spec = C.load_catalog()
        for name in C.REQUIRED_FORBIDDEN:
            self.assertIn(name, spec.forbidden_names())

    def test_feature_names_exclude_forbidden(self) -> None:
        spec = C.load_catalog()
        self.assertNotIn(S.DURATION, spec.feature_names())
        self.assertNotIn(S.PDAYS, spec.feature_names())
        self.assertNotIn(S.TARGET_RAW, spec.feature_names())

    def test_agrees_with_schema(self) -> None:
        C.assert_agrees_with_schema(C.load_catalog())  # 不抛就是一致

    def test_fingerprint_is_stable(self) -> None:
        first = C.load_catalog().fingerprint()
        second = C.load_catalog().fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_field_lookup_reports_missing(self) -> None:
        spec = C.load_catalog()
        with self.assertRaises(C.CatalogError):
            spec.field("不存在的列")


class CatalogValidationTest(unittest.TestCase):
    def _base(self) -> dict:
        return {
            "schema_version": 1,
            "fields": [
                {
                    "name": S.TARGET_RAW,
                    "type": "binary",
                    "display": "标签",
                    "forbidden": True,
                    "reason": "标签本身",
                    "categories": ["是", "否"],
                },
                {
                    "name": S.DURATION,
                    "type": "numeric",
                    "display": "通话时长",
                    "forbidden": True,
                    "reason": "结果泄漏",
                    "min": 0,
                    "max": 5000,
                },
                {
                    "name": S.PDAYS,
                    "type": "numeric",
                    "display": "距上次联系",
                    "forbidden": True,
                    "reason": "-1 是哨兵",
                    "min": -1,
                    "max": 900,
                },
            ],
        }

    def test_missing_reason_rejected(self) -> None:
        raw = self._base()
        raw["fields"][1]["reason"] = None
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("reason", str(ctx.exception))

    def test_numeric_needs_bounds(self) -> None:
        raw = self._base()
        del raw["fields"][2]["max"]
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("min 与 max", str(ctx.exception))

    def test_binary_needs_two_categories(self) -> None:
        raw = self._base()
        raw["fields"][0]["categories"] = ["是"]
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("binary", str(ctx.exception))

    def test_unknown_field_type_rejected(self) -> None:
        raw = self._base()
        raw["fields"][2]["type"] = "float"
        with self.assertRaises(Exception):
            C.CatalogSpec.model_validate(raw)

    def test_missing_forbidden_column_rejected(self) -> None:
        raw = self._base()
        raw["fields"] = raw["fields"][:2]
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("禁用列", str(ctx.exception))

    def test_forbidden_flag_not_set_rejected(self) -> None:
        raw = self._base()
        raw["fields"][1]["forbidden"] = False
        raw["fields"][1]["reason"] = None
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("forbidden", str(ctx.exception))

    def test_duplicate_name_rejected(self) -> None:
        raw = self._base()
        raw["fields"].append(dict(raw["fields"][2]))
        with self.assertRaises(Exception) as ctx:
            C.CatalogSpec.model_validate(raw)
        self.assertIn("重复", str(ctx.exception))


class CatalogOnDiskTest(unittest.TestCase):
    def test_notebook_path_is_not_read(self) -> None:
        # 目录必须只有一份，别在 notebook 里另抄一份字段表
        spec_path = C.CATALOG_PATH
        self.assertTrue(spec_path.exists())
        self.assertEqual(spec_path.name, "features.yaml")
        self.assertTrue(Path(spec_path).parent.name == "configs")


if __name__ == "__main__":
    unittest.main()
