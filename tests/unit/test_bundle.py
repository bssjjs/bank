"""M4 模型包的守门测试：契约字段、校验和、缺列拒绝、不覆盖、版本指针。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from bank import bundle as B
from bank import catalog, dataset, features
from bank import schema as S
from bank_training.data import load_frozen
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _new_root() -> Path:
    """在仓库 .tmp 下开空目录：系统临时目录受沙箱 ACL 限制，清理时删不掉。"""
    root = S.ROOT / ".tmp" / "test-bundle" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def _doc_kwargs(catalog_spec: catalog.CatalogSpec, names: list[str], version: str) -> dict:
    inputs = [
        field.model_dump(include={"name", "type", "display", "unit", "min", "max", "categories"})
        for field in catalog_spec.fields
        if not field.derived and not field.forbidden
    ]
    return {
        "version": version,
        "feature_set": "B",
        "feature_names": names,
        "feature_types": {field.name: str(field.type) for field in catalog_spec.fields},
        "inputs": inputs,
        "catalog_fingerprint": catalog_spec.fingerprint(),
        "decision": {
            "mode": "quota",
            "quota_frac": 0.1,
            "quota_n": 36,
            "level_cutoffs": {"优先": 0.7, "备选": 0.4},
            "threshold": 0.5,
            "explain_top_k": 5,
        },
        "metrics": {"ap": 0.6, "roc_auc": 0.65, "brier": 0.28},
        "training": {
            "run_id": "test-run",
            "experiment": "test",
            "data_fingerprint_train": "0" * 16,
            "data_fingerprint_valid": "1" * 16,
            "config_fingerprint": "2" * 16,
            "git_sha": "abcdef0",
            "trained_at": "2026-11-22T00:00:00+08:00",
            "rows_train": 500,
            "rows_valid": 200,
            "n_features": len(names),
        },
        "model_file": {"name": B.MODEL_FILE, "sha256": "a" * 64, "size": 1},
    }


class BundleFixture(unittest.TestCase):
    """用真实冻结集的一小段训一个小模型，包是真的能读回来打分的。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.valid, _ = load_frozen()
        cls.spec = catalog.load_catalog()
        cls.names = dataset.model_features("B")
        cls.model = Pipeline(
            [
                ("pre", features.build_preprocessor(cls.spec, cls.names, dense=True)),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )
        cls.model.fit(cls.train[cls.names].head(800), cls.train["label"].head(800))

    def _root(self) -> Path:
        root = _new_root()
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def _write(self, root: Path, version: str = "9.9.9") -> Path:
        kwargs = _doc_kwargs(self.spec, self.names, version)
        kwargs.pop("model_file")
        return B.write_bundle(model=self.model, root=root, **kwargs)


class BuildDocTest(BundleFixture):
    def test_doc_has_checksum_and_contract(self) -> None:
        doc = B.build_doc(**_doc_kwargs(self.spec, self.names, "1.2.3"))
        self.assertEqual(doc["contract"], "model_bundle.v1")
        self.assertEqual(len(doc["checksum"]["value"]), 64)
        self.assertNotIn("checksum", B._payload(doc))

    def test_feature_types_must_cover_all_features(self) -> None:
        kwargs = _doc_kwargs(self.spec, self.names, "1.2.3")
        kwargs["feature_types"] = {"年龄": "numeric"}
        with self.assertRaises(B.BundleError):
            B.build_doc(**kwargs)

    def test_bad_version_rejected(self) -> None:
        with self.assertRaises(Exception):
            B.build_doc(**_doc_kwargs(self.spec, self.names, "1.2"))

    def test_required_columns_follow_inputs(self) -> None:
        doc = B.build_doc(**_doc_kwargs(self.spec, self.names, "1.2.3"))
        self.assertEqual(doc["required_columns"], [item["name"] for item in doc["inputs"]])
        self.assertNotIn("是否认购", doc["required_columns"])


class WriteLoadTest(BundleFixture):
    def test_roundtrip_score_and_describe(self) -> None:
        loaded = B.load_bundle(self._write(self._root() / "serving"))
        self.assertEqual(loaded.version, "9.9.9")
        self.assertEqual(len(loaded.feature_names), len(self.names))
        score = loaded.score(self.valid.head(1))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        described = loaded.describe()
        self.assertEqual(len(described["inputs"]), len(loaded.doc.required_columns))
        self.assertEqual(described["decision"]["quota_n"], 36)

    def test_missing_column_is_refused(self) -> None:
        loaded = B.load_bundle(self._write(self._root() / "serving"))
        broken = self.valid.head(2).drop(columns=["职业"])
        with self.assertRaises(B.MissingColumnsError) as caught:
            loaded.score_frame(broken)
        self.assertIn("职业", str(caught.exception))

    def test_levels_use_frozen_cutoffs(self) -> None:
        loaded = B.load_bundle(self._write(self._root() / "serving"))
        self.assertEqual(loaded.level(0.95), "优先")
        self.assertEqual(loaded.level(0.5), "备选")
        self.assertEqual(loaded.level(0.1), "不建议")

    def test_explain_returns_top_k(self) -> None:
        loaded = B.load_bundle(self._write(self._root() / "serving"))
        items = loaded.explain(self.valid.head(1))
        self.assertEqual(len(items), 5)
        self.assertEqual(set(items[0]), {"feature", "value", "contribution"})


class NoOverwriteTest(BundleFixture):
    def test_second_write_is_refused(self) -> None:
        root = self._root() / "serving"
        path = self._write(root)
        before = (path / B.BUNDLE_JSON).read_text(encoding="utf-8")
        with self.assertRaises(B.BundleError):
            self._write(root)
        self.assertEqual((path / B.BUNDLE_JSON).read_text(encoding="utf-8"), before)
        self.assertEqual([item.name for item in root.iterdir()], ["9.9.9"])


class TamperTest(BundleFixture):
    def test_changed_json_is_refused(self) -> None:
        path = self._write(self._root() / "serving")
        doc = json.loads((path / B.BUNDLE_JSON).read_text(encoding="utf-8"))
        doc["metrics"]["ap"] = 0.99
        (path / B.BUNDLE_JSON).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(B.BundleError) as caught:
            B.load_bundle(path)
        self.assertIn("校验和", str(caught.exception))

    def test_changed_model_file_is_refused(self) -> None:
        path = self._write(self._root() / "serving")
        (path / B.MODEL_FILE).write_bytes(b"not a joblib file")
        with self.assertRaises(B.BundleError):
            B.load_bundle(path)


class CurrentPointerTest(BundleFixture):
    def test_pointer_roundtrip_and_missing_version(self) -> None:
        root = self._root() / "serving"
        with self.assertRaises(B.BundleError):
            B.read_current(root)
        self._write(root)
        B.write_current("9.9.9", root)
        self.assertEqual(B.read_current(root), "9.9.9")
        self.assertEqual(B.load_bundle(root).version, "9.9.9")
        with self.assertRaises(B.BundleError):
            B.write_current("1.0.0", root)


if __name__ == "__main__":
    unittest.main()
