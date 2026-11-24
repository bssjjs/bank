"""契约测试：/docs 里能看到的字段、范围、枚举，必须和模型包、和目录对得上。"""

from __future__ import annotations

import unittest
from typing import Any

from bank import bundle as B
from bank import catalog
from bank_api.app import create_app
from fastapi.testclient import TestClient

from tests.support import ensure_served_bundle


class OpenApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_served_bundle()
        cls.bundle = B.load_bundle(B.SERVING_DIR)
        cls.spec = catalog.load_catalog()
        cls.client = TestClient(create_app())
        cls.doc: dict[str, Any] = cls.client.get("/openapi.json").json()

    def _body_schema(self) -> dict[str, Any]:
        return self.doc["components"]["schemas"]["PredictInput"]

    def test_required_fields_match_bundle(self) -> None:
        body = self._body_schema()
        self.assertEqual(body["required"], list(self.bundle.doc.required_columns))
        self.assertEqual(set(body["properties"]), set(self.bundle.doc.required_columns))

    def test_every_field_has_description(self) -> None:
        for name, prop in self._body_schema()["properties"].items():
            self.assertTrue(prop.get("description"), f"{name} 少了中文说明")

    def test_numeric_ranges_are_in_docs(self) -> None:
        props = self._body_schema()["properties"]
        self.assertEqual(props["年龄"]["minimum"], 18.0)
        self.assertEqual(props["年龄"]["maximum"], 95.0)
        self.assertEqual(props["距上次联系天数"]["minimum"], -1.0)
        self.assertEqual(props["距上次联系天数"]["maximum"], 871.0)

    def test_categorical_enums_come_from_catalog(self) -> None:
        props = self._body_schema()["properties"]
        field = self.spec.by_name()["职业"]
        self.assertEqual(set(props["职业"]["enum"]), {str(item) for item in field.categories})
        self.assertIn("未知", props["职业"]["enum"])

    def test_batch_limit_is_documented(self) -> None:
        content = self.doc["paths"]["/v1/predict/batch"]["post"]["requestBody"]["content"]
        schema = content["application/json"]["schema"]
        self.assertEqual(schema["maxItems"], 100)
        self.assertEqual(schema["items"], {"$ref": "#/components/schemas/PredictInput"})

    def test_required_routes_exist(self) -> None:
        paths = set(self.doc["paths"])
        self.assertEqual(
            paths,
            {"/health", "/ready", "/v1/model", "/metrics", "/v1/predict", "/v1/predict/batch"},
        )
        self.assertIn("get", self.doc["paths"]["/health"])
        self.assertIn("get", self.doc["paths"]["/ready"])
        self.assertIn("get", self.doc["paths"]["/metrics"])


if __name__ == "__main__":
    unittest.main()
