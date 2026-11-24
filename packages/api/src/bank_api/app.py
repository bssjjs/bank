"""M5 推理服务：只读模型包，不 import 训练代码。

三个约定：
1. 请求体字段由模型包生成（`schemas.build_input_model`），越界/枚举错由 pydantic 拦，
   统一转成 FastAPI 的 422，`detail[].loc` 精确指到字段名；
2. 打分与解释都走 `bundle.score` / `bundle.explain`，与导出时同一条代码路径；
3. 每条预测追加写一行 jsonl（`bank.predictions` 的 v2 记录：时间、契约、请求号、模型版本、
   分数、等级、是否对照、输入指纹、名额口径），不写客户特征值——输入只留 sha256。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from bank import bundle as B
from bank import predictions as P
from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from bank_api.config import ApiConfig, load_api_config
from bank_api.schemas import (
    LEVEL_ADVICE,
    BatchResult,
    Factor,
    Prediction,
    build_input_model,
    to_frame,
)

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append(path: Path, record: dict[str, Any]) -> None:
    # 并发下也不许两行粘在一起：锁在进程内，单 worker 起服务（见 README）
    with _LOCK:
        P.append(path, record)


def _body_errors(error: ValidationError) -> list[dict[str, Any]]:
    return [{**item, "loc": ("body", *item["loc"])} for item in error.errors()]


class _ApiApp(FastAPI):
    """把动态请求模型塞进 OpenAPI：字段名/类型/范围/枚举都要能在 /docs 里看到。"""

    def __init__(
        self, *args: Any, input_model: type[BaseModel], batch_limit: int, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_model = input_model
        self.batch_limit = batch_limit

    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema
        schema = get_openapi(title=self.title, version=self.version, routes=self.routes)
        body = self.input_model.model_json_schema(ref_template="#/components/schemas/{model}")
        definitions = body.pop("$defs", {})
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        components["PredictInput"] = body
        components.update(definitions)
        ref = {"$ref": "#/components/schemas/PredictInput"}
        schema["paths"]["/v1/predict"]["post"]["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": ref}},
        }
        schema["paths"]["/v1/predict/batch"]["post"]["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "array", "maxItems": self.batch_limit, "items": ref}
                }
            },
        }
        self.openapi_schema = schema
        return schema


def create_app(
    bundle_path: Path | None = None,
    log_path: Path | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """建服务。模型包加载失败不崩，`/health` 照常 200，`/ready` 与预测接口返回 503。"""
    config: ApiConfig = load_api_config(config_path) if config_path else load_api_config()
    log_file = log_path or config.log_path

    state: dict[str, Any] = {"bundle": None, "error": None, "input_model": None}
    try:
        bundle = B.load_bundle(bundle_path or B.SERVING_DIR)
        state["bundle"] = bundle
        state["input_model"] = build_input_model(bundle)
    except Exception as error:  # 启动时没包也能起来，方便 /health 与部署探针
        state["error"] = f"{type(error).__name__}: {error}"
        state["input_model"] = None

    app = _ApiApp(
        title="银行电话营销 · 外呼排序服务",
        version="1.0.0",
        input_model=state["input_model"] or BaseModel,
        batch_limit=config.serving.batch_limit,
    )

    def not_ready() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": f"模型包未就绪：{state['error']}"},
        )

    def request_id() -> str:
        return f"{config.serving.request_id_prefix}-{uuid.uuid4().hex[:12]}"

    def predict_one(payload: dict[str, Any], rid: str, control: bool | None) -> Prediction:
        bundle = state["bundle"]
        if bundle is None:
            raise RuntimeError("模型包未就绪")
        model = state["input_model"]
        try:
            data = model(**payload)
        except ValidationError as error:
            raise RequestValidationError(_body_errors(error)) from error
        clean = data.model_dump(mode="json")
        frame = to_frame(clean)
        score = bundle.score(frame)
        level = bundle.level(score)
        # 没显式指定就按 request_id 哈希抽 10% 作对照组（确定性抽样，重放结果一致）
        flag = P.is_control(rid) if control is None else control
        decision = bundle.doc.decision
        _append(
            log_file,
            P.record(
                request_id=rid,
                model_version=bundle.version,
                score=score,
                level=level,
                control=flag,
                payload=clean,
                quota_frac=decision.quota_frac,
                quota_n=decision.quota_n,
            ),
        )
        factors = [Factor(**item) for item in bundle.explain(frame)]
        return Prediction(
            request_id=rid,
            model_version=bundle.version,
            score=round(score, 6),
            level=level,
            factors=factors,
            advice=LEVEL_ADVICE[level],
        )

    @app.get("/health", summary="存活探针：进程起来了就 200")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "bank-api"}

    @app.get("/ready", summary="就绪探针：模型包读得回来才 200")
    def ready() -> Any:
        if state["bundle"] is None:
            return not_ready()
        return {"status": "ready", "model_version": state["bundle"].version}

    @app.get("/v1/model", summary="模型信息：前端靠它生成表单")
    def model_info() -> Any:
        if state["bundle"] is None:
            return not_ready()
        return state["bundle"].describe()

    @app.post("/v1/predict", response_model=Prediction, summary="单条预测")
    def predict(
        payload: dict[str, Any],
        control: bool | None = Query(None, description="对照组标记；不填按 10% 哈希抽样自动判定"),
    ) -> Any:
        if state["bundle"] is None:
            return not_ready()
        return predict_one(payload, request_id(), control)

    @app.post("/v1/predict/batch", response_model=BatchResult, summary="批量预测")
    def predict_batch(items: list[dict[str, Any]]) -> Any:
        if state["bundle"] is None:
            return not_ready()
        if not items:
            raise RequestValidationError(
                [{"loc": ("body",), "msg": "批量请求不能为空", "type": "value_error"}]
            )
        if len(items) > config.serving.batch_limit:
            raise RequestValidationError(
                [
                    {
                        "loc": ("body",),
                        "msg": f"一次最多 {config.serving.batch_limit} 条，收到 {len(items)} 条",
                        "type": "value_error",
                    }
                ]
            )
        rid = request_id()
        # 一个批次一个请求号：对照组标记按批次算，同一批里的条目共用一个标记
        flag = P.is_control(rid)
        return BatchResult(
            model_version=state["bundle"].version,
            request_id=rid,
            items=[predict_one(item, rid, flag) for item in items],
        )

    return app


app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
