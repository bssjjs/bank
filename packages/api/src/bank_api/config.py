"""M5 服务配置：`configs/api.yaml`，拼错键或越界值启动即报错。"""

from __future__ import annotations

from pathlib import Path

from bank import schema as S
from bank.config import ConfigError
from bank.yamlkit import YamlError, load_mapping
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_PATH = S.ROOT / "configs" / "api.yaml"


class ApiConfigError(ConfigError):
    """服务配置不合法。"""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServingCfg(_Strict):
    batch_limit: int = Field(ge=1, le=1000)
    request_id_prefix: str = Field(min_length=1, max_length=16)


class LogCfg(_Strict):
    path: str = Field(min_length=1)


class ApiConfig(_Strict):
    schema_version: int
    serving: ServingCfg
    log: LogCfg

    @model_validator(mode="after")
    def _sane(self) -> ApiConfig:
        if self.schema_version != 1:
            raise ValueError(f"schema_version 必须是 1，收到 {self.schema_version}")
        return self

    @property
    def log_path(self) -> Path:
        """配置里写相对仓库根的路径，换台机器不用改。"""
        path = Path(self.log.path)
        return path if path.is_absolute() else S.ROOT / path


def load_api_config(path: Path = CONFIG_PATH) -> ApiConfig:
    try:
        data = load_mapping(path, what="服务配置")
    except YamlError as error:
        raise ApiConfigError(str(error)) from error
    try:
        return ApiConfig.model_validate(data)
    except Exception as error:
        raise ApiConfigError(f"{path} 不合法：{error}") from error
