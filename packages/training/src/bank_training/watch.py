"""F6：定时检查（`python -m bank_training.watch`）。

一次检查做三件事，产出可查的报告与可投递的告警：

1. **漂移**：读 `data/processed/drift.json`（`bank_training.drift` 的产物），
   按 `configs/watch.yaml` 的阈值分级。
2. **回流**：读 `data/processed/feedback.json`（`bank_training.backfill` 的产物），
   看标签覆盖率与认购率相对基准的变化。
3. **服务**：可选，读 `GET /metrics`；服务不可达本身是一条 warn 告警，
   **不让监控自己崩掉**——监控崩了比指标变差更难发现。

重训建议（ADR 0008 的三条重训条件，这里覆盖两条；第三条要人等口径）：
- 同一特征 PSI ≥ 告警线，**连续两次检查**都超 → `psi_two_runs`。
- 回流认购率相对基准变化 ≥ `label_shift_pct` → `label_shift`。

退出码：有 `alert` 级告警返回 1（可以挂在 cron / 流水线的判定上），否则 0。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from bank import alerts as A
from bank import schema as S
from bank import yamlkit
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CONTRACT = "watch_report.v1"
OUT_PATH = S.ROOT / "data" / "processed" / "watch.json"
HISTORY_PATH = S.ROOT / "data" / "processed" / "watch_history.jsonl"
DRIFT_PATH = S.ROOT / "data" / "processed" / "drift.json"
FEEDBACK_PATH = S.ROOT / "data" / "processed" / "feedback.json"
CONFIG_PATH = S.ROOT / "configs" / "watch.yaml"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class WatchError(Exception):
    """配置或输入读不了：说清哪个文件、哪一项。"""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(_Strict):
    psi_watch: float = Field(ge=0.0, le=10.0)
    psi_alert: float = Field(gt=0.0, le=10.0)
    label_shift_pct: float = Field(gt=0.0, le=1000.0)
    coverage_min: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if self.psi_alert <= self.psi_watch:
            raise ValueError(
                "thresholds.psi_alert 必须大于 psi_watch（关注线在告警线下面才有意义）"
            )
        return self


class ChannelCfg(_Strict):
    kind: str = Field(min_length=1)
    path: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> ChannelCfg:
        if self.kind == "console":
            if self.path is not None or self.url is not None:
                raise ValueError("console 渠道不需要 path/url")
        elif self.kind == "file":
            if not self.path:
                raise ValueError("file 渠道必须写 path")
            if self.url is not None:
                raise ValueError("file 渠道不认 url")
        elif self.kind == "webhook":
            if not self.url:
                raise ValueError("webhook 渠道必须写 url")
            if not (self.url.startswith("http://") or self.url.startswith("https://")):
                raise ValueError("webhook 渠道的 url 必须以 http:// 或 https:// 开头")
            if self.path is not None:
                raise ValueError("webhook 渠道不认 path")
        else:
            raise ValueError(f"未知渠道 kind={self.kind!r}：只能是 console/file/webhook")
        return self


class ScheduleCfg(_Strict):
    interval_seconds: int = Field(ge=10, le=86_400)


class WatchConfig(_Strict):
    schema_version: int = Field(ge=1)
    thresholds: Thresholds
    channels: list[ChannelCfg] = Field(min_length=1)
    schedule: ScheduleCfg


def load_watch_config(path: Path = CONFIG_PATH) -> WatchConfig:
    try:
        raw = yamlkit.load_mapping(path, what="监控配置")
    except yamlkit.YamlError as exc:
        raise WatchError(str(exc)) from exc
    try:
        config = WatchConfig.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "<根>"
            lines.append(f"{where}: {error['msg']}")
        raise WatchError("监控配置校验失败：\n  " + "\n  ".join(lines)) from exc
    if config.schema_version != 1:
        raise WatchError(f"监控配置版本 {config.schema_version} 不认识：当前只支持 1")
    return config


def channels_from(config: WatchConfig, root: Path = S.ROOT) -> list[A.Channel]:
    """把配置里的渠道实例化。相对路径按仓库根解析，避免"在哪跑就从哪算"。"""
    built: list[A.Channel] = []
    for item in config.channels:
        if item.kind == "console":
            built.append(A.ConsoleChannel())
        elif item.kind == "file":
            path = Path(str(item.path))
            built.append(A.FileChannel(path if path.is_absolute() else root / path))
        else:
            built.append(A.WebhookChannel(str(item.url)))
    return built


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise WatchError(f"找不到{what}：{path}。先跑产出它的命令，再检查。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WatchError(f"{what} {path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise WatchError(f"{what} {path} 顶层必须是对象")
    return payload


def _feature_rows(drift: dict[str, Any]) -> list[dict[str, Any]]:
    """漂移报告的 features 可能是「列表」也可能是「名字→指标」的对象，两种都收。"""
    features = drift.get("features") or []
    if isinstance(features, dict):
        rows: list[dict[str, Any]] = []
        for name in sorted(features):
            entry = features[name]
            rows.append(
                {"feature": name, "psi": float(entry["psi"])}
                if isinstance(entry, dict)
                else {"feature": name, "psi": float(entry)}
            )
        return rows
    if not isinstance(features, list):
        raise WatchError("漂移报告的 features 必须是列表（每条含 feature/psi）或对象（名字→指标）")
    rows = []
    for entry in features:
        if not isinstance(entry, dict) or "feature" not in entry or "psi" not in entry:
            raise WatchError("漂移报告 features 的每一行都要有 feature 与 psi")
        rows.append({"feature": str(entry["feature"]), "psi": float(entry["psi"])})
    return rows


def psi_alerts(drift: dict[str, Any], thresholds: Thresholds) -> list[A.Alert]:
    """按 PSI 分级。只看报告里已有的特征——报告没算的不在这里补。"""
    found: list[A.Alert] = []
    for row in _feature_rows(drift):
        name, psi = row["feature"], row["psi"]
        if psi >= thresholds.psi_alert:
            found.append(
                A.Alert(
                    "alert",
                    "drift",
                    f"psi_{name}",
                    f"{name} 的 PSI={psi:.4f} ≥ 告警线 {thresholds.psi_alert}",
                    psi,
                )
            )
        elif psi >= thresholds.psi_watch:
            found.append(
                A.Alert(
                    "warn",
                    "drift",
                    f"psi_{name}",
                    f"{name} 的 PSI={psi:.4f} ≥ 关注线 {thresholds.psi_watch}",
                    psi,
                )
            )
    return found


def label_alerts(drift: dict[str, Any], thresholds: Thresholds) -> list[A.Alert]:
    """标签分布的变化：`drift.json` 的 `label.relative_change_pct`。"""
    label = drift.get("label") or {}
    if not isinstance(label, dict) or "relative_change_pct" not in label:
        return []
    shift = float(label["relative_change_pct"])
    if abs(shift) < thresholds.label_shift_pct:
        return []
    before = label.get("reference_rate")
    after = label.get("current_rate")
    head = f"认购率相对基准变化 {shift:+.1f}% ≥ 阈值 {thresholds.label_shift_pct:.0f}%"
    return [A.Alert("alert", "drift", "label_shift", head + f"（{before} → {after}）", shift)]


def feedback_alerts(feedback: dict[str, Any], thresholds: Thresholds) -> list[A.Alert]:
    """回流里最该被查的量：标签覆盖率。

    认购率变化由 `label_alerts` 从漂移报告读（那里有基准期与当前期两个比例可算），
    这里只在回流汇总**自己带了** `label_shift_pct` 时才再报一次。
    """
    found: list[A.Alert] = []
    if "coverage" in feedback:
        coverage = float(feedback["coverage"])
        if coverage < thresholds.coverage_min:
            head = f"回流覆盖率 {coverage:.1%} < 下限 {thresholds.coverage_min:.0%}"
            found.append(
                A.Alert(
                    "warn",
                    "feedback",
                    "coverage_low",
                    head + "：标签回来的比例太低，结论要打折扣",
                    coverage,
                )
            )
    shift = feedback.get("label_shift_pct")
    if shift is not None and abs(float(shift)) >= thresholds.label_shift_pct:
        head = f"回流汇总里认购率变化 {float(shift):+.1f}%"
        found.append(
            A.Alert(
                "alert",
                "feedback",
                "label_shift",
                head + f" ≥ 阈值 {thresholds.label_shift_pct:.0f}%",
                float(shift),
            )
        )
    return found


def _fetch_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("响应顶层不是对象")
    return payload


def service_alerts(base_url: str, fetch: Any = None) -> list[A.Alert]:
    """读 `/metrics`。服务不在线不算崩溃：给它一条 warn，检查继续。"""
    reader = fetch or _fetch_json
    url = base_url.rstrip("/") + "/metrics"
    try:
        payload = reader(url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return [
            A.Alert(
                "warn",
                "service",
                "service_unreachable",
                f"服务读不到 /metrics（{exc}）：监控这轮看不见线上状态",
            )
        ]
    found: list[A.Alert] = []
    model = payload.get("model") or {}
    if isinstance(model, dict) and not model.get("available", True):
        found.append(
            A.Alert(
                "alert",
                "service",
                "model_unavailable",
                "服务没有可用模型包：/v1/predict 现在会 503",
            )
        )
    reload_info = payload.get("reload") or {}
    failures = int(reload_info.get("reload_failures") or 0)
    if failures > 0:
        found.append(
            A.Alert(
                "warn",
                "service",
                "reload_failures",
                f"热加载失败 {failures} 次；最近一次：{reload_info.get('last_reload_error')}",
                float(failures),
            )
        )
    return found


def breached_features(found: list[A.Alert]) -> list[str]:
    """这轮达到「告警线」的特征名，写进历史给下一轮比。"""
    prefix = "psi_"
    return sorted(
        alert.code[len(prefix) :]
        for alert in found
        if alert.source == "drift" and alert.level == "alert" and alert.code.startswith(prefix)
    )


def previous_breaches(history_path: Path) -> list[str]:
    if not history_path.exists():
        return []
    lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise WatchError(f"历史文件 {history_path} 最后一行不是合法 JSON：{exc}") from exc
    return [str(name) for name in last.get("breaches") or []]


def retrain_alerts(found: list[A.Alert], before: list[str]) -> list[A.Alert]:
    """连续两轮都越线的特征 → 建议重训（ADR 0008 第一条）。"""
    repeated = sorted(set(breached_features(found)) & set(before))
    if not repeated:
        return []
    joined = "、".join(repeated)
    return [
        A.Alert(
            "alert",
            "retrain",
            "psi_two_runs",
            f"{joined} 连续两次检查都越过告警线：按 ADR 0008 该复核重训",
        )
    ]


def run_once(
    *,
    config: WatchConfig | None = None,
    drift_path: Path = DRIFT_PATH,
    feedback_path: Path = FEEDBACK_PATH,
    history_path: Path = HISTORY_PATH,
    out_path: Path = OUT_PATH,
    base_url: str | None = None,
    fetch: Any = None,
    send: bool = True,
) -> tuple[list[A.Alert], int]:
    """跑一轮检查，返回（告警列表, 退出码）。测试可以直接传路径与渠道替身。"""
    cfg = config or load_watch_config()
    drift = _read_json(drift_path, "漂移报告")
    found = psi_alerts(drift, cfg.thresholds)
    found.extend(label_alerts(drift, cfg.thresholds))
    if feedback_path.exists():
        found.extend(feedback_alerts(_read_json(feedback_path, "回流汇总"), cfg.thresholds))
    else:
        found.append(
            A.Alert(
                "warn",
                "feedback",
                "missing_feedback",
                f"还没有回流汇总 {feedback_path.name}：标签回流这条线是瞎的",
            )
        )
    if base_url:
        found.extend(service_alerts(base_url, fetch=fetch))
    found.extend(retrain_alerts(found, previous_breaches(history_path)))

    if send:
        for channel in channels_from(cfg):
            found.extend(channel.send(found))

    payload = {
        "contract": CONTRACT,
        "ts": A.now(),
        "drift_report": str(drift_path),
        "counts": {level: sum(1 for a in found if a.level == level) for level in A.LEVELS},
        "breaches": breached_features(found),
        "alerts": [alert.as_dict() for alert in found],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps({"ts": payload["ts"], "breaches": payload["breaches"]}, ensure_ascii=False)
            + "\n"
        )

    code = 1 if A.highest(found) == "alert" else 0
    return found, code


def _summary(found: list[A.Alert], code: int) -> str:
    counts = {level: sum(1 for a in found if a.level == level) for level in A.LEVELS}
    head = f"检查完成：info {counts['info']} / warn {counts['warn']} / alert {counts['alert']}"
    tail = "有 alert 级告警：需要人看" if code else "没有 alert 级告警"
    return f"{head}；{tail}"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths: dict[str, Path] = {
        "--drift": DRIFT_PATH,
        "--feedback": FEEDBACK_PATH,
        "--history": HISTORY_PATH,
        "--out": OUT_PATH,
    }
    overrides: dict[str, Path] = {}
    interval: int | None = None
    base_url: str | None = None
    send = True
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--no-send":
            send = False
            index += 1
        elif item in paths:
            overrides[item] = Path(args[index + 1])
            index += 2
        elif item == "--interval":
            interval = int(args[index + 1])
            index += 2
        elif item == "--base-url":
            base_url = args[index + 1]
            index += 2
        else:
            print(f"看不懂的参数：{item}", file=sys.stderr)
            return 2
    try:
        config = load_watch_config()
    except WatchError as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 1
    chosen = {
        "drift_path": overrides.get("--drift", DRIFT_PATH),
        "feedback_path": overrides.get("--feedback", FEEDBACK_PATH),
        "history_path": overrides.get("--history", HISTORY_PATH),
        "out_path": overrides.get("--out", OUT_PATH),
    }
    while True:
        try:
            found, code = run_once(config=config, base_url=base_url, send=send, **chosen)
        except WatchError as exc:
            print(f"检查失败：{exc}", file=sys.stderr)
            return 1
        for alert in found:
            print(f"[{alert.level}] {alert.source}/{alert.code}: {alert.message}")
        print(_summary(found, code))
        print(f"报告：{chosen['out_path']}")
        if interval is None or interval <= 0:
            return code
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
