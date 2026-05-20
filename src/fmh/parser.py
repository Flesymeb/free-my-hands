from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from fmh.models import DeploymentRequest, EventSource


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParserConfig:
    allowed_path_prefixes: tuple[str, ...] = (
        "/",
        "s3://",
        "oss://",
        "hdfs://",
        "hf://",
        "gs://",
    )
    required_trigger_words: tuple[str, ...] = ("deploy_vllm", "部署模型", "启动模型")


_KEY_ALIASES = {
    "weight_path": "weight_path",
    "weights": "weight_path",
    "path": "weight_path",
    "model_path": "weight_path",
    "checkpoint": "weight_path",
    "ckpt": "weight_path",
    "权重路径": "weight_path",
    "路径": "weight_path",
    "model_name": "model_name",
    "model": "model_name",
    "name": "model_name",
    "模型名": "model_name",
    "gpu_count": "gpu_count",
    "gpus": "gpu_count",
    "gpu": "gpu_count",
    "卡数": "gpu_count",
    "gpu_type": "gpu_type",
    "gpu型号": "gpu_type",
    "卡型": "gpu_type",
    "port": "port",
    "端口": "port",
    "env_name": "env_name",
    "env": "env_name",
    "环境": "env_name",
    "extra_args": "extra_args",
    "args": "extra_args",
    "vllm_args": "extra_args",
    "额外参数": "extra_args",
    "document_id": "document_id",
    "doc_id": "document_id",
    "文档id": "document_id",
    "status_block_id": "status_block_id",
    "block_id": "status_block_id",
    "状态块id": "status_block_id",
}


def parse_deployment_request(source: EventSource, config: ParserConfig | None = None) -> DeploymentRequest:
    config = config or ParserConfig()
    text = source.text.strip()
    if not text:
        raise ParseError("empty deployment request")

    fields = _parse_key_values(text)
    has_trigger = any(word in text for word in config.required_trigger_words)
    has_path_key = "weight_path" in fields
    if not has_trigger and not has_path_key:
        raise ParseError(
            "request must contain deploy_vllm or an explicit weight_path/model_path field"
        )

    weight_path = fields.get("weight_path", "").strip()
    if not weight_path:
        weight_path = _extract_first_path(text, config.allowed_path_prefixes)
    if not weight_path:
        raise ParseError("missing required field: weight_path")
    if not _is_allowed_weight_path(weight_path, config.allowed_path_prefixes):
        raise ParseError(f"unsupported weight_path prefix: {weight_path}")

    model_name = fields.get("model_name") or _derive_model_name(weight_path)
    gpu_count = _parse_int(fields.get("gpu_count", "1"), "gpu_count")
    if gpu_count < 1:
        raise ParseError("gpu_count must be >= 1")
    port = None
    if fields.get("port"):
        port = _parse_int(fields["port"], "port")

    request_id = _make_request_id(source, weight_path, model_name)
    return DeploymentRequest(
        request_id=request_id,
        source_type=source.source_type,
        source_ref=source.source_ref,
        requester=source.requester,
        weight_path=weight_path,
        model_name=model_name,
        gpu_count=gpu_count,
        gpu_type=fields.get("gpu_type", ""),
        port=port,
        env_name=fields.get("env_name", ""),
        extra_args=fields.get("extra_args", ""),
        raw_text=text,
        metadata={"feishu": _extract_feishu_metadata(source, fields)},
    )


def _parse_key_values(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^:=：]+)\s*[:=：]\s*(.+?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace("-", "_")
        value = match.group(2).strip()
        canonical = _KEY_ALIASES.get(key)
        if canonical:
            fields[canonical] = value
    return fields


def _extract_first_path(text: str, allowed_prefixes: tuple[str, ...]) -> str:
    path_pattern = r"(?P<path>(?:/|s3://|oss://|hdfs://|hf://|gs://)[^\s`'\"，,]+)"
    for match in re.finditer(path_pattern, text):
        candidate = match.group("path").strip()
        if _is_allowed_weight_path(candidate, allowed_prefixes):
            return candidate
    return ""


def _is_allowed_weight_path(path: str, allowed_prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for prefix in allowed_prefixes)


def _derive_model_name(weight_path: str) -> str:
    if "://" in weight_path:
        tail = weight_path.rstrip("/").rsplit("/", 1)[-1]
    else:
        tail = PurePosixPath(weight_path).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", tail).strip("-") or "model"


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ParseError(f"{field_name} must be an integer") from exc


def _make_request_id(source: EventSource, weight_path: str, model_name: str) -> str:
    stable = f"{source.source_type.value}:{source.source_ref}:{weight_path}:{model_name}"
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]
    return f"req-{digest}"


def _extract_feishu_metadata(source: EventSource, fields: dict[str, str]) -> dict[str, str]:
    event = source.raw_event.get("event", source.raw_event)
    document_id = (
        fields.get("document_id")
        or event.get("document_id")
        or event.get("obj_token")
        or (source.source_ref if source.source_type.value == "doc" else "")
    )
    status_block_id = (
        fields.get("status_block_id")
        or event.get("status_block_id")
        or event.get("block_id")
    )
    return {
        "document_id": str(document_id or ""),
        "status_block_id": str(status_block_id or ""),
    }
