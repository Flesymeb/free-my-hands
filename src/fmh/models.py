from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from fmh.time_utils import utc_now_iso


class SourceType(StrEnum):
    DOC = "doc"
    GROUP_MESSAGE = "group_message"
    MANUAL = "manual"


class RequestStatus(StrEnum):
    PENDING = "pending"
    LAUNCHING_RESOURCE = "launching_resource"
    RESOURCE_READY = "resource_ready"
    CONFIGURING_ENV = "configuring_env"
    STARTING_VLLM = "starting_vllm"
    HEALTH_CHECKING = "health_checking"
    SERVING = "serving"
    DRY_RUN_COMPLETE = "dry_run_complete"
    FAILED_PARSE = "failed_parse"
    FAILED_RESOURCE = "failed_resource"
    FAILED_ENV = "failed_env"
    FAILED_VLLM = "failed_vllm"
    FAILED_HEALTH_CHECK = "failed_health_check"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    RequestStatus.SERVING,
    RequestStatus.DRY_RUN_COMPLETE,
    RequestStatus.FAILED_PARSE,
    RequestStatus.FAILED_RESOURCE,
    RequestStatus.FAILED_ENV,
    RequestStatus.FAILED_VLLM,
    RequestStatus.FAILED_HEALTH_CHECK,
    RequestStatus.CANCELLED,
}


@dataclass
class Requester:
    user_id: str
    display_name: str = ""

    @classmethod
    def unknown(cls) -> "Requester":
        return cls(user_id="unknown")


@dataclass
class DeploymentRequest:
    request_id: str
    source_type: SourceType
    source_ref: str
    requester: Requester
    weight_path: str
    model_name: str
    gpu_count: int = 1
    gpu_type: str = ""
    port: int | None = None
    env_name: str = ""
    extra_args: str = ""
    raw_text: str = ""
    status: RequestStatus = RequestStatus.PENDING
    tmux_session: str = ""
    endpoint: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentRequest":
        payload = dict(data)
        payload["source_type"] = SourceType(payload["source_type"])
        payload["status"] = RequestStatus(payload["status"])
        payload["requester"] = Requester(**payload["requester"])
        return cls(**payload)


@dataclass
class StateEvent:
    request_id: str
    state_from: str
    state_to: str
    summary: str
    raw_output_ref: str = ""
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventSource:
    source_type: SourceType
    source_ref: str
    requester: Requester
    text: str
    raw_event: dict[str, Any] = field(default_factory=dict)
