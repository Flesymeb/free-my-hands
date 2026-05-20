from __future__ import annotations

import shlex
import sqlite3
from dataclasses import dataclass
from typing import Protocol

from fmh.config import AppConfig
from fmh.health import VLLMHealthChecker
from fmh.models import DeploymentRequest, RequestStatus, TERMINAL_STATUSES
from fmh.runner import BaseRunner, DryRunRunner, RunnerError
from fmh.store import StateStore


class NotificationClient(Protocol):
    def send_private_text(self, open_id: str, text: str) -> None:
        ...

    def send_private_card(self, open_id: str, card: dict[str, object]) -> None:
        ...

    def update_document_status(self, request: DeploymentRequest, text: str) -> None:
        ...


@dataclass(frozen=True)
class DeploymentPlan:
    session_name: str
    port: int
    endpoint: str
    rlaunch_command: str
    setup_command: str
    vllm_command: str


class DeploymentOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        store: StateStore,
        runner: BaseRunner,
        notifier: NotificationClient,
    ) -> None:
        self.config = config
        self.store = store
        self.runner = runner
        self.notifier = notifier
        self.health_checker = VLLMHealthChecker(
            timeout_sec=config.vllm.health_timeout_sec,
            interval_sec=config.vllm.health_interval_sec,
        )

    def submit(self, request: DeploymentRequest) -> DeploymentRequest:
        existing = self.store.get_request(request.request_id)
        if existing:
            return existing

        try:
            request = self.store.create_request(request)
        except sqlite3.IntegrityError:
            raced = self.store.get_request(request.request_id)
            if raced:
                return raced
            raise
        plan = self._build_plan(request)
        request.metadata["plan"] = plan.__dict__

        try:
            return self._execute(request, plan)
        except Exception as exc:
            failed = self.store.transition(
                request.request_id,
                RequestStatus.FAILED_VLLM,
                f"unhandled deployment error: {exc}",
                error=str(exc),
            )
            self._notify_final(failed)
            return failed

    def _execute(self, request: DeploymentRequest, plan: DeploymentPlan) -> DeploymentRequest:
        request = self.store.transition(
            request.request_id,
            RequestStatus.LAUNCHING_RESOURCE,
            "starting tmux session and resource allocation",
            tmux_session=plan.session_name,
            port=plan.port,
            endpoint=plan.endpoint,
            metadata=request.metadata,
        )

        try:
            result = self.runner.ensure_session(plan.session_name, self.config.runner.workdir)
        except RunnerError as exc:
            failed = self.store.transition(
                request.request_id,
                RequestStatus.FAILED_RESOURCE,
                f"failed to create tmux session: {exc}",
                error=str(exc),
            )
            self._notify_final(failed)
            return failed

        if plan.rlaunch_command:
            try:
                result = self.runner.run(plan.session_name, "rlaunch", plan.rlaunch_command)
            except RunnerError as exc:
                failed = self.store.transition(
                    request.request_id,
                    RequestStatus.FAILED_RESOURCE,
                    f"rlaunch failed: {exc}",
                    error=str(exc),
                )
                self._notify_final(failed)
                return failed
            raw_output_ref = result.log_path
            summary = "rlaunch command submitted"
        else:
            raw_output_ref = result.log_path
            summary = "no rlaunch command configured; skipped allocation step"

        request = self.store.transition(
            request.request_id,
            RequestStatus.RESOURCE_READY,
            summary,
            raw_output_ref=raw_output_ref,
        )

        if plan.setup_command:
            request = self.store.transition(
                request.request_id,
                RequestStatus.CONFIGURING_ENV,
                "submitting environment setup commands",
            )
            try:
                result = self.runner.run(plan.session_name, "env", plan.setup_command)
            except RunnerError as exc:
                failed = self.store.transition(
                    request.request_id,
                    RequestStatus.FAILED_ENV,
                    f"environment setup failed: {exc}",
                    error=str(exc),
                )
                self._notify_final(failed)
                return failed
            setup_log = result.log_path
        else:
            setup_log = ""

        request = self.store.transition(
            request.request_id,
            RequestStatus.STARTING_VLLM,
            "submitting vLLM server command",
            raw_output_ref=setup_log,
        )
        try:
            result = self.runner.run(plan.session_name, "vllm", plan.vllm_command)
        except RunnerError as exc:
            failed = self.store.transition(
                request.request_id,
                RequestStatus.FAILED_VLLM,
                f"vLLM command failed: {exc}",
                error=str(exc),
            )
            self._notify_final(failed)
            return failed

        if isinstance(self.runner, DryRunRunner):
            complete = self.store.transition(
                request.request_id,
                RequestStatus.DRY_RUN_COMPLETE,
                "dry run completed; no GPU job or vLLM server was started",
                raw_output_ref=result.log_path,
            )
            self._notify_final(complete)
            return complete

        request = self.store.transition(
            request.request_id,
            RequestStatus.HEALTH_CHECKING,
            "waiting for vLLM health check",
            raw_output_ref=result.log_path,
        )
        health = self.health_checker.wait_until_ready(plan.endpoint)
        if not health.ok:
            failed = self.store.transition(
                request.request_id,
                RequestStatus.FAILED_HEALTH_CHECK,
                health.detail,
                error=health.detail,
            )
            self._notify_final(failed)
            return failed

        served = self.store.transition(
            request.request_id,
            RequestStatus.SERVING,
            health.detail,
        )
        self._notify_final(served)
        return served

    def _build_plan(self, request: DeploymentRequest) -> DeploymentPlan:
        port = request.port or self._derive_port(request.request_id)
        session_name = self._session_name(request.request_id)
        endpoint = f"http://{self.config.vllm.public_host}:{port}"
        context = self._command_context(request, port)
        rlaunch_command = self.config.rlaunch.command_template.format_map(context).strip()
        setup_command = " && ".join(self.config.rlaunch.setup_commands).strip()
        vllm_command = self.config.vllm.command_template.format_map(context).strip()
        return DeploymentPlan(
            session_name=session_name,
            port=port,
            endpoint=endpoint,
            rlaunch_command=rlaunch_command,
            setup_command=setup_command,
            vllm_command=vllm_command,
        )

    def _command_context(self, request: DeploymentRequest, port: int) -> dict[str, str | int]:
        extra_args = request.extra_args or self.config.vllm.default_extra_args
        return {
            "request_id": shlex.quote(request.request_id),
            "weight_path": shlex.quote(request.weight_path),
            "model_name": shlex.quote(request.model_name),
            "gpu_count": request.gpu_count,
            "gpu_type": shlex.quote(request.gpu_type),
            "env_name": shlex.quote(request.env_name),
            "host": shlex.quote(self.config.vllm.host),
            "public_host": shlex.quote(self.config.vllm.public_host),
            "port": port,
            "extra_args": _sanitize_extra_args(extra_args),
        }

    def _derive_port(self, request_id: str) -> int:
        suffix = int(request_id.rsplit("-", 1)[-1][:4], 16)
        return self.config.vllm.port_start + suffix % 1000

    def _session_name(self, request_id: str) -> str:
        return f"{self.config.runner.tmux_prefix}-{request_id[-8:]}"

    def _notify_final(self, request: DeploymentRequest) -> None:
        if request.status not in TERMINAL_STATUSES:
            return
        message = format_status_message(request)
        try:
            self.notifier.update_document_status(request, message)
        except Exception:
            pass
        if not _can_private_notify(request.requester.user_id):
            return
        try:
            self.notifier.send_private_card(request.requester.user_id, format_status_card(request))
        except Exception:
            try:
                self.notifier.send_private_text(request.requester.user_id, message)
            except Exception:
                return


def format_status_message(request: DeploymentRequest) -> str:
    lines = [
        f"free-my-hands deployment: {request.status.value}",
        f"request_id: {request.request_id}",
        f"model: {request.model_name}",
        f"weight_path: {request.weight_path}",
    ]
    if request.endpoint:
        lines.append(f"endpoint: {request.endpoint}")
    if request.tmux_session:
        lines.append(f"tmux: {request.tmux_session}")
    if request.error:
        lines.append(f"error: {request.error}")
    return "\n".join(lines)


def format_status_card(request: DeploymentRequest) -> dict[str, object]:
    color = {
        RequestStatus.SERVING: "green",
        RequestStatus.DRY_RUN_COMPLETE: "blue",
        RequestStatus.FAILED_PARSE: "red",
        RequestStatus.FAILED_RESOURCE: "red",
        RequestStatus.FAILED_ENV: "red",
        RequestStatus.FAILED_VLLM: "red",
        RequestStatus.FAILED_HEALTH_CHECK: "red",
        RequestStatus.CANCELLED: "grey",
    }.get(request.status, "blue")

    fields = {
        "request_id": request.request_id,
        "status": request.status.value,
        "model": request.model_name,
        "weight_path": request.weight_path,
        "gpu": f"{request.gpu_count} {request.gpu_type}".strip(),
        "endpoint": request.endpoint or "-",
        "tmux": request.tmux_session or "-",
    }
    if request.error:
        fields["error"] = request.error

    elements: list[dict[str, object]] = [
        {
            "tag": "div",
            "fields": [
                {
                    "is_short": key not in {"weight_path", "error"},
                    "text": {"tag": "lark_md", "content": f"**{key}**\n{value}"},
                }
                for key, value in fields.items()
            ],
        }
    ]
    if request.endpoint:
        elements.extend(
            [
                {"tag": "hr"},
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "Open endpoint"},
                            "type": "primary",
                            "url": request.endpoint,
                        }
                    ],
                },
            ]
        )

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "free-my-hands deployment"},
            "template": color,
        },
        "elements": elements,
    }


def _sanitize_extra_args(extra_args: str) -> str:
    if not extra_args.strip():
        return ""
    return shlex.join(shlex.split(extra_args))


def _can_private_notify(user_id: str) -> bool:
    return bool(user_id and user_id != "unknown" and not user_id.startswith("cli_"))
