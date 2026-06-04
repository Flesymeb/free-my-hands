from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fmh.config import AppConfig
from fmh.deployed_doc import update_deployed_models_row
from fmh.feishu import FeishuOpenAPIClient, NullFeishuClient
from fmh.operator_review import mention_text, review_result_card
from fmh.reusable_workers import (
    DeployedModelRow,
    ModelPathInfo,
    build_plan_for_row,
    choose_reusable_row,
    is_reusable_worker_state,
    parse_deployed_models_table,
    reuse_flag_allows_scan,
)
from fmh.store import StateStore
from fmh.task_status import task_status_card, task_status_with_stage
from fmh.time_utils import utc_now_iso
from fmh.weight_conversion import run_weight_conversion

log = logging.getLogger(__name__)
_DEPLOYED_DOC_WRITE_LOCK = threading.Lock()
_WEIGHT_CONVERSION_LOCK = threading.Lock()


class ReusableDeploymentError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


class ReusableDeploymentExecutor:
    def __init__(
        self,
        config: AppConfig,
        store: StateStore,
        feishu_client: FeishuOpenAPIClient | NullFeishuClient,
    ) -> None:
        self.config = config
        self.store = store
        self.feishu = feishu_client

    def execute_if_enabled(self, review: dict[str, Any], decision: dict[str, Any]) -> bool:
        if not self.config.reusable_workers.auto_deploy_approved:
            return False
        if str(decision.get("decision") or "").upper() != "APPROVE":
            return False

        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        if payload.get("stage") != "reuse_row_selected":
            return False

        review_id = str(review.get("review_id") or "")
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        if not review_id or not isinstance(plan, dict) or not isinstance(row, dict):
            return False

        if not _row_can_auto_reuse(row, self.config):
            self._mark_needs_human(
                review,
                decision,
                "选中的 worker 不满足自动复用条件：空闲行可用，或 required tasks 已完成且无 running 标记；刚部署未测试的行不可自动停。",
            )
            return False

        approval_summary = str(decision.get("approval_summary") or decision.get("summary") or "审核通过。")
        deploy_decision = {
            **decision,
            "approval_summary": approval_summary,
            "deploy_status": "deploying",
            "deploy_started_at": utc_now_iso(),
        }
        self.store.decide_review(review_id, "deploying", deploy_decision)
        updated = self.store.get_review(review_id) or review
        self._mark_task_entry_status(updated, "deploying", summary=f"deploying {review_id}")
        self._update_task_status(updated, deploy_decision)

        try:
            result = self._execute(review_id, plan)
        except Exception as exc:
            log.exception("reusable deployment failed: %s", review_id)
            error_text = str(exc)
            error_class = _classify_deployment_error(error_text)
            failure_summary = f"真实部署失败（{error_class['label']}）：{error_text}"
            failed_decision = {
                **decision,
                "approval_summary": approval_summary,
                "deploy_status": "failed",
                "summary": failure_summary,
                "execution_summary": failure_summary,
                "error": error_text,
                "error_type": error_class["type"],
                "recommended_action": error_class["action"],
                "deploy_completed_at": utc_now_iso(),
            }
            self.store.decide_review(review_id, "deploy_failed", failed_decision)
            updated = self.store.get_review(review_id) or review
            self._mark_task_entry_status(updated, "deploy_failed", summary=failure_summary)
            self._send_card(updated, failed_decision)
            return True

        document_error = ""
        try:
            self._write_table_values(plan, "final_table_values")
        except Exception as exc:
            log.exception("failed to write final deployed-model table values: %s", review_id)
            document_error = str(exc)
        success_decision = {
            **decision,
            "approval_summary": approval_summary,
            "deploy_status": "deployed",
            "summary": f"真实部署完成：{result['model_id']} 已在 {result['endpoint']} 可见。",
            "execution_summary": f"真实部署完成：{result['model_id']} 已在 {result['endpoint']} 可见。",
            "deploy_completed_at": utc_now_iso(),
            "endpoint": result["endpoint"],
            "worker": result["worker"],
            "model_id": result["model_id"],
        }
        if document_error:
            success_decision = {
                **success_decision,
                "document_status": "failed",
                "document_error": document_error,
                "summary": (
                    f"真实部署完成：{result['model_id']} 已在 {result['endpoint']} 可见；"
                    f"但文档回填失败：{_short(document_error, 180)}"
                ),
            }
        self.store.decide_review(review_id, "deployed", success_decision)
        updated = self.store.get_review(review_id) or review
        self._mark_task_entry_status(updated, "deployed", summary=success_decision["summary"])
        final_decision = self._notify_post_deploy_bot(updated, success_decision)
        final_decision = self._notify_manual_subtask_completion_if_present(updated, final_decision)
        if final_decision is not success_decision:
            self.store.decide_review(review_id, "deployed", final_decision)
            updated = self.store.get_review(review_id) or updated
        self._send_card(updated, final_decision)
        return True

    def _execute(self, review_id: str, plan: dict[str, Any]) -> dict[str, str]:
        path = _expect_dict(plan, "path")
        self._refresh_plan_row_from_doc(plan, path)
        row = _expect_dict(plan, "row")
        ip = _expect_str(row, "ip")
        model_id = _expect_str(path, "model_id")
        session = _expect_str(plan, "tmux_session_guess")
        port = int(self.config.reusable_workers.default_port)
        endpoint = f"http://{ip}:{port}"

        if _row_already_serves_model(row, path, self.config.reusable_workers.deploying_marker):
            served_ids = self._endpoint_model_ids(endpoint)
            if model_id in served_ids:
                return {"worker": ip, "model_id": model_id, "endpoint": endpoint}

        conversion = plan.get("weight_conversion") if isinstance(plan.get("weight_conversion"), dict) else None
        if conversion and conversion.get("required", True):
            self._run_weight_conversion(conversion)
            self._mark_conversion_done(review_id, conversion)
        direct_worker_available = self._preflight(ip, session, plan, endpoint)
        self._write_table_values(plan, "deploying_table_values")
        self._ensure_worker_tmux_connected(ip, session)
        self._stop_existing_vllm(ip, session, endpoint)
        if direct_worker_available is False:
            self._verify_worker_path_after_stop(ip, session, plan)
        vllm_command = _expect_str(plan, "vllm_command")
        self._ensure_worker_tmux_connected(ip, session)
        self._send_vllm_command(review_id, session, vllm_command)
        self._wait_until_serving(session, endpoint, model_id)
        return {"worker": ip, "model_id": model_id, "endpoint": endpoint}

    def _run_weight_conversion(self, conversion: dict[str, Any]) -> None:
        with _WEIGHT_CONVERSION_LOCK:
            run_weight_conversion(conversion, self.config.weight_conversion)

    def _ensure_worker_tmux_connected(self, ip: str, session: str) -> None:
        if self._worker_tmux_connected(session):
            return
        target = _worker_tmux_target(session)
        reconnect = _worker_reconnect_command(ip)
        setup = _worker_shell_setup_command()
        self._run_dev(f"tmux send-keys -t {shlex.quote(target)} C-c", timeout=20, check=False)
        self._run_dev(
            f"tmux send-keys -t {shlex.quote(target)} {shlex.quote(reconnect)} C-m",
            timeout=20,
            check=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            time.sleep(2)
            if self._worker_tmux_connected(session):
                self._run_dev(
                    f"tmux send-keys -t {shlex.quote(target)} {shlex.quote(setup)} C-m",
                    timeout=20,
                    check=True,
                )
                return
        pane = self._capture_pane(session)
        raise ReusableDeploymentError(f"failed to reconnect worker ssh in tmux session {session}\n{_short(pane, 800)}")

    def _worker_tmux_connected(self, session: str) -> bool:
        result = self._run_dev(
            f"tmux list-panes -t {shlex.quote(_worker_tmux_target(session))} -F '#{{pane_current_command}}'",
            timeout=20,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().splitlines()[:1] == ["ssh"]

    def _mark_conversion_done(self, review_id: str, conversion: dict[str, Any]) -> None:
        review = self.store.get_review(review_id)
        if not review:
            return
        output_path = str(conversion.get("output_path") or "")
        self._update_task_status(
            review,
            {
                "decision": "APPROVE",
                "deploy_status": "conversion_done",
                "summary": _conversion_done_summary(output_path),
                "execution_summary": "转换完成，正在进入 tmux 启动 vLLM。",
            },
        )

    def _preflight(self, ip: str, session: str, plan: dict[str, Any], endpoint: str) -> bool:
        path = _expect_dict(plan, "path")
        worker_path = _expect_str(path, "worker_path")
        self._run_dev(f"tmux has-session -t {shlex.quote(session)}", timeout=20, check=True)
        windows = self._run_dev(
            f"tmux list-windows -t {shlex.quote(session)} -F '#{{window_index}}:#{{window_name}}:#{{pane_current_command}}'",
            timeout=20,
            check=True,
        ).stdout.splitlines()
        if not _has_worker_window(windows):
            raise ReusableDeploymentError(f"tmux session {session} has no worker ssh window")

        path_check = self._run_worker(
            ip,
            _model_dir_probe_command(worker_path),
            timeout=20,
            check=False,
        )
        if path_check.returncode != 0 and _should_fallback_worker_to_tmux(path_check):
            log.warning("direct worker ssh unavailable for %s; will use tmux worker pane after stopping vLLM", ip)
            return False
        if path_check.returncode != 0:
            detail = "\n".join(part for part in (path_check.stdout, path_check.stderr) if part)
            raise ReusableDeploymentError(f"remote command failed ({path_check.returncode}): {_short(detail, 800)}")
        self._apply_model_dir_probe(plan, path_check.stdout)
        return True

    def _verify_worker_path_after_stop(self, ip: str, session: str, plan: dict[str, Any]) -> None:
        path = _expect_dict(plan, "path")
        worker_path = _expect_str(path, "worker_path")
        path_check = self._run_worker(
            ip,
            _model_dir_probe_command(worker_path),
            timeout=30,
            check=True,
            tmux_session=session,
        ).stdout
        self._apply_model_dir_probe(plan, path_check)

    def _apply_model_dir_probe(self, plan: dict[str, Any], output: str) -> None:
        path = _expect_dict(plan, "path")
        worker_path = _expect_str(path, "worker_path")
        status, resolved_path, detail = _parse_model_dir_probe(output)
        if status == "OK":
            return
        if status == "RESOLVED" and resolved_path:
            self._replace_plan_worker_path(plan, resolved_path)
            return
        if status == "MISSING":
            raise ReusableDeploymentError(f"model path missing on worker: {worker_path}")
        if status == "AMBIGUOUS":
            raise ReusableDeploymentError(
                "model path is not a loadable HF directory and has multiple candidate child model dirs: "
                f"{detail}"
            )
        raise ReusableDeploymentError(
            "model path is not a loadable HF directory: "
            f"{worker_path}. Expected config.json/params.json in that directory, or exactly one child model directory."
        )

    def _replace_plan_worker_path(self, plan: dict[str, Any], resolved_path: str) -> None:
        path = _expect_dict(plan, "path")
        old_worker_path = _expect_str(path, "worker_path")
        table_path = _table_path_for_worker_path(resolved_path, self.config)
        model_id = _expect_str(path, "model_id")
        path["worker_path"] = resolved_path
        path["table_path"] = table_path
        plan["vllm_command"] = _replace_vllm_model_arg(_expect_str(plan, "vllm_command"), old_worker_path, resolved_path)
        final_values = plan.get("final_table_values") if isinstance(plan.get("final_table_values"), dict) else {}
        deploying_values = plan.get("deploying_table_values") if isinstance(plan.get("deploying_table_values"), dict) else {}
        final_values["模型"] = table_path
        final_values["模型id"] = model_id
        deploying_values["模型"] = f"{table_path}{self.config.reusable_workers.deploying_marker}"
        deploying_values["模型id"] = f"{model_id}{self.config.reusable_workers.deploying_marker}"
        plan["final_table_values"] = final_values
        plan["deploying_table_values"] = deploying_values

    def _stop_existing_vllm(self, ip: str, session: str, endpoint: str) -> None:
        target = _worker_tmux_target(session)
        stop_signal = self._run_dev(
            f"tmux send-keys -t {shlex.quote(target)} C-c",
            timeout=60,
            check=False,
        )
        if stop_signal.returncode != 0:
            log.warning(
                "failed to send tmux interrupt to %s before fallback cleanup: %s",
                target,
                _short("\n".join(part for part in (stop_signal.stdout, stop_signal.stderr) if part), 300),
            )
        if self._wait_endpoint_down(endpoint, timeout_sec=45):
            self._kill_leftover_gpu_apps(ip, session)
            return

        port = int(self.config.reusable_workers.default_port)
        self._run_worker(ip, _kill_port_listeners_command(port), timeout=30, check=False, tmux_session=session)
        if not self._wait_endpoint_down(endpoint, timeout_sec=30):
            served_ids = self._endpoint_model_ids(endpoint)
            detail = f" endpoint still serving model ids {served_ids}" if served_ids else ""
            raise ReusableDeploymentError(f"failed to stop existing vLLM on {endpoint}.{detail}")
        self._kill_leftover_gpu_apps(ip, session)

    def _wait_endpoint_down(self, endpoint: str, *, timeout_sec: int) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            result = self._run_dev(
                f"curl -sS --max-time 3 {shlex.quote(endpoint + '/v1/models')}",
                timeout=8,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return True
            time.sleep(3)
        return False

    def _kill_leftover_gpu_apps(self, ip: str, session: str) -> None:
        gpu_apps = self._run_worker(ip, _nvidia_compute_apps_command(), timeout=30, check=False, tmux_session=session)
        if gpu_apps.returncode != 0:
            gpu_output = "\n".join(part for part in (gpu_apps.stdout, gpu_apps.stderr) if part).strip()
            raise ReusableDeploymentError(f"failed to query worker GPU apps after stopping vLLM: {_short(gpu_output, 400)}")
        pids = _gpu_app_pids(gpu_apps.stdout)
        if not pids:
            return
        self._run_worker(ip, _kill_pids_command(pids), timeout=30, check=False, tmux_session=session)
        time.sleep(5)
        remaining = self._run_worker(ip, _nvidia_compute_apps_command(), timeout=30, check=False, tmux_session=session)
        if remaining.returncode != 0:
            gpu_output = "\n".join(part for part in (remaining.stdout, remaining.stderr) if part).strip()
            raise ReusableDeploymentError(f"failed to verify worker GPU apps after cleanup: {_short(gpu_output, 400)}")
        remaining_output = remaining.stdout.strip()
        if remaining_output:
            raise ReusableDeploymentError(f"worker GPUs are still busy after stopping vLLM: {_short(remaining_output, 400)}")

    def _endpoint_model_ids(self, endpoint: str) -> list[str]:
        result = self._run_dev(
            f"curl -sS --max-time 5 {shlex.quote(endpoint + '/v1/models')}",
            timeout=10,
            check=False,
        )
        output = result.stdout.strip()
        if result.returncode != 0 or not output:
            return []
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []
        return [
            str(item.get("id") or "")
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    def _send_vllm_command(self, review_id: str, session: str, vllm_command: str) -> None:
        target = _worker_tmux_target(session)
        command = (
            "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
            "/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}; "
            f"echo FMH_DEPLOY_START {shlex.quote(review_id)} $(date -Is); "
            f"{vllm_command}"
        )
        remote = (
            f"tmux send-keys -t {shlex.quote(target)} {shlex.quote(command)} C-m"
        )
        self._run_dev(remote, timeout=20, check=True)

    def _wait_until_serving(self, session: str, endpoint: str, model_id: str) -> None:
        deadline = time.time() + self.config.reusable_workers.deploy_timeout_sec
        wrong_model_seen = 0
        last_detail = ""
        while time.time() < deadline:
            time.sleep(self.config.reusable_workers.deploy_poll_interval_sec)
            result = self._run_dev(
                f"curl -sS --max-time 8 {shlex.quote(endpoint + '/v1/models')}",
                timeout=15,
                check=False,
            )
            output = result.stdout.strip()
            if result.returncode != 0 or not output:
                last_detail = (result.stderr or output).strip()
                continue
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                last_detail = _short(output, 400)
                continue
            ids = [
                str(item.get("id") or "")
                for item in data.get("data", [])
                if isinstance(item, dict)
            ]
            if model_id in ids:
                return
            if ids:
                wrong_model_seen += 1
                last_detail = f"endpoint returned model ids {ids}, expected {model_id}"
                if wrong_model_seen >= 3:
                    raise ReusableDeploymentError(last_detail)

        pane = self._capture_pane(session)
        detail = last_detail or "endpoint did not become ready before timeout"
        raise ReusableDeploymentError(f"{detail}\n{_short(pane, 2000)}")

    def _capture_pane(self, session: str) -> str:
        result = self._run_dev(
            f"tmux capture-pane -pt {shlex.quote(_worker_tmux_target(session))} -S -120",
            timeout=20,
            check=False,
        )
        return "\n".join(part for part in (result.stdout, result.stderr) if part)

    def _run_worker(
        self,
        ip: str,
        command: str,
        *,
        timeout: int,
        check: bool,
        tmux_session: str = "",
    ) -> RemoteResult:
        result = self._run_dev(_worker_command(ip, command), timeout=timeout, check=False)
        if result.returncode != 0 and tmux_session and _should_fallback_worker_to_tmux(result):
            result = self._run_worker_pane(tmux_session, command, timeout=timeout)
        if check and result.returncode != 0:
            detail = "\n".join(part for part in (result.stdout, result.stderr) if part)
            raise ReusableDeploymentError(f"remote command failed ({result.returncode}): {_short(detail, 800)}")
        return result

    def _run_worker_pane(self, session: str, command: str, *, timeout: int) -> RemoteResult:
        token = f"FMH_{uuid.uuid4().hex}"
        begin = f"__{token}_BEGIN__"
        err = f"__{token}_ERR__"
        end = f"__{token}_END__"
        out_path = f"/tmp/{token}.out"
        err_path = f"/tmp/{token}.err"
        script = (
            f"({command}) > {shlex.quote(out_path)} 2> {shlex.quote(err_path)}; "
            "rc=$?; "
            f"printf '\\n{begin}\\n'; cat {shlex.quote(out_path)}; "
            f"printf '\\n{err}\\n'; cat {shlex.quote(err_path)}; "
            f"printf '\\n{end}:%s\\n' \"$rc\"; "
            f"rm -f {shlex.quote(out_path)} {shlex.quote(err_path)}"
        )
        wrapped = "bash -lc " + shlex.quote(script)
        target = _worker_tmux_target(session)
        self._run_dev(
            f"tmux send-keys -t {shlex.quote(target)} {shlex.quote(wrapped)} C-m",
            timeout=20,
            check=True,
        )
        deadline = time.time() + timeout
        last_capture = ""
        while time.time() < deadline:
            capture = self._run_dev(
                f"tmux capture-pane -pt {shlex.quote(target)} -S -2000",
                timeout=20,
                check=False,
            )
            last_capture = capture.stdout
            parsed = _parse_tmux_worker_result(command, last_capture, begin, err, end)
            if parsed is not None:
                return parsed
            time.sleep(1)
        return RemoteResult(
            command=command,
            returncode=124,
            stdout=last_capture,
            stderr=f"worker tmux command timed out waiting for {end}",
        )

    def _run_dev(self, command: str, *, timeout: int, check: bool) -> RemoteResult:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-CAXY",
            self.config.reusable_workers.dev_host,
            command,
        ]
        try:
            result = subprocess.run(
                args,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_output_to_text(exc.stdout)
            stderr = _timeout_output_to_text(exc.stderr)
            detail = f"remote command timed out after {timeout}s: {command}"
            remote = RemoteResult(
                command=command,
                returncode=124,
                stdout=stdout,
                stderr="\n".join(part for part in (stderr, detail) if part),
            )
            if check:
                raise ReusableDeploymentError(detail)
            return remote
        remote = RemoteResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if check and result.returncode != 0:
            detail = "\n".join(part for part in (result.stdout, result.stderr) if part)
            raise ReusableDeploymentError(f"remote command failed ({result.returncode}): {_short(detail, 800)}")
        return remote

    def _write_table_values(self, plan: dict[str, Any], key: str) -> None:
        values = plan.get(key) if isinstance(plan.get(key), dict) else {}
        path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
        if path:
            self._refresh_plan_row_from_doc(plan, path)
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        row_index = int(row.get("row_index") or 0)
        if not values or not row_index or isinstance(self.feishu, NullFeishuClient):
            return
        with _DEPLOYED_DOC_WRITE_LOCK:
            update_deployed_models_row(
                self.feishu,
                self.config.reusable_workers,
                row_index=row_index,
                values={str(k): str(v) for k, v in values.items()},
            )

    def _refresh_plan_row_from_doc(self, plan: dict[str, Any], path: dict[str, Any]) -> None:
        if isinstance(self.feishu, NullFeishuClient):
            return
        get_doc_markdown = getattr(self.feishu, "get_doc_markdown", None)
        if not callable(get_doc_markdown):
            return
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        ip = str(row.get("ip") or "").strip()
        row_index = int(row.get("row_index") or 0)
        if not ip:
            return
        markdown = str(get_doc_markdown(self.config.reusable_workers.deployed_models_doc_token) or "")
        rows = parse_deployed_models_table(markdown)
        current = next((candidate for candidate in rows if candidate.row_index == row_index), None)
        if current and current.ip == ip:
            self._set_plan_row_if_safe(plan, current, path)
            return
        replacement = next((candidate for candidate in rows if candidate.ip == ip), None)
        if replacement is None:
            replacement = choose_reusable_row(
                rows,
                self.config.reusable_workers,
                required_gpu_count=int(row.get("gpu_count") or 0),
            )
            if replacement is None:
                raise ReusableDeploymentError(
                    f"selected worker {ip} is no longer present in deployed-models document and no replacement worker is reusable"
                )
            self._replace_plan_row(plan, replacement, path)
            return
        self._set_plan_row_if_safe(plan, replacement, path)

    def _set_plan_row_if_safe(self, plan: dict[str, Any], row: DeployedModelRow, path: dict[str, Any]) -> None:
        row_dict = row.to_dict(self.config.reusable_workers)
        if not _current_row_safe_for_plan(row_dict, path, self.config):
            raise ReusableDeploymentError(
                "selected worker row changed in deployed-models document and is no longer safe to reuse: "
                f"{row.ip} row {row.row_index}"
            )
        original = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        plan["row"] = {**original, **row_dict}

    def _replace_plan_row(self, plan: dict[str, Any], row: DeployedModelRow, path: dict[str, Any]) -> None:
        replacement = build_plan_for_row(_model_path_info_from_plan_path(path), row, self.config.reusable_workers).to_dict()
        preserved = {key: value for key, value in plan.items() if key not in replacement}
        plan.clear()
        plan.update(replacement)
        plan.update(preserved)

    def _mark_needs_human(self, review: dict[str, Any], decision: dict[str, Any], reason: str) -> None:
        review_id = str(review.get("review_id") or "")
        human_decision = {
            **decision,
            "deploy_status": "needs_human",
            "summary": reason,
            "decided_at": utc_now_iso(),
        }
        self.store.decide_review(review_id, "needs_human", human_decision)
        updated = self.store.get_review(review_id) or review
        self._mark_task_entry_status(updated, "needs_human", summary=reason)
        self._send_card(updated, human_decision)

    def _mark_task_entry_status(self, review: dict[str, Any], status: str, *, summary: str = "") -> None:
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        task_key = str(context.get("task_key") or "")
        item_key = str(context.get("item_key") or "")
        if not task_key or not item_key:
            return
        self.store.mark_processed_item(
            task_key,
            item_key,
            status,
            request_id=str(review.get("review_id") or ""),
            summary=summary,
        )

    def _send_card(self, review: dict[str, Any], decision: dict[str, Any]) -> None:
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        reply_to = str(context.get("reply_to_message_id") or "")
        source_chat_id = str(context.get("source_chat_id") or "")
        target_chat_id = source_chat_id or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        if not target_chat_id and not reply_to:
            return
        simple_text = _simple_status_text(review, decision)
        alert_text = _human_alert_text(self.config, review, decision)
        source_updated = self._update_task_status(review, decision)
        try:
            if source_updated:
                if alert_text:
                    if reply_to:
                        self.feishu.reply_text(reply_to, alert_text)
                    elif target_chat_id:
                        self.feishu.send_chat_text(target_chat_id, alert_text)
            elif reply_to:
                self.feishu.reply_card(reply_to, review_result_card(review, decision))
                if alert_text:
                    self.feishu.reply_text(reply_to, alert_text)
            else:
                self.feishu.send_chat_card(target_chat_id, review_result_card(review, decision))
                if alert_text:
                    self.feishu.send_chat_text(target_chat_id, alert_text)
        except Exception:
            fallback_text = alert_text or simple_text
            if reply_to:
                self.feishu.reply_text(reply_to, fallback_text)
            elif target_chat_id:
                self.feishu.send_chat_text(target_chat_id, fallback_text)

    def _update_task_status(self, review: dict[str, Any], decision: dict[str, Any]) -> bool:
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        task_key = str(context.get("status_task_key") or context.get("task_key") or "")
        source_chat_id = str(context.get("source_chat_id") or "")
        message_id = str(context.get("status_message_id") or "")
        if not task_key or not source_chat_id:
            return False
        state = self.store.get_task_status(task_key)
        if message_id and str(state.get("source_message_id") or "") not in {"", message_id}:
            state = {}
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        model_id = str(decision.get("model_id") or path.get("model_id") or _subject_model(str(review.get("subject_id") or "")))
        model = str(path.get("table_path") or path.get("original_path") or path.get("worker_path") or "")
        worker = str(decision.get("worker") or row.get("ip") or _subject_worker(str(review.get("subject_id") or "")))
        address = _row_address(row)
        endpoint = str(decision.get("endpoint") or "")
        deploy_status = str(decision.get("deploy_status") or review.get("status") or "")
        execution_summary = _short(
            str(decision.get("execution_summary") or decision.get("summary") or decision.get("error") or ""),
            180,
        )
        approval_summary = _approval_card_summary(decision, payload)
        title = str(context.get("task_title") or state.get("title") or payload.get("title") or "")
        review_id = str(review.get("review_id") or "")

        stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
        has_codex_stage = isinstance(stages.get("codex"), dict)
        if str(decision.get("decision") or "").upper() == "APPROVE" and (
            not has_codex_stage or deploy_status == "deploying"
        ):
            state = task_status_with_stage(
                state,
                "codex",
                "通过",
                approval_summary,
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
            )

        if deploy_status == "deploying":
            if isinstance(plan.get("weight_conversion"), dict):
                conversion = plan["weight_conversion"]
                state = task_status_with_stage(
                    state,
                    "convert",
                    "进行中",
                    f"正在转换到 {_short(str(conversion.get('output_path') or ''), 120)}。",
                    title=title,
                    source_chat_id=source_chat_id,
                    source_message_id=message_id,
                    model_id=model_id,
                    model=model,
                    worker=worker,
                    address=address,
                )
            else:
                state = task_status_with_stage(
                    state,
                    "execute",
                    "进行中",
                    "已进入 tmux，正在启动 vLLM。",
                    title=title,
                    source_chat_id=source_chat_id,
                    source_message_id=message_id,
                    model_id=model_id,
                    model=model,
                    worker=worker,
                    address=address,
                )
        elif deploy_status == "conversion_done":
            state = task_status_with_stage(
                state,
                "convert",
                "完成",
                _conversion_card_summary(str(decision.get("summary") or "权重转换完成。")),
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
            )
            state = task_status_with_stage(
                state,
                "execute",
                "进行中",
                str(decision.get("execution_summary") or "转换完成，正在进入 tmux 启动 vLLM。"),
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
            )
        elif deploy_status == "deployed":
            state = task_status_with_stage(
                state,
                "execute",
                "完成",
                "vLLM 已通过 /v1/models 校验。",
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
                endpoint=endpoint,
            )
            if str(decision.get("document_status") or "") == "failed":
                state = task_status_with_stage(
                    state,
                    "document",
                    "需人工",
                    _short(str(decision.get("document_error") or "文档回填失败。"), 180),
                    endpoint=endpoint,
                )
            else:
                state = task_status_with_stage(state, "document", "完成", "已回填已部署模型文档。", endpoint=endpoint)
        elif deploy_status in {"failed", "deploy_failed"}:
            failed_stage = "execute"
            if isinstance(plan.get("weight_conversion"), dict):
                convert_stage = stages.get("convert") if isinstance(stages.get("convert"), dict) else {}
                if str(convert_stage.get("status") or "") != "完成":
                    failed_stage = "convert"
            state = task_status_with_stage(
                state,
                failed_stage,
                "失败",
                execution_summary,
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
            )
        elif deploy_status == "needs_human":
            state = task_status_with_stage(
                state,
                "execute",
                "需人工",
                execution_summary,
                title=title,
                source_chat_id=source_chat_id,
                source_message_id=message_id,
                model_id=model_id,
                model=model,
                worker=worker,
                address=address,
            )

        if review_id:
            state["review_id"] = review_id
        state["card_actions_enabled"] = self.config.approval.allow_card_actions
        message_id = str(state.get("source_message_id") or "")
        if not message_id:
            self.store.set_task_status(task_key, state)
            return False
        try:
            self.feishu.update_card(message_id, task_status_card(state))
        except Exception:
            log.exception("failed to update source task status card")
            self.store.set_task_status(task_key, state)
            return False
        self.store.set_task_status(task_key, state)
        return True

    def _notify_post_deploy_bot(self, review: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        notify = self.config.post_deploy_notify
        if not notify.enabled or not notify.target_open_id or isinstance(self.feishu, NullFeishuClient):
            return decision
        if str(decision.get("deploy_status") or "") != "deployed":
            return decision

        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}

        chat_id = str(context.get("source_chat_id") or "") or notify.chat_id or self.config.feishu.default_chat_id
        if not chat_id:
            return decision

        target_name = notify.target_name or notify.target_open_id
        worker = str(decision.get("worker") or row.get("ip") or _subject_worker(str(review.get("subject_id") or "")))
        model_id = str(decision.get("model_id") or path.get("model_id") or _subject_model(str(review.get("subject_id") or "")))
        endpoint = str(decision.get("endpoint") or "")
        task_title = str(context.get("task_title") or payload.get("title") or "")
        values = {
            "target_open_id": notify.target_open_id,
            "target_name": target_name,
            "worker": worker,
            "model_id": model_id,
            "endpoint": endpoint,
            "task_title": task_title,
        }
        try:
            if notify.card_enabled:
                message_id = self.feishu.send_chat_card(chat_id, _post_deploy_notify_card(values))
            else:
                message_id = self.feishu.send_chat_text(chat_id, notify.message_template.format(**values))
        except Exception as exc:
            log.exception("failed to notify post-deploy bot")
            if notify.card_enabled:
                try:
                    message_id = self.feishu.send_chat_text(chat_id, notify.message_template.format(**values))
                    return {
                        **decision,
                        "post_deploy_notify_status": "sent_text_fallback",
                        "post_deploy_notify_message_id": message_id,
                        "post_deploy_notify_chat_id": chat_id,
                    }
                except Exception:
                    log.exception("failed to send text fallback for post-deploy bot")
            return {
                **decision,
                "post_deploy_notify_status": "failed",
                "post_deploy_notify_error": str(exc),
            }
        return {
            **decision,
            "post_deploy_notify_status": "sent",
            "post_deploy_notify_message_id": message_id,
            "post_deploy_notify_chat_id": chat_id,
        }

    def _notify_manual_subtask_completion_if_present(
        self,
        review: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(self.feishu, NullFeishuClient):
            return decision
        if str(decision.get("deploy_status") or "") != "deployed":
            return decision
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        subtask_guid = str(context.get("subtask_guid") or "")
        if not subtask_guid:
            return decision
        chat_id = str(context.get("source_chat_id") or "") or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        if not chat_id:
            return decision

        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
        mention = mention_text(self.config.approval)
        model_id = str(decision.get("model_id") or path.get("model_id") or _subject_model(str(review.get("subject_id") or "")))
        task_title = str(context.get("task_title") or payload.get("title") or "")
        prefix = f"{mention} " if mention else ""
        lines = [
            f"{prefix}模型已部署，请手动完成飞书子任务。",
            f"模型: {model_id}",
        ]
        if task_title:
            lines.append(f"来源任务: {task_title}")
        try:
            message_id = self.feishu.send_chat_text(chat_id, "\n".join(lines))
        except Exception as exc:
            log.exception("failed to send manual subtask completion notice")
            return {
                **decision,
                "manual_subtask_completion_notice_status": "failed",
                "manual_subtask_completion_notice_error": str(exc),
            }
        return {
            **decision,
            "manual_subtask_completion_notice_status": "sent",
            "manual_subtask_completion_notice_message_id": message_id,
            "manual_subtask_completion_notice_chat_id": chat_id,
        }


def _post_deploy_notify_card(values: dict[str, str]) -> dict[str, Any]:
    task_title = values.get("task_title", "")
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    f'<at id="{values["target_open_id"].replace(chr(34), "")}"></at> '
                    "**新模型已部署，请处理**"
                ),
            },
        },
        {"tag": "hr"},
        {
            "tag": "div",
            "fields": [
                _card_field("worker", values.get("worker", "")),
                _card_field("endpoint", values.get("endpoint", "")),
                _card_field("model_id", values.get("model_id", "")),
            ],
        },
    ]
    if task_title:
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"来源任务：{task_title}",
                    }
                ],
            }
        )
    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "新模型待处理"},
            "template": "green",
        },
        "elements": elements,
    }


def _card_field(label: str, value: str) -> dict[str, Any]:
    return {
        "is_short": True,
        "text": {"tag": "lark_md", "content": f"**{label}** {_md_escape(value)}"},
    }


def _worker_command(ip: str, command: str) -> str:
    opts = (
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null "
        "-o BatchMode=yes "
        "-o ConnectTimeout=8 "
        "-o UpdateHostKeys=no "
        "-o ServerAliveInterval=60 "
        "-o ServerAliveCountMax=3"
    )
    return f"ssh {opts} {shlex.quote(ip)} {shlex.quote(command)}"


def _worker_tmux_target(session: str) -> str:
    return f"{session}:0"


def _worker_reconnect_command(ip: str) -> str:
    return shlex.join(
        [
            "ssh",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "UpdateHostKeys=no",
            ip,
        ]
    )


def _worker_shell_setup_command() -> str:
    return (
        "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}; "
        "export PATH=/usr/local/nvidia/bin:${PATH:-}; "
        "sed -i 's/ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {\"partial_rotary_factor\"}/"
        "ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation or []) | {\"partial_rotary_factor\"}/' "
        "/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py 2>/dev/null || true"
    )


def _model_dir_probe_command(worker_path: str) -> str:
    return f"""DIR={shlex.quote(worker_path)}
if [ ! -d "$DIR" ]; then
  printf 'MISSING\\n'
  exit 0
fi
if [ -f "$DIR/config.json" ] || [ -f "$DIR/params.json" ]; then
  printf 'OK\\t%s\\n' "$DIR"
  exit 0
fi
candidates=$(
  find "$DIR" -mindepth 1 -maxdepth 2 -type f \\( -name config.json -o -name params.json \\) -print 2>/dev/null |
  while IFS= read -r cfg; do
    model_dir=$(dirname "$cfg")
    if find "$model_dir" -maxdepth 1 -type f \\( -name '*.safetensors' -o -name 'model.safetensors.index.json' -o -name 'pytorch_model*.bin' \\) -print -quit 2>/dev/null | grep -q .; then
      printf '%s\\n' "$model_dir"
    fi
  done | sort -u | sed -n '1,2p'
)
count=$(printf '%s\\n' "$candidates" | sed '/^$/d' | wc -l | tr -d ' ')
if [ "$count" = "1" ]; then
  printf 'RESOLVED\\t%s\\n' "$candidates"
elif [ "$count" = "0" ]; then
  printf 'INVALID\\n'
else
  printf 'AMBIGUOUS\\t%s\\n' "$(printf '%s' "$candidates" | paste -sd ',' -)"
fi"""


def _parse_model_dir_probe(output: str) -> tuple[str, str, str]:
    line = next((raw.strip() for raw in output.splitlines() if raw.strip()), "")
    if not line:
        return "INVALID", "", ""
    parts = line.split("\t", 2)
    status = parts[0].strip().upper()
    resolved = parts[1].strip() if len(parts) > 1 else ""
    detail = parts[2].strip() if len(parts) > 2 else resolved
    if status == "AMBIGUOUS" and len(parts) > 1:
        detail = parts[1].strip()
    return status, resolved, detail


def _table_path_for_worker_path(worker_path: str, config: AppConfig) -> str:
    table_prefix = config.reusable_workers.table_model_prefix.rstrip("/")
    if worker_path.startswith(table_prefix + "/"):
        return worker_path[len(table_prefix) + 1 :]
    return worker_path.lstrip("/")


def _replace_vllm_model_arg(command: str, old_path: str, new_path: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command.replace(old_path, new_path)
    for index, part in enumerate(parts[:-1]):
        if part == "--model":
            parts[index + 1] = new_path
            return shlex.join(parts)
    return command.replace(old_path, new_path)


def _should_fallback_worker_to_tmux(result: RemoteResult) -> bool:
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part).lower()
    return (
        "permission denied" in detail
        or "client_global_hostkeys_prove_confirm" in detail
        or "server gave bad signature" in detail
    )


def _parse_tmux_worker_result(
    command: str,
    text: str,
    begin: str,
    err: str,
    end: str,
) -> RemoteResult | None:
    end_prefix = end + ":"
    end_pos = text.rfind(end_prefix)
    if end_pos < 0:
        return None
    rc_text = text[end_pos + len(end_prefix) :].splitlines()[0].strip()
    try:
        returncode = int(rc_text.split()[0])
    except (IndexError, ValueError):
        return None
    begin_pos = text.rfind(begin, 0, end_pos)
    err_pos = text.rfind(err, 0, end_pos)
    if begin_pos < 0 or err_pos < begin_pos:
        return None
    stdout = text[begin_pos + len(begin) : err_pos].strip("\r\n")
    stderr = text[err_pos + len(err) : end_pos].strip("\r\n")
    return RemoteResult(command=command, returncode=returncode, stdout=stdout, stderr=stderr)


def _row_already_serves_model(row: dict[str, Any], path: dict[str, Any], marker: str) -> bool:
    model = _strip_deploying_marker(str(row.get("model") or ""), marker)
    model_id = _strip_deploying_marker(str(row.get("model_id") or ""), marker)
    return model == str(path.get("table_path") or "").strip() and model_id == str(path.get("model_id") or "").strip()


def _strip_deploying_marker(value: str, marker: str) -> str:
    text = value.strip()
    if marker and text.endswith(marker):
        text = text[: -len(marker)].strip()
    return text


def _has_worker_window(windows: list[str]) -> bool:
    for raw in windows:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        index, name, command = parts
        if name == "ssh" or (index == "0" and command in {"ssh", "bash", "zsh"}):
            return True
    return False


def _nvidia_compute_apps_command() -> str:
    return (
        "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
        "/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}; "
        "/usr/local/nvidia/bin/nvidia-smi "
        "--query-compute-apps=pid,process_name,used_gpu_memory "
        "--format=csv,noheader,nounits"
    )


def _kill_port_listeners_command(port: int) -> str:
    port = int(port)
    return (
        "pids=$(("
        f"lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true; "
        f"fuser -n tcp {port} 2>/dev/null || true"
        ") | tr ' ' '\\n' | sed '/^$/d' | sort -u); "
        "if [ -n \"$pids\" ]; then kill $pids 2>/dev/null || true; sleep 5; kill -9 $pids 2>/dev/null || true; fi"
    )


def _kill_pids_command(pids: list[str]) -> str:
    safe_pids = [pid for pid in pids if pid.isdigit()]
    if not safe_pids:
        return "true"
    joined = " ".join(shlex.quote(pid) for pid in safe_pids)
    return f"kill {joined} 2>/dev/null || true; sleep 3; kill -9 {joined} 2>/dev/null || true"


def _gpu_app_pids(output: str) -> list[str]:
    pids: list[str] = []
    for line in output.splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit() and first not in pids:
            pids.append(first)
    return pids


def _row_can_auto_reuse(row: dict[str, Any], config: AppConfig) -> bool:
    if not reuse_flag_allows_scan(
        str(row.get("reuse") or row.get("复用") or ""),
        column_present=_reuse_column_present(row),
    ):
        return False
    return is_reusable_worker_state(
        str(row.get("model") or ""),
        str(row.get("model_id") or ""),
        str(row.get("tested_tasks") or ""),
        config.reusable_workers,
    )


def _reuse_column_present(row: dict[str, Any]) -> bool:
    if "reuse_column_present" in row:
        return bool(row.get("reuse_column_present"))
    return "reuse" in row or "复用" in row


def _current_row_safe_for_plan(row: dict[str, Any], path: dict[str, Any], config: AppConfig) -> bool:
    if _row_already_serves_model(row, path, config.reusable_workers.deploying_marker):
        return True
    return _row_can_auto_reuse(row, config)


def _model_path_info_from_plan_path(path: dict[str, Any]) -> ModelPathInfo:
    worker_path = _expect_str(path, "worker_path")
    table_path = str(path.get("table_path") or worker_path.lstrip("/"))
    return ModelPathInfo(
        original_path=str(path.get("original_path") or worker_path),
        worker_path=worker_path,
        table_path=table_path,
        model_id=_expect_str(path, "model_id"),
    )


def _expect_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ReusableDeploymentError(f"missing plan.{key}")
    return value


def _expect_str(parent: dict[str, Any], key: str) -> str:
    value = str(parent.get(key) or "").strip()
    if not value:
        raise ReusableDeploymentError(f"missing required field: {key}")
    return value


def _short(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _classify_deployment_error(error: str) -> dict[str, str]:
    text = str(error or "")
    lower = text.lower()
    if "weight conversion failed" in lower or "conversion" in lower and "failed" in lower:
        return {
            "type": "weight_conversion",
            "label": "权重转换失败",
            "action": "检查转换机资源、转换脚本输出和目标目录；137 通常优先看内存/OOM。",
        }
    if "can't find window" in lower or "has no worker ssh window" in lower or "failed to reconnect" in lower:
        return {
            "type": "worker_tmux_disconnected",
            "label": "worker tmux/SSH 断开",
            "action": "先用 reconnect-plan 或文档里的 SSH 历史命令恢复 worker 窗口，再重试。",
        }
    if "server gave bad signature" in lower or "hostkeys" in lower or "host key" in lower:
        return {
            "type": "ssh_host_key",
            "label": "SSH host key 问题",
            "action": "清理或更新 known_hosts，确认命令包含 UpdateHostKeys=no 后重试。",
        }
    if "gpu" in lower and ("busy" in lower or "not idle" in lower or "still busy" in lower):
        return {
            "type": "gpu_busy",
            "label": "GPU 残留进程",
            "action": "进入 worker 检查 nvidia-smi，清理残留 vLLM/worker 进程后重试。",
        }
    if "model path missing" in lower or "model directory is not deployable" in lower or "remote command failed" in lower and "ls" in lower:
        return {
            "type": "model_path",
            "label": "模型路径异常",
            "action": "检查子任务路径、前缀转换和 HF/config 文件是否存在。",
        }
    if "feishu http 401" in lower or "authentication token expired" in lower:
        return {
            "type": "feishu_auth",
            "label": "飞书认证失效",
            "action": "刷新用户 token 或检查应用 tenant token 配置。",
        }
    if "feishu http 403" in lower or "no permission" in lower or "forbidden" in lower:
        return {
            "type": "feishu_permission",
            "label": "飞书权限不足",
            "action": "检查 bot 是否被加入文档/任务，或对应 API 权限是否已开通。",
        }
    if "temporary failure in name resolution" in lower or "name resolution" in lower or "connecterror" in lower:
        return {
            "type": "network",
            "label": "网络/DNS 异常",
            "action": "检查代理、DNS、Feishu OpenAPI 和 worker 网络连通性后重试。",
        }
    if "/v1/models" in lower or "model id" in lower or "health" in lower:
        return {
            "type": "vllm_health",
            "label": "vLLM 健康检查失败",
            "action": "查看 tmux vLLM 日志，确认端口、served-model-name 和启动参数。",
        }
    return {
        "type": "unknown",
        "label": "未分类错误",
        "action": "查看任务卡片和 tmux 日志，必要时人工处理后重试。",
    }


def _timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _approval_card_summary(decision: dict[str, Any], payload: dict[str, Any]) -> str:
    raw = str(decision.get("approval_summary") or decision.get("summary") or payload.get("summary") or "审核通过。")
    if "已部署模型文档满足复用条件" in raw:
        return "复用条件通过；worker、路径、tmux session、卡数检查通过。"
    return _short(raw, 160)


def _conversion_done_summary(output_path: str) -> str:
    output_name = output_path.strip().rstrip("/").rsplit("/", 1)[-1]
    if output_name:
        return f"权重转换完成：{output_name}"
    return "权重转换完成。"


def _conversion_card_summary(summary: str) -> str:
    text = str(summary or "").strip()
    marker = "权重转换完成："
    if marker in text:
        _, value = text.split(marker, 1)
        output_name = value.strip().rstrip("/").rsplit("/", 1)[-1]
        if output_name:
            return f"{marker}{output_name}"
    return text or "权重转换完成。"


def _md_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for char in ("*", "_", "~", "`", "[", "]", "(", ")"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def _simple_status_text(review: dict[str, Any], decision: dict[str, Any]) -> str:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
    row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
    model_id = str(decision.get("model_id") or path.get("model_id") or _subject_model(str(review.get("subject_id") or "")))
    endpoint = str(decision.get("endpoint") or "")
    worker = str(decision.get("worker") or row.get("ip") or _subject_worker(str(review.get("subject_id") or "")))
    status = str(decision.get("deploy_status") or review.get("status") or "")
    if status == "deployed":
        return f"模型部署完成：{model_id}\nworker: {worker}\nendpoint: {endpoint}"
    if status in {"failed", "deploy_failed"}:
        return f"模型部署失败：{model_id}\n原因：{_short(str(decision.get('summary') or decision.get('error') or ''), 160)}"
    if status == "needs_human":
        return f"模型部署需要人工处理：{model_id}\nworker: {worker}\n原因：{_short(str(decision.get('summary') or decision.get('error') or ''), 160)}"
    return f"模型部署状态：{model_id} {status}"


def _human_alert_text(config: AppConfig, review: dict[str, Any], decision: dict[str, Any]) -> str:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
    row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
    status = str(decision.get("deploy_status") or review.get("status") or "")
    if status not in {"failed", "deploy_failed", "needs_human"}:
        return ""
    mention = mention_text(config.approval)
    if not mention:
        return ""
    model_id = str(decision.get("model_id") or path.get("model_id") or _subject_model(str(review.get("subject_id") or "")))
    worker = str(decision.get("worker") or row.get("ip") or _subject_worker(str(review.get("subject_id") or "")))
    reason = _short(str(decision.get("summary") or decision.get("error") or ""), 220)
    return f"{mention} 模型部署需要人工处理：{model_id}\nworker: {worker}\n原因：{reason}"


def _row_address(row: dict[str, Any]) -> str:
    ip = str(row.get("ip") or "").strip()
    gpu_count = str(row.get("gpu_count") or "").strip()
    if ip and gpu_count:
        return f"{ip} ({gpu_count}卡)"
    return ip or str(row.get("address") or "").strip()


def _subject_worker(subject_id: str) -> str:
    return subject_id.split(":", 1)[0] if ":" in subject_id else subject_id


def _subject_model(subject_id: str) -> str:
    return subject_id.split(":", 1)[1] if ":" in subject_id else ""
