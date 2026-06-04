from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
import logging
import re
import signal
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from fmh.config import AppConfig, CodexReviewConfig
from fmh.feishu import FeishuOpenAPIClient, NullFeishuClient
from fmh.operator_review import (
    ReviewPacket,
    human_fallback_footer,
    mention_text,
    normalize_decision,
    review_card,
    review_result_card,
    review_status_for_decision,
)
from fmh.reusable_workers import is_reusable_worker_state, reuse_flag_allows_scan
from fmh.reusable_executor import ReusableDeploymentExecutor
from fmh.store import StateStore
from fmh.task_status import task_status_card, task_status_with_stage
from fmh.time_utils import utc_now_iso

log = logging.getLogger(__name__)
_STALE_APPROVED_REVIEW_SEC = 3600


class CodexAuditError(RuntimeError):
    pass


class CodexReviewAuditor:
    def __init__(
        self,
        config: AppConfig,
        store: StateStore,
        feishu_client: FeishuOpenAPIClient | NullFeishuClient,
    ) -> None:
        self.config = config
        self.store = store
        self.feishu = feishu_client
        self.executor = ReusableDeploymentExecutor(config, store, feishu_client)
        self.max_parallel_deployments = max(1, int(config.reusable_workers.max_parallel_deployments))
        self._pool = ThreadPoolExecutor(
            max_workers=self.max_parallel_deployments,
            thread_name_prefix="fmh-review",
        )
        self._futures: dict[str, Future[None]] = {}
        self._future_workers: dict[str, str] = {}
        self._futures_lock = threading.Lock()
        self._wake_event = threading.Event()

    def run_forever(self) -> None:
        self._install_wake_signal_handler()
        log.info(
            "codex review auditor started: interval=%ss max_parallel_deployments=%s",
            self.config.codex_review.interval_sec,
            self.max_parallel_deployments,
        )
        while True:
            count = self.process_once()
            log.info(
                "codex review auditor tick: started=%s active=%s",
                count,
                self.active_count(),
            )
            self._wake_event.wait(self.config.codex_review.interval_sec)
            self._wake_event.clear()

    def process_once(self, *, wait: bool = False) -> int:
        self._collect_finished()
        self._recover_orphaned_inflight_reviews()
        self._recover_stale_approved_reviews()
        count = 0
        started_review_ids: list[str] = []
        active_workers = self._active_workers()
        statuses = ["pending", "retry_requested"]
        if self.config.reusable_workers.auto_deploy_approved:
            statuses.append("approved")
        for status in statuses:
            for review in self.store.list_reviews(limit=20, status=status):
                if self.active_count() >= self.max_parallel_deployments:
                    break
                worker = _review_worker(review)
                if worker and worker in active_workers:
                    continue
                review_id = str(review.get("review_id") or "")
                if not review_id:
                    continue
                claim_status = "deploying" if status == "approved" else "codex_reviewing"
                claim_decision = {"source": "auditor", "status": "claimed", "claimed_at": utc_now_iso()}
                if status == "approved":
                    previous = review.get("decision") if isinstance(review.get("decision"), dict) else {}
                    claim_decision = {
                        **previous,
                        "decision": "APPROVE",
                        "source": str(previous.get("source") or "auditor"),
                        "status": "claimed_approved",
                        "claimed_at": utc_now_iso(),
                    }
                if not self.store.claim_review(
                    review_id,
                    from_statuses=(status,),
                    to_status=claim_status,
                    decision=claim_decision,
                ):
                    continue
                claimed = self.store.get_review(review_id) or review
                self._submit_review(claimed, worker)
                if worker:
                    active_workers.add(worker)
                started_review_ids.append(review_id)
                count += 1
            if self.active_count() >= self.max_parallel_deployments:
                break
        if wait:
            self._wait_for(started_review_ids)
        return count

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def active_count(self) -> int:
        with self._futures_lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def _submit_review(self, review: dict[str, Any], worker: str) -> None:
        review_id = str(review.get("review_id") or "")
        future = self._pool.submit(self._process_review_safe, review)
        with self._futures_lock:
            self._futures[review_id] = future
            self._future_workers[review_id] = worker

    def _process_review_safe(self, review: dict[str, Any]) -> None:
        review_id = str(review.get("review_id") or "")
        try:
            self._process_review(review)
        except Exception as exc:
            log.exception("review auditor internal error: %s", review_id)
            decision = {
                "decision": "REQUEST_INFO",
                "source": "auditor",
                "error": f"review auditor internal error: {exc}",
                "decided_at": utc_now_iso(),
            }
            self.store.decide_review(review_id, "needs_human", decision)

    def _wait_for(self, review_ids: list[str]) -> None:
        for review_id in review_ids:
            with self._futures_lock:
                future = self._futures.get(review_id)
            if future is not None:
                future.result()
        self._collect_finished()

    def _collect_finished(self) -> None:
        with self._futures_lock:
            finished = [review_id for review_id, future in self._futures.items() if future.done()]
            for review_id in finished:
                self._futures.pop(review_id, None)
                self._future_workers.pop(review_id, None)

    def _recover_orphaned_inflight_reviews(self) -> None:
        with self._futures_lock:
            active_review_ids = {
                review_id for review_id, future in self._futures.items() if not future.done()
            }
        for status in ("codex_reviewing", "deploying"):
            for review in self.store.list_reviews(limit=100, status=status):
                review_id = str(review.get("review_id") or "")
                if not review_id or review_id in active_review_ids:
                    continue
                decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
                self.store.decide_review(
                    review_id,
                    "retry_requested",
                    {
                        **decision,
                        "source": "auditor",
                        "status": "recovered_after_restart",
                        "recovered_from_status": status,
                        "summary": "审核进程重启后恢复未完成部署，已重新排队。",
                        "decided_at": utc_now_iso(),
                    },
                )
                log.warning("requeued orphaned in-flight review after restart: %s from %s", review_id, status)

    def _recover_stale_approved_reviews(self) -> None:
        for review in self.store.list_reviews(limit=100, status="approved"):
            age = _review_age_sec(review)
            if age < _STALE_APPROVED_REVIEW_SEC:
                continue
            review_id = str(review.get("review_id") or "")
            if not review_id:
                continue
            decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
            self.store.decide_review(
                review_id,
                "needs_human",
                {
                    **decision,
                    "source": "auditor",
                    "status": "stale_approved",
                    "summary": "历史 approved review 超过 1 小时未执行，已转人工以避免误部署。",
                    "decided_at": utc_now_iso(),
                },
            )
            log.warning("marked stale approved review as needs_human: %s", review_id)

    def _active_workers(self) -> set[str]:
        active: set[str] = set()
        with self._futures_lock:
            for review_id, future in self._futures.items():
                if not future.done():
                    worker = self._future_workers.get(review_id, "")
                    if worker:
                        active.add(worker)
        for status in ("codex_reviewing", "deploying"):
            for review in self.store.list_reviews(limit=100, status=status):
                worker = _review_worker(review)
                if worker:
                    active.add(worker)
        return active

    def _install_wake_signal_handler(self) -> None:
        def wake(_signum: int, _frame: object) -> None:
            log.info("codex review auditor wake signal received")
            self._wake_event.set()

        try:
            signal.signal(signal.SIGUSR1, wake)
        except (AttributeError, ValueError):
            log.debug("SIGUSR1 wake signal is unavailable in this runtime")

    def _process_review(self, review: dict[str, Any]) -> None:
        review_id = str(review["review_id"])
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        existing_decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
        if str(review.get("status") or "") == "deploying" and str(existing_decision.get("decision") or "").upper() == "APPROVE":
            self.executor.execute_if_enabled(review, existing_decision)
            return
        policy_decision = deterministic_review_decision(self.config, payload)
        if policy_decision is not None:
            status = review_status_for_decision(str(policy_decision["decision"]))
            decision = {
                **policy_decision,
                "source": "policy",
                "decided_at": utc_now_iso(),
            }
            self.store.decide_review(review_id, status, decision)
            deferred_to_deploy = decision["decision"] == "APPROVE" and self.config.reusable_workers.auto_deploy_approved
            if self.config.codex_review.send_result_cards and not deferred_to_deploy:
                updated = self.store.get_review(review_id) or review
                self._send_card(
                    review_result_card(updated, decision),
                    f"审核结果: {review_id} {decision['decision']}",
                    review=updated,
                )
            if decision["decision"] == "APPROVE":
                updated = self.store.get_review(review_id) or review
                self.executor.execute_if_enabled(updated, decision)
            return

        if not self._codex_enabled():
            self._request_human(review, "Codex 审核已关闭，当前阶段没有确定性通过规则。")
            return

        self.store.decide_review(
            review_id,
            "codex_reviewing",
            {"source": "codex", "status": "running", "started_at": utc_now_iso()},
        )
        try:
            decision = run_codex_review(self.config.codex_review, payload)
        except Exception as exc:
            log.exception("codex review failed: %s", review_id)
            self._request_human(review, str(exc))
            return

        status = review_status_for_decision(str(decision["decision"]))
        decision = {
            **decision,
            "source": "codex",
            "decided_at": utc_now_iso(),
        }
        self.store.decide_review(review_id, status, decision)
        deferred_to_deploy = decision["decision"] == "APPROVE" and self.config.reusable_workers.auto_deploy_approved
        if self.config.codex_review.send_result_cards and not deferred_to_deploy:
            updated = self.store.get_review(review_id) or review
            self._send_card(
                review_result_card(updated, decision),
                f"Codex 审核结果: {review_id} {decision['decision']}",
                review=updated,
            )
        if decision["decision"] == "APPROVE":
            updated = self.store.get_review(review_id) or review
            self.executor.execute_if_enabled(updated, decision)

    def _request_human(self, review: dict[str, Any], error: str) -> None:
        review_id = str(review["review_id"])
        decision = {
            "decision": "REQUEST_INFO",
            "source": "codex",
            "error": error,
            "decided_at": utc_now_iso(),
        }
        self.store.decide_review(review_id, "needs_human", decision)
        updated = self.store.get_review(review_id) or review
        self._update_task_status(updated, decision, error)
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        packet = ReviewPacket.from_dict(payload)
        footer = human_fallback_footer(self.config.approval, _shorten(error, 800))
        self._send_card(
            review_card(packet, self.config.approval, footer=footer),
            f"Codex 审核失败，需要人工确认: {review_id}",
            review=updated,
        )
        mention = mention_text(self.config.approval)
        if mention:
            self._send_text(
                f"{mention} Codex 自动审核失败，需要人工确认。\nreview_id: {review_id}\n原因：{_shorten(error, 220)}",
                review=updated,
            )

    def _send_card(self, card: dict[str, Any], fallback_text: str, *, review: dict[str, Any] | None = None) -> None:
        chat_id = _review_source_chat_id(review or {}) or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        if not chat_id:
            return
        try:
            self.feishu.send_chat_card(chat_id, card)
        except Exception:
            self.feishu.send_chat_text(chat_id, fallback_text)

    def _send_text(self, text: str, *, review: dict[str, Any] | None = None) -> None:
        chat_id = _review_source_chat_id(review or {}) or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        if not chat_id:
            return
        try:
            self.feishu.send_chat_text(chat_id, text)
        except Exception:
            log.exception("failed to send human fallback text")

    def _update_task_status(self, review: dict[str, Any], decision: dict[str, Any], error: str) -> None:
        payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        task_key = str(context.get("status_task_key") or context.get("task_key") or "")
        source_chat_id = str(context.get("source_chat_id") or "")
        message_id = str(context.get("status_message_id") or "")
        if not task_key or not source_chat_id or not message_id:
            return
        state = self.store.get_task_status(task_key)
        if str(state.get("source_message_id") or "") not in {"", message_id}:
            state = {}
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        path = plan.get("path") if isinstance(plan.get("path"), dict) else {}
        row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
        state = task_status_with_stage(
            state,
            "codex",
            "需人工",
            _shorten(error, 180),
            title=str(context.get("task_title") or state.get("title") or ""),
            source_chat_id=source_chat_id,
            source_message_id=message_id,
            model_id=str(path.get("model_id") or ""),
            model=str(path.get("table_path") or path.get("original_path") or path.get("worker_path") or ""),
            worker=str(row.get("ip") or ""),
            address=_row_address(row),
            review_id=str(review.get("review_id") or ""),
        )
        state["card_actions_enabled"] = self.config.approval.allow_card_actions
        try:
            self.feishu.update_card(message_id, task_status_card(state))
        except Exception:
            log.exception("failed to update source task status card")
        self.store.set_task_status(task_key, state)

    def _codex_enabled(self) -> bool:
        value = self.store.get_setting(self.config.codex_review.runtime_toggle_key)
        if value is None:
            return self.config.codex_review.enabled
        return value == "true"


def run_codex_review(config: CodexReviewConfig, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = _auditor_prompt(str(payload.get("codex_prompt") or json.dumps(payload, ensure_ascii=False, indent=2)))
    command = _command(config.command)
    cwd = Path(config.cd).expanduser()
    if isinstance(config.command, str):
        result = subprocess.run(
            config.command,
            input=prompt,
            text=True,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            timeout=config.timeout_sec,
            check=False,
        )
    else:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            cwd=str(cwd),
            capture_output=True,
            timeout=config.timeout_sec,
            check=False,
        )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    output = _shorten(output, config.max_output_chars)
    if result.returncode != 0:
        raise CodexAuditError(f"codex exited {result.returncode}: {output}")
    decision = parse_codex_decision(output)
    if decision is None:
        raise CodexAuditError(f"codex output has no decision: {output}")
    decision["raw_output"] = output
    return decision


def deterministic_review_decision(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("stage") != "reuse_row_selected":
        return None
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    row = plan.get("row") if isinstance(plan.get("row"), dict) else context.get("selected_row")
    path = plan.get("path") if isinstance(plan.get("path"), dict) else context.get("path")
    if not isinstance(row, dict) or not isinstance(path, dict):
        return None

    if not reuse_flag_allows_scan(
        str(row.get("reuse") or row.get("复用") or ""),
        column_present=_reuse_column_present(row),
    ):
        return None

    if not is_reusable_worker_state(
        str(row.get("model") or ""),
        str(row.get("model_id") or ""),
        str(row.get("tested_tasks") or ""),
        config.reusable_workers,
    ):
        return None

    ip = str(row.get("ip") or "")
    gpu_count = _int_or_zero(row.get("gpu_count"))
    worker_path = str(path.get("worker_path") or "")
    vllm_command = str(plan.get("vllm_command") or "")
    tmux_session = str(plan.get("tmux_session_guess") or "")
    if not ip or not gpu_count or not worker_path or not vllm_command or not tmux_session:
        return None
    if not worker_path.startswith(config.reusable_workers.worker_model_prefix.rstrip("/") + "/"):
        return None
    suffix = "_".join(ip.split(".")[-2:])
    if suffix not in tmux_session:
        return None
    if f"--data-parallel-size {gpu_count}" not in vllm_command:
        return None

    return {
        "decision": "APPROVE",
        "summary": (
            "已部署模型文档满足复用条件：复用列允许扫描，空闲行或 required tasks 已完成且无 running 标记，"
            "路径、tmux session 和卡数检查通过。"
        ),
        "risks": [
            "后续 before_stop_vllm 阶段仍需确认进入正确 tmux session 后再停旧服务。",
            "启动后仍需通过 /v1/models 校验模型 id。",
        ],
        "next_actions": [
            "进入 before_stop_vllm 阶段，确认 tmux session 后停止旧 vLLM。",
            "运行计划中的 vllm_command。",
            "启动后检查 /v1/models，并在通过后写回文档。",
        ],
    }


def _reuse_column_present(row: dict[str, Any]) -> bool:
    if "reuse_column_present" in row:
        return bool(row.get("reuse_column_present"))
    return "reuse" in row or "复用" in row


def parse_codex_decision(output: str) -> dict[str, Any] | None:
    for obj in _json_objects(output):
        decision = obj.get("decision")
        if isinstance(decision, str):
            try:
                normalized = normalize_decision(decision)
            except ValueError:
                continue
            return {**obj, "decision": normalized}
    match = re.search(r"\bdecision\s*[:=]\s*(APPROVE|BLOCK|RETRY|REQUEST_INFO)\b", output, re.I)
    if match:
        return {"decision": normalize_decision(match.group(1)), "summary": _first_non_empty_line(output)}
    return None


def _json_objects(output: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _row_address(row: dict[str, Any]) -> str:
    ip = str(row.get("ip") or "").strip()
    gpu_count = str(row.get("gpu_count") or "").strip()
    if ip and gpu_count:
        return f"{ip} ({gpu_count}卡)"
    return ip or str(row.get("address") or "").strip()


def _review_age_sec(review: dict[str, Any]) -> float:
    updated_at = str(review.get("updated_at") or review.get("created_at") or "")
    if not updated_at:
        return 0.0
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    now = datetime.now(updated.tzinfo)
    return max(0.0, (now - updated).total_seconds())


def _review_worker(review: dict[str, Any]) -> str:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
    if row.get("ip"):
        return str(row.get("ip") or "")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    selected = context.get("selected_row") if isinstance(context.get("selected_row"), dict) else {}
    if selected.get("ip"):
        return str(selected.get("ip") or "")
    subject = str(review.get("subject_id") or payload.get("subject_id") or "")
    if ":" in subject:
        return subject.split(":", 1)[0]
    return ""


def _review_source_chat_id(review: dict[str, Any]) -> str:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    return str(context.get("source_chat_id") or "")


def _auditor_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\n"
        "最后必须输出一个单独的 JSON 对象，格式如下：\n"
        "{\"decision\":\"APPROVE|BLOCK|RETRY|REQUEST_INFO\","
        "\"summary\":\"一句话结论\","
        "\"risks\":[\"主要风险\"],"
        "\"next_actions\":[\"下一步\"]}\n"
        "不要执行部署、停止服务、写文档等有副作用操作。只做审核。"
    )


def _command(command: list[str] | str) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 80] + f"\n... truncated {len(text) - limit + 80} chars"


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
