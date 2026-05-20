from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Protocol

from fmh.approval import decide_review
from fmh.config import AppConfig
from fmh.models import EventSource, Requester, SourceType
from fmh.operator_review import (
    make_error_review,
    make_reuse_plan_review,
    mention_text,
    parse_review_command,
    review_card,
    review_result_card,
)
from fmh.orchestrator import DeploymentOrchestrator
from fmh.parser import ParseError, parse_deployment_request
from fmh.reusable_workers import build_reusable_deployment_plan, parse_deployed_models_table
from fmh.store import StateStore
from fmh.task_status import task_status_card, task_status_with_stage

log = logging.getLogger(__name__)


class PollingFeishuClient(Protocol):
    def list_chats(self, page_size: int = 50) -> list[dict[str, Any]]:
        ...

    def list_messages(
        self,
        chat_id: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        page_size: int = 50,
        sort_type: str = "ByCreateTimeAsc",
    ) -> list[dict[str, Any]]:
        ...

    def get_document_raw_content(self, document_id: str) -> str:
        ...

    def get_doc_markdown(self, doc_token: str, doc_type: str = "docx") -> str:
        ...

    def get_task(self, task_guid: str) -> dict[str, Any]:
        ...

    def list_subtasks(self, task_guid: str, page_size: int = 50) -> list[dict[str, Any]]:
        ...

    def complete_task(self, task_guid: str) -> None:
        ...

    def send_chat_text(self, chat_id: str, text: str) -> str:
        ...

    def send_chat_card(self, chat_id: str, card: dict[str, Any]) -> str:
        ...

    def reply_text(self, message_id: str, text: str) -> str:
        ...

    def reply_card(self, message_id: str, card: dict[str, Any]) -> str:
        ...

    def update_card(self, message_id: str, card: dict[str, Any]) -> None:
        ...

    def add_message_reaction(self, message_id: str, emoji_type: str) -> str:
        ...

    def get_bot_open_id(self) -> str:
        ...


@dataclass(frozen=True)
class PollStats:
    scanned: int = 0
    submitted: int = 0
    ignored: int = 0
    failed: int = 0
    manual_polls: int = 0
    task_summaries: tuple[str, ...] = ()

    def add(self, other: "PollStats") -> "PollStats":
        return PollStats(
            scanned=self.scanned + other.scanned,
            submitted=self.submitted + other.submitted,
            ignored=self.ignored + other.ignored,
            failed=self.failed + other.failed,
            manual_polls=self.manual_polls + other.manual_polls,
            task_summaries=_merge_unique(self.task_summaries, other.task_summaries),
        )


class FeishuPollingWorker:
    def __init__(
        self,
        config: AppConfig,
        store: StateStore,
        feishu_client: PollingFeishuClient,
        orchestrator: DeploymentOrchestrator,
    ) -> None:
        self.config = config
        self.store = store
        self.feishu = feishu_client
        self.orchestrator = orchestrator
        self._discovered_chat_ids: list[str] = []
        self._discovered_chat_ids_at = 0
        self._bot_open_id = self.config.feishu.bot_open_id.strip()
        self._bot_open_id_checked = bool(self._bot_open_id)

    def run_forever(self) -> None:
        log.info("polling started: interval=%ss", self.config.polling.interval_sec)
        while True:
            try:
                stats = self.poll_once()
                log.info(
                    "polling tick: scanned=%s submitted=%s ignored=%s failed=%s",
                    stats.scanned,
                    stats.submitted,
                    stats.ignored,
                    stats.failed,
                )
            except Exception:
                log.exception("polling tick failed")
            time.sleep(self.config.polling.interval_sec)

    def poll_once(self, *, lookback_sec: int | None = None) -> PollStats:
        stats = PollStats()
        for chat_id in self._chat_ids():
            stats = stats.add(self._poll_chat(chat_id, lookback_sec=lookback_sec))
        for document_id in self.config.polling.document_ids:
            stats = stats.add(self._poll_document(document_id))
        if (
            self.config.polling.deploy_todo_subtasks
            and self.config.polling.watch_known_todo_tasks
            and not stats.manual_polls
        ):
            stats = stats.add(self._poll_known_todo_tasks(force=lookback_sec is not None))
        return stats

    def _poll_chat(self, chat_id: str, *, lookback_sec: int | None) -> PollStats:
        source_key = f"chat:{chat_id}"
        now = int(time.time())
        cursor = self.store.get_cursor(source_key)
        if lookback_sec is not None:
            start_time = max(0, now - lookback_sec)
        elif cursor is None:
            if self.config.polling.process_existing_on_first_run:
                start_time = max(0, now - self.config.polling.initial_lookback_sec)
            else:
                self.store.set_cursor(source_key, str(now))
                return PollStats()
        else:
            start_time = max(0, int(cursor) - 5)

        messages = self.feishu.list_messages(
            chat_id,
            start_time=start_time,
            end_time=now,
            page_size=self.config.polling.page_size,
            sort_type="ByCreateTimeAsc",
        )
        stats = PollStats()
        max_seen = int(cursor) if cursor else now
        for item in sorted(messages, key=_message_create_time):
            msg_id = str(item.get("message_id") or item.get("msg_id") or "")
            if not msg_id:
                continue
            max_seen = max(max_seen, _message_create_time(item))
            if self.store.has_processed_item(source_key, msg_id):
                continue
            item_stats = self._handle_message_item(chat_id, source_key, item, msg_id)
            stats = stats.add(item_stats)

        self.store.set_cursor(source_key, str(max_seen or now))
        return stats

    def _poll_document(self, document_id: str) -> PollStats:
        source_key = f"doc:{document_id}"
        content = self.feishu.get_document_raw_content(document_id).strip()
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        cursor = self.store.get_cursor(source_key)
        if cursor is None and not self.config.polling.process_existing_on_first_run:
            self.store.set_cursor(source_key, digest)
            return PollStats()
        if cursor == digest:
            return PollStats()

        self.store.set_cursor(source_key, digest)
        if not content:
            return PollStats(scanned=1, ignored=1)

        source = EventSource(
            source_type=SourceType.DOC,
            source_ref=document_id,
            requester=Requester(user_id="unknown", display_name="doc"),
            text=content,
            raw_event={"event": {"document_id": document_id, "doc_text": content}},
        )
        return self._submit_source(source_key, digest, source)

    def _poll_known_todo_tasks(self, *, force: bool = False) -> PollStats:
        chat_ids = self._chat_ids()
        chat_id = self.config.feishu.default_chat_id or (chat_ids[0] if chat_ids else "")
        stats = PollStats()
        now = int(time.time())
        checked = 0
        max_per_tick = max(0, int(self.config.polling.known_todo_max_per_tick))
        interval = max(0, int(self.config.polling.known_todo_check_interval_sec))
        for task_id in self.store.list_todo_task_ids():
            if max_per_tick and checked >= max_per_tick:
                break
            task_key = f"todo:{task_id}"
            due_retry = self._task_has_due_retry(task_key, now)
            if not force and not due_retry and interval:
                last_checked = _safe_int(self.store.get_setting(_known_todo_checked_key(task_id)))
                if last_checked and now - last_checked < interval:
                    continue
            source_chat_id = self.store.get_setting(f"todo_task_source_chat:{task_id}") or chat_id
            task_stats, _, _ = self._process_task_subtasks(task_id, task_key, chat_id=source_chat_id)
            self.store.set_setting(_known_todo_checked_key(task_id), str(now))
            stats = stats.add(task_stats)
            checked += 1
        return stats

    def _task_has_due_retry(self, task_key: str, now: int) -> bool:
        with self.store._connect() as conn:  # noqa: SLF001
            rows = conn.execute(
                """
                SELECT item_id FROM processed_items
                WHERE source_key = ? AND status = 'retry_waiting'
                """,
                (task_key,),
            ).fetchall()
        for row in rows:
            retry_at = _safe_int(self.store.get_setting(_retry_setting_key(task_key, str(row["item_id"]))))
            if retry_at <= now:
                return True
        return False

    def _handle_message_item(
        self,
        chat_id: str,
        source_key: str,
        item: dict[str, Any],
        msg_id: str,
    ) -> PollStats:
        msg_type = str(item.get("msg_type") or item.get("message_type") or "")
        if self.config.polling.ignore_self_messages and _is_self_or_bot_message(item, self.config.feishu.app_id):
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary="self/bot message")
            return PollStats(scanned=1, ignored=1)

        if msg_type == "todo" and self.config.polling.deploy_todo_subtasks:
            return self._handle_todo_item(chat_id, source_key, item, msg_id)
        if msg_type and msg_type not in {"text", "post"}:
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary=f"unsupported msg_type: {msg_type}")
            return PollStats(scanned=1, ignored=1)

        text = _message_text(item)
        if not text:
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary="empty or non-text message")
            return PollStats(scanned=1, ignored=1)
        if _looks_like_card_payload(text):
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary="interactive card payload")
            return PollStats(scanned=1, ignored=1)

        mentions_current_bot = _mentions_current_bot(item, text, self.config.feishu.app_id, self._current_bot_open_id())
        if mentions_current_bot:
            self._react_to_detected_task(msg_id)

        control = _parse_codex_control_command(text)
        if control is not None:
            return self._handle_codex_control_command(chat_id, source_key, msg_id, control)
        if _parse_help_command(text, mentions_current_bot=mentions_current_bot):
            return self._handle_help_command(chat_id, source_key, msg_id)
        if _parse_node_status_command(text):
            return self._handle_node_status_command(chat_id, source_key, msg_id)
        if _parse_manual_poll_command(text):
            return self._handle_manual_poll_command(chat_id, source_key, msg_id)

        command = parse_review_command(text) if self.config.approval.allow_group_commands else None
        if command is None and self.config.approval.allow_group_commands:
            command = self._parse_reply_review_command(text, item)
        if command is not None:
            return self._handle_review_command(chat_id, source_key, msg_id, item, command)

        source = EventSource(
            source_type=SourceType.GROUP_MESSAGE,
            source_ref=msg_id,
            requester=_message_requester(item),
            text=text,
            raw_event={"event": {"message": item, "chat_id": chat_id}},
        )
        return self._submit_source(source_key, msg_id, source, chat_id=chat_id)

    def _handle_todo_item(
        self,
        chat_id: str,
        source_key: str,
        item: dict[str, Any],
        msg_id: str,
    ) -> PollStats:
        task_id = _todo_task_id(item)
        if not task_id:
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary="todo message missing task_id")
            return PollStats(scanned=1, ignored=1)

        self._react_to_detected_task(msg_id)
        task_key = f"todo:{task_id}"
        if self.store.has_processed_item(source_key, msg_id):
            return PollStats()
        self.store.set_setting(f"todo_task_message:{task_id}", msg_id)
        self.store.set_setting(f"todo_task_source_chat:{task_id}", chat_id)

        stats, submitted_ids, failed = self._process_task_subtasks(
            task_id,
            task_key,
            chat_id=chat_id,
            message_item=item,
            reply_to_message_id=msg_id,
        )
        if submitted_ids:
            self.store.mark_processed_item(
                source_key,
                msg_id,
                "submitted",
                request_id=",".join(submitted_ids),
                summary=f"submitted {len(submitted_ids)} subtask deployments",
            )
        elif failed:
            self.store.mark_processed_item(source_key, msg_id, "failed_parse", summary=f"{failed} subtasks failed parse")
        else:
            self.store.mark_processed_item(source_key, msg_id, "ignored", summary="all subtasks already processed")

        return stats

    def _process_task_subtasks(
        self,
        task_id: str,
        task_key: str,
        *,
        chat_id: str = "",
        message_item: dict[str, Any] | None = None,
        reply_to_message_id: str = "",
    ) -> tuple[PollStats, list[str], int]:
        if not reply_to_message_id:
            reply_to_message_id = self.store.get_setting(f"todo_task_message:{task_id}") or ""
        if not chat_id:
            chat_id = self.store.get_setting(f"todo_task_source_chat:{task_id}") or ""
        try:
            parent = self.feishu.get_task(task_id)
            subtasks = self.feishu.list_subtasks(task_id, page_size=self.config.polling.page_size)
        except Exception as exc:
            self.store.mark_processed_item(task_key, f"fetch:{int(time.time())}", "failed_task_fetch", summary=str(exc))
            return PollStats(scanned=1, failed=1), [], 1

        entries = _deployment_entries_from_task(parent, subtasks, self.config.polling.relative_weight_path_prefix)
        if not entries:
            return PollStats(scanned=1, ignored=1), [], 0

        entry_pairs = [(entry, _task_entry_key(task_id, entry)) for entry in entries]
        entry_states = {item_key: self.store.get_processed_item(task_key, item_key) for _, item_key in entry_pairs}
        entry_states = self._refresh_task_entry_states_from_reviews(task_key, entry_states)
        pending_pairs = [
            (entry, item_key)
            for entry, item_key in entry_pairs
            if self._should_process_task_entry(task_key, item_key)
        ]
        pending_entries = [entry for entry, _ in pending_pairs]
        if not pending_entries:
            return PollStats(scanned=1, ignored=1), [], 0

        task_title = _task_title(parent, task_id)
        pending_entry_keys = [item_key for _, item_key in pending_pairs]
        counts = _task_entry_status_counts(entry_states, pending_entry_keys)
        detection_detail = _task_detection_detail(len(entries), len(pending_entries), counts)
        stats = PollStats(scanned=1)
        submitted_ids: list[str] = []
        task_summaries: list[str] = []
        failed = 0
        reserved_row_indices = self._reserved_reusable_rows(entry_states)
        for entry, item_key in pending_pairs:
            status_task_key = _task_item_status_key(task_key, item_key)
            item_state = self._update_task_status_card(
                status_task_key,
                chat_id,
                "detect",
                "完成",
                detection_detail,
                title=task_title,
                source_ref=task_id,
                model_id=_model_id_from_weight_path(str(entry.get("weight_path") or "")),
                model=str(entry.get("weight_path") or ""),
            )
            status_message_id = str(item_state.get("source_message_id") or "")
            entry["reply_to_message_id"] = reply_to_message_id
            entry["source_chat_id"] = chat_id
            entry["task_key"] = task_key
            entry["status_task_key"] = status_task_key
            entry["item_key"] = item_key
            entry["task_title"] = task_title
            entry["status_message_id"] = status_message_id
            entry["parent_task_id"] = task_id
            if self.config.reusable_workers.enabled:
                try:
                    review_id, row_index = self._create_reuse_review(
                        entry,
                        reserved_row_indices=reserved_row_indices,
                    )
                    if row_index:
                        reserved_row_indices.add(row_index)
                except Exception as exc:
                    if _is_no_reusable_worker_error(exc):
                        self._schedule_reuse_plan_retry(
                            task_key,
                            status_task_key,
                            item_key,
                            entry,
                            chat_id,
                            status_message_id,
                            task_title,
                            task_id,
                            str(exc),
                        )
                        continue
                    failed += 1
                    packet = make_error_review(
                        stage="reuse_plan",
                        subject_id=item_key,
                        error=str(exc),
                        context={"entry": entry},
                    )
                    self.store.create_review(packet.review_id, packet.stage.value, packet.subject_id, packet.to_dict())
                    if chat_id and self.config.polling.notify_chat_on_accept:
                        self._send_chat_card_or_text(
                            chat_id,
                            review_card(packet, self.config.approval),
                            f"部署计划生成失败: {exc}",
                        )
                    self.store.mark_processed_item(task_key, item_key, "failed_review", summary=str(exc))
                    continue
                submitted_ids.append(review_id)
                task_summaries.append(_task_summary_line(task_title, str(entry.get("weight_path") or "")))
                self.store.mark_processed_item(
                    task_key,
                    item_key,
                    "review_pending",
                    request_id=review_id,
                    summary="operator review pending",
                )
                continue
            source = EventSource(
                source_type=SourceType.GROUP_MESSAGE,
                source_ref=item_key,
                requester=_message_requester(message_item or {}),
                text=_deployment_text_from_entry(entry),
                raw_event={
                    "event": {
                        "message": message_item or {},
                        "chat_id": chat_id,
                        "task": parent,
                        "subtask": entry["subtask"],
                    }
                },
            )
            try:
                request = parse_deployment_request(source)
            except ParseError as exc:
                failed += 1
                self.store.mark_processed_item(task_key, item_key, "failed_parse", summary=str(exc))
                continue
            result = self.orchestrator.submit(request)
            submitted_ids.append(result.request_id)
            task_summaries.append(_task_summary_line(task_title, str(entry.get("weight_path") or "")))
            self._update_task_status_card(
                status_task_key,
                chat_id,
                "execute",
                "进行中",
                f"已提交部署请求 {result.request_id}，状态 {result.status.value}。",
                title=task_title,
                source_ref=task_id,
                model_id=_model_id_from_weight_path(str(entry.get("weight_path") or "")),
                model=str(entry.get("weight_path") or ""),
                source_message_id=status_message_id,
            )
            self.store.mark_processed_item(
                task_key,
                item_key,
                "submitted",
                request_id=result.request_id,
                summary=result.status.value,
            )

        if submitted_ids and not self.config.reusable_workers.auto_deploy_approved:
            if chat_id and self.config.polling.notify_chat_on_accept:
                self._send_chat_card_or_text(
                    chat_id,
                    _todo_accept_card(len(submitted_ids), submitted_ids, review_mode=self.config.reusable_workers.enabled),
                    "已接收任务子项部署:\n" + "\n".join(f"- {rid}" for rid in submitted_ids[:10]),
                    reply_to_message_id=reply_to_message_id,
                )
        if submitted_ids and self.config.reusable_workers.enabled:
            self._wake_review_auditor(len(submitted_ids))
        return PollStats(
            scanned=stats.scanned,
            submitted=len(submitted_ids),
            ignored=0 if submitted_ids or failed else 1,
            failed=failed,
            task_summaries=tuple(task_summaries),
        ), submitted_ids, failed

    def _should_process_task_entry(self, task_key: str, item_key: str) -> bool:
        processed = self.store.get_processed_item(task_key, item_key)
        if processed is None:
            return True
        if str(processed.get("status") or "") != "retry_waiting":
            return False
        due_at = _safe_int(self.store.get_setting(_retry_setting_key(task_key, item_key)))
        return due_at <= int(time.time())

    def _reserved_reusable_rows(self, entry_states: dict[str, dict[str, object] | None]) -> set[int]:
        reserved: set[int] = set()
        for state in entry_states.values():
            if not state:
                continue
            if str(state.get("status") or "") not in {"review_pending", "codex_reviewing", "approved", "deploying"}:
                continue
            review_id = str(state.get("request_id") or "")
            if not review_id:
                continue
            review = self.store.get_review(review_id)
            if not review:
                continue
            row = _review_plan_row(review)
            row_index = _safe_int(row.get("row_index") if row else "")
            if row_index:
                reserved.add(row_index)
        return reserved

    def _refresh_task_entry_states_from_reviews(
        self,
        task_key: str,
        entry_states: dict[str, dict[str, object] | None],
    ) -> dict[str, dict[str, object] | None]:
        refreshed = dict(entry_states)
        for item_key, state in entry_states.items():
            if not state:
                continue
            if str(state.get("status") or "") not in {"review_pending", "codex_reviewing", "approved", "deploying"}:
                continue
            review_id = str(state.get("request_id") or "")
            if not review_id:
                continue
            review = self.store.get_review(review_id)
            if not review:
                continue
            mapped_status = {
                "deployed": "deployed",
                "deploy_failed": "deploy_failed",
                "needs_human": "needs_human",
            }.get(str(review.get("status") or ""))
            if not mapped_status:
                continue
            decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
            summary = str(decision.get("summary") or decision.get("error") or mapped_status)
            self.store.mark_processed_item(task_key, item_key, mapped_status, request_id=review_id, summary=summary)
            refreshed[item_key] = self.store.get_processed_item(task_key, item_key)
        return refreshed

    def _schedule_reuse_plan_retry(
        self,
        task_key: str,
        status_task_key: str,
        item_key: str,
        entry: dict[str, Any],
        chat_id: str,
        status_message_id: str,
        task_title: str,
        task_id: str,
        reason: str,
    ) -> None:
        delay = max(60, int(self.config.polling.reuse_plan_retry_delay_sec))
        due_at = int(time.time()) + delay
        self.store.set_setting(_retry_setting_key(task_key, item_key), str(due_at))
        self.store.mark_processed_item(
            task_key,
            item_key,
            "retry_waiting",
            summary=f"retry_at={due_at}; {reason}",
        )
        self._update_task_status_card(
            status_task_key,
            chat_id,
            "codex",
            "等待中",
            f"当前没有可用 worker，已安排 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(due_at))} 自动重试。",
            title=task_title,
            source_ref=task_id,
            source_message_id=status_message_id,
            model_id=_model_id_from_weight_path(str(entry.get("weight_path") or "")),
            model=str(entry.get("weight_path") or ""),
        )

    def _submit_source(
        self,
        source_key: str,
        item_id: str,
        source: EventSource,
        *,
        chat_id: str = "",
    ) -> PollStats:
        try:
            request = parse_deployment_request(source)
        except ParseError as exc:
            status = "ignored"
            summary = str(exc)
            if _looks_like_deploy_intent(source.text):
                status = "failed_parse"
                if chat_id and self.config.polling.notify_chat_on_accept:
                    self._handle_parse_failure(chat_id, source_key, source, summary)
            self.store.mark_processed_item(source_key, item_id, status, summary=summary)
            return PollStats(scanned=1, ignored=1 if status == "ignored" else 0, failed=1 if status != "ignored" else 0)

        result = self.orchestrator.submit(request)
        self.store.mark_processed_item(
            source_key,
            item_id,
            "submitted",
            request_id=result.request_id,
            summary=result.status.value,
        )
        if chat_id and self.config.polling.notify_chat_on_accept:
            self._send_chat_card_or_text(
                chat_id,
                _deployment_accept_card(
                    result.request_id,
                    result.model_name,
                    result.weight_path,
                    result.status.value,
                ),
                f"已接收部署任务: {result.request_id}\n模型: {result.model_name}\n状态: {result.status.value}",
            )
        return PollStats(scanned=1, submitted=1)

    def _handle_codex_control_command(
        self,
        chat_id: str,
        source_key: str,
        msg_id: str,
        control: str,
    ) -> PollStats:
        key = self.config.codex_review.runtime_toggle_key
        if control == "status":
            value = self.store.get_setting(key)
            enabled = self.config.codex_review.enabled if value is None else value == "true"
        else:
            enabled = control == "on"
            self.store.set_setting(key, "true" if enabled else "false")
        self.store.mark_processed_item(source_key, msg_id, "codex_control", summary=f"codex {control}")
        if chat_id and self.config.polling.notify_chat_on_accept:
            state = "开启" if enabled else "关闭"
            self._send_chat_card_or_text(
                chat_id,
                _status_card("Codex 审核开关", "green" if enabled else "orange", {"状态": state}),
                f"Codex 审核已{state}",
            )
        return PollStats(scanned=1, submitted=1)

    def _handle_help_command(self, chat_id: str, source_key: str, msg_id: str) -> PollStats:
        self.store.mark_processed_item(source_key, msg_id, "help", summary="help requested")
        if chat_id and self.config.polling.notify_chat_on_accept:
            self._send_chat_card_or_text(
                chat_id,
                _help_card(),
                "可用指令：检测任务、检测节点、Codex 开关、重试/取消。",
                reply_to_message_id=msg_id,
            )
        return PollStats(scanned=1, ignored=1)

    def _handle_node_status_command(self, chat_id: str, source_key: str, msg_id: str) -> PollStats:
        self.store.mark_processed_item(source_key, msg_id, "node_status", summary="node status requested")
        if not chat_id or not self.config.polling.notify_chat_on_accept:
            return PollStats(scanned=1, ignored=1)

        doc_token = self.config.reusable_workers.deployed_models_doc_token.strip()
        if not doc_token:
            self._send_chat_card_or_text(
                chat_id,
                _status_card("节点检测失败", "red", {"原因": "未配置已部署模型文档 token"}),
                "节点检测失败：未配置已部署模型文档 token",
                reply_to_message_id=msg_id,
            )
            return PollStats(scanned=1, failed=1)
        try:
            content = self.feishu.get_doc_markdown(doc_token)
            rows = parse_deployed_models_table(content)
        except Exception as exc:
            self._send_chat_card_or_text(
                chat_id,
                _status_card("节点检测失败", "red", {"原因": _short_text(str(exc), 180)}),
                f"节点检测失败：{exc}",
                reply_to_message_id=msg_id,
            )
            return PollStats(scanned=1, failed=1)

        self._send_chat_card_or_text(
            chat_id,
            _node_status_card(rows, self.config.reusable_workers),
            _node_status_fallback(rows, self.config.reusable_workers),
            reply_to_message_id=msg_id,
        )
        return PollStats(scanned=1, ignored=1)

    def _handle_manual_poll_command(self, chat_id: str, source_key: str, msg_id: str) -> PollStats:
        self.store.mark_processed_item(source_key, msg_id, "manual_poll", summary="manual poll requested")
        lookback_sec = max(0, int(self.config.polling.manual_poll_lookback_sec))
        stats = self._poll_chat(chat_id, lookback_sec=lookback_sec)
        if self.config.polling.deploy_todo_subtasks and self.config.polling.watch_known_todo_tasks:
            stats = stats.add(self._poll_known_todo_tasks(force=True))
        if chat_id and self.config.polling.notify_chat_on_accept:
            if stats.failed:
                title = "任务检查有失败"
                color = "red"
                summary = f"检测到 {stats.failed} 个任务读取或解析失败；已按现有规则进入重试或人工处理。"
                fallback = f"任务检查有失败：{stats.failed} 个"
            elif stats.submitted:
                title = "任务检查完成"
                color = "green"
                summary = f"已处理 {stats.submitted} 个新部署项，后续进度会更新到对应模型部署卡片。"
                fallback = f"任务检查完成：已处理 {stats.submitted} 个新任务"
            else:
                title = "目前无新任务"
                color = "grey"
                summary = "最近没有发现新的可部署子任务。"
                fallback = "目前无新任务。"
            self._send_chat_card_or_text(
                chat_id,
                _manual_poll_result_card(
                    title,
                    color,
                    summary,
                    task_lines=stats.task_summaries,
                    recent_lines=_recent_task_status_lines(self.store),
                ),
                fallback,
                reply_to_message_id=msg_id,
            )
        return PollStats(
            scanned=stats.scanned + 1,
            submitted=stats.submitted,
            ignored=0 if stats.submitted or stats.failed else 1,
            failed=stats.failed,
            manual_polls=1,
            task_summaries=stats.task_summaries,
        )

    def _handle_parse_failure(
        self,
        chat_id: str,
        source_key: str,
        source: EventSource,
        summary: str,
    ) -> None:
        issue_key = "parse:" + hashlib.sha1(f"{source_key}:{summary}".encode("utf-8")).hexdigest()[:16]
        issue = self.store.increment_issue_count(issue_key, summary)
        count = int(issue["count"])
        threshold = max(1, self.config.polling.max_parse_failures_before_handoff)
        if count < threshold:
            self._send_chat_card_or_text(
                chat_id,
                _status_card(
                    "部署请求解析失败",
                    "red",
                    {"来源": source.source_ref, "次数": f"{count}/{threshold}", "原因": _short_text(summary, 160)},
                ),
                f"部署请求解析失败: {summary}",
            )
            return
        if count == threshold and not issue["alerted"]:
            self.store.mark_issue_alerted(issue_key)
            self._send_chat_card_or_text(
                chat_id,
                _manual_handoff_card(self.config, "连续解析失败，已停止重复提醒", summary),
                f"连续解析失败，已停止重复提醒: {summary}",
            )
            alert_text = _manual_handoff_text(self.config, "连续解析失败，已停止重复提醒", summary)
            if alert_text:
                self._send_detail_text(chat_id, alert_text)

    def _handle_review_command(
        self,
        chat_id: str,
        source_key: str,
        msg_id: str,
        item: dict[str, Any],
        command: dict[str, str],
    ) -> PollStats:
        requester = _message_requester(item)
        try:
            review = decide_review(
                self.store,
                review_id=command["review_id"],
                decision=command["decision"],
                actor=requester.user_id,
                source="feishu_message",
                note=command.get("note", ""),
            )
        except (KeyError, ValueError) as exc:
            self.store.mark_processed_item(source_key, msg_id, "failed_review_decision", summary=str(exc))
            if chat_id and self.config.polling.notify_chat_on_accept:
                self._send_chat_card_or_text(
                    chat_id,
                    _status_card("审核指令失败", "red", {"原因": str(exc)}),
                    f"审核指令失败: {exc}",
                )
            return PollStats(scanned=1, failed=1)

        self.store.mark_processed_item(
            source_key,
            msg_id,
            "review_decided",
            request_id=command["review_id"],
            summary=command["decision"],
        )
        if chat_id and self.config.polling.notify_chat_on_accept:
            decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
            self._send_chat_card_or_text(
                chat_id,
                review_result_card(review, decision),
                f"审核结果: {command['review_id']} {command['decision']}",
            )
        if command["decision"] in {"APPROVE", "RETRY"}:
            self._wake_review_auditor(1)
        return PollStats(scanned=1, submitted=1)

    def _parse_reply_review_command(self, text: str, item: dict[str, Any]) -> dict[str, str] | None:
        decision = _parse_short_review_decision(text)
        if not decision:
            return None
        reply_targets = _message_reply_target_ids(item)
        if not reply_targets:
            return None
        for review in self.store.list_reviews(limit=200):
            payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
            context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
            message_ids = {
                str(context.get("status_message_id") or ""),
                str(context.get("reply_to_message_id") or ""),
            }
            if message_ids.intersection(reply_targets):
                return {
                    "review_id": str(review.get("review_id") or ""),
                    "decision": decision,
                    "note": "inferred from replied status card",
                }
        return None

    def _chat_ids(self) -> list[str]:
        configured = [chat_id for chat_id in self.config.polling.chat_ids if chat_id]
        if configured:
            return configured
        if self.config.polling.auto_discover_chats:
            discovered = self._auto_discovered_chat_ids()
            if discovered:
                return discovered
        if self.config.feishu.default_chat_id:
            return [self.config.feishu.default_chat_id]
        return []

    def _auto_discovered_chat_ids(self) -> list[str]:
        now = int(time.time())
        interval = max(30, int(self.config.polling.chat_discovery_interval_sec))
        if self._discovered_chat_ids and now - self._discovered_chat_ids_at < interval:
            return self._discovered_chat_ids
        try:
            chats = self.feishu.list_chats(page_size=self.config.polling.page_size)
        except Exception:
            log.exception("failed to auto-discover Feishu chats")
            return self._discovered_chat_ids
        chat_ids = []
        for chat in chats:
            chat_id = str(chat.get("chat_id") or "").strip()
            if chat_id and chat_id not in chat_ids:
                chat_ids.append(chat_id)
        self._discovered_chat_ids = chat_ids
        self._discovered_chat_ids_at = now
        return chat_ids

    def _current_bot_open_id(self) -> str:
        if self._bot_open_id or self._bot_open_id_checked:
            return self._bot_open_id
        self._bot_open_id_checked = True
        try:
            self._bot_open_id = self.feishu.get_bot_open_id().strip()
        except Exception:
            log.exception("failed to resolve Feishu bot open_id")
        return self._bot_open_id

    def _create_reuse_review(
        self,
        entry: dict[str, Any],
        *,
        reserved_row_indices: set[int] | None = None,
    ) -> tuple[str, int]:
        content = self.feishu.get_doc_markdown(self.config.reusable_workers.deployed_models_doc_token)
        rows = parse_deployed_models_table(content)
        plan = build_reusable_deployment_plan(
            content,
            entry["weight_path"],
            self.config.reusable_workers,
            required_gpu_count=0,
            excluded_row_indices=reserved_row_indices,
        )
        packet = make_reuse_plan_review(weight_path=entry["weight_path"], plan=plan, rows=rows)
        payload = packet.to_dict()
        if entry.get("reply_to_message_id"):
            payload.setdefault("context", {})["reply_to_message_id"] = str(entry["reply_to_message_id"])
        if entry.get("source_chat_id"):
            payload.setdefault("context", {})["source_chat_id"] = str(entry["source_chat_id"])
        if entry.get("status_message_id"):
            payload.setdefault("context", {})["status_message_id"] = str(entry["status_message_id"])
        if entry.get("task_key"):
            payload.setdefault("context", {})["task_key"] = str(entry["task_key"])
        if entry.get("status_task_key"):
            payload.setdefault("context", {})["status_task_key"] = str(entry["status_task_key"])
        if entry.get("item_key"):
            payload.setdefault("context", {})["item_key"] = str(entry["item_key"])
        if entry.get("task_title"):
            payload.setdefault("context", {})["task_title"] = str(entry["task_title"])
        if entry.get("parent_task_id"):
            payload.setdefault("context", {})["parent_task_id"] = str(entry["parent_task_id"])
        if entry.get("subtask_guid"):
            payload.setdefault("context", {})["subtask_guid"] = str(entry["subtask_guid"])
        self.store.create_review(packet.review_id, packet.stage.value, packet.subject_id, payload)
        if entry.get("task_key") and entry.get("source_chat_id"):
            self._update_task_status_card(
                str(entry["task_key"]),
                str(entry["source_chat_id"]),
                "codex",
                "待审核",
                f"已生成复用计划 {packet.review_id}。",
                title=str(entry.get("task_title") or ""),
                source_ref=str(packet.subject_id),
                model_id=str(packet.plan.get("path", {}).get("model_id") or ""),
                model=str(packet.plan.get("path", {}).get("table_path") or packet.plan.get("path", {}).get("original_path") or ""),
                worker=str(packet.plan.get("row", {}).get("ip") or ""),
                address=_plan_address(packet.plan),
                source_message_id=str(entry.get("status_message_id") or ""),
            )
        chat_id = str(entry.get("source_chat_id") or "") or self.config.feishu.default_chat_id
        if chat_id and self.config.polling.notify_chat_on_accept and not self.config.reusable_workers.auto_deploy_approved:
            self._send_chat_card_or_text(
                chat_id,
                review_card(packet, self.config.approval, include_actions=False),
                f"待审核部署计划: {packet.review_id}",
                reply_to_message_id=str(entry.get("reply_to_message_id") or ""),
            )
        row = packet.plan.get("row") if isinstance(packet.plan.get("row"), dict) else {}
        return packet.review_id, _safe_int(row.get("row_index") if row else "")

    def _send_chat_card_or_text(
        self,
        chat_id: str,
        card: dict[str, Any],
        fallback_text: str,
        *,
        reply_to_message_id: str = "",
    ) -> None:
        target_chat_id = chat_id or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        try:
            if reply_to_message_id:
                self.feishu.reply_card(reply_to_message_id, card)
            elif target_chat_id:
                self.feishu.send_chat_card(target_chat_id, card)
        except Exception:
            if reply_to_message_id:
                self.feishu.reply_text(reply_to_message_id, fallback_text)
            elif target_chat_id:
                self.feishu.send_chat_text(target_chat_id, fallback_text)

    def _send_detail_text(self, source_chat_id: str, text: str) -> None:
        target_chat_id = source_chat_id or self.config.feishu.default_chat_id or self.config.approval.fallback_chat_id
        if not target_chat_id:
            return
        try:
            self.feishu.send_chat_text(target_chat_id, text)
        except Exception:
            log.exception("failed to send detail text alert")

    def _react_to_detected_task(self, message_id: str) -> None:
        emoji = self.config.polling.task_detected_reaction_emoji.strip()
        if not message_id or not emoji:
            return
        try:
            self.feishu.add_message_reaction(message_id, emoji)
        except Exception:
            log.exception("failed to add detected-task reaction")

    def _wake_review_auditor(self, review_count: int) -> None:
        if not self.config.polling.wake_review_auditor_on_submit:
            return
        session = self.config.runner.tmux_prefix.strip()
        if not session:
            return
        target = f"{session}:review-auditor"
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", target, "-F", "#{pane_pid}"],
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            )
        except Exception:
            log.debug("failed to locate review-auditor tmux pane", exc_info=True)
            return
        if result.returncode != 0:
            log.debug("review-auditor tmux pane not found: %s", result.stderr.strip())
            return
        woke = 0
        for line in result.stdout.splitlines():
            try:
                os.kill(int(line.strip()), signal.SIGUSR1)
            except (OSError, ValueError):
                continue
            woke += 1
        if woke:
            log.info("woke review-auditor for %s new review(s)", review_count)

    def _update_task_status_card(
        self,
        task_key: str,
        source_chat_id: str,
        stage: str,
        status: str,
        detail: str,
        *,
        title: str = "",
        source_ref: str = "",
        source_message_id: str = "",
        model_id: str = "",
        model: str = "",
        worker: str = "",
        address: str = "",
        endpoint: str = "",
        force_new_message: bool = False,
    ) -> dict[str, object]:
        state = {} if force_new_message else self.store.get_task_status(task_key)
        if source_message_id and str(state.get("source_message_id") or "") not in {"", source_message_id}:
            state = {}
        state = task_status_with_stage(
            state,
            stage,
            status,
            detail,
            title=title,
            source_ref=source_ref,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            model_id=model_id,
            model=model,
            worker=worker,
            address=address,
            endpoint=endpoint,
        )
        state["card_actions_enabled"] = self.config.approval.allow_card_actions
        if not source_chat_id:
            self.store.set_task_status(task_key, state)
            return state
        message_id = str(state.get("source_message_id") or "")
        try:
            if message_id:
                self.feishu.update_card(message_id, task_status_card(state))
            else:
                message_id = self.feishu.send_chat_card(source_chat_id, task_status_card(state))
                if message_id:
                    state["source_message_id"] = message_id
        except Exception:
            log.exception("failed to update source task status card")
        self.store.set_task_status(task_key, state)
        return state


def _message_text(item: dict[str, Any]) -> str:
    body = item.get("body") if isinstance(item.get("body"), dict) else {}
    for value in (body.get("content"), item.get("content"), item.get("text")):
        text = _coerce_text(value)
        if text:
            return text
    return ""


def _is_self_or_bot_message(item: dict[str, Any], app_id: str) -> bool:
    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    sender_type = str(sender.get("sender_type") or item.get("sender_type") or "").lower()
    if sender_type in {"app", "bot"}:
        return True
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    return bool(app_id and sender_id.get("app_id") == app_id)


def _parse_codex_control_command(text: str) -> str | None:
    stripped = text.strip().lower()
    mapping = {
        "codex on": "on",
        "codex off": "off",
        "codex status": "status",
        "fmh codex on": "on",
        "fmh codex off": "off",
        "fmh codex status": "status",
        "开启codex": "on",
        "关闭codex": "off",
        "codex状态": "status",
    }
    return mapping.get(stripped)


def _parse_help_command(text: str, *, mentions_current_bot: bool = False) -> bool:
    stripped = _manual_command_text(text)
    if mentions_current_bot and not stripped:
        return True
    if not mentions_current_bot:
        return False
    return stripped in {"help", "?", "帮助", "指令", "命令", "可用指令", "使用帮助"}


def _parse_node_status_command(text: str) -> bool:
    stripped = _manual_command_text(text)
    return stripped in {
        "nodes",
        "node status",
        "workers",
        "worker status",
        "检测节点",
        "检查节点",
        "节点状态",
        "查看节点",
        "查看节点状态",
        "检测worker",
        "检查worker",
        "worker状态",
    }


def _parse_manual_poll_command(text: str) -> bool:
    stripped = _manual_command_text(text)
    commands = {
        "poll",
        "poll now",
        "fmh poll",
        "scan",
        "scan tasks",
        "check tasks",
        "refresh tasks",
        "detect tasks",
        "检查任务",
        "检测任务",
        "查看任务",
        "查任务",
        "刷新任务",
        "触发轮询",
        "轮询",
        "检查子任务",
        "检测子任务",
        "刷新子任务",
        "看一下任务",
    }
    return stripped in commands


def _manual_command_text(text: str) -> str:
    stripped = re.sub(r"<at\b[^>]*>.*?</at>", " ", text, flags=re.IGNORECASE)
    stripped = re.sub(r"<at\b[^>]*/>", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"@\S+", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip().lower()
    return stripped.strip("。.!！ ")


def _contains_at_mention(text: str) -> bool:
    return bool(re.search(r"<at\b", text, flags=re.IGNORECASE) or re.search(r"(^|\s)@\S+", text))


def _mentions_current_bot(item: dict[str, Any], text: str, app_id: str, bot_open_id: str = "") -> bool:
    if not _contains_at_mention(text):
        return False
    current_ids = {value.strip() for value in (app_id, bot_open_id) if value.strip()}
    mentions = _message_mentions(item)
    if current_ids:
        for mention in mentions:
            mentioned_id = str(mention.get("id") or mention.get("app_id") or mention.get("user_id") or "").strip()
            id_type = str(mention.get("id_type") or mention.get("type") or "").lower()
            if mentioned_id in current_ids or (id_type in {"app_id", "open_id"} and mentioned_id in current_ids):
                return True
        text_mention_ids = _at_tag_ids(text)
        if current_ids.intersection(text_mention_ids):
            return True
        if mentions or text_mention_ids:
            return False
    # If Feishu did not include structured mention ids, keep the existing manual
    # @ command behavior visible by acknowledging the command.
    return _parse_manual_poll_command(text)


def _message_mentions(item: dict[str, Any]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for value in (item.get("mentions"), item.get("mention")):
        if isinstance(value, list):
            mentions.extend(mention for mention in value if isinstance(mention, dict))
    content = _message_content_json(item)
    for value in (content.get("mentions"), content.get("mention")):
        if isinstance(value, list):
            mentions.extend(mention for mention in value if isinstance(mention, dict))
    return mentions


def _at_tag_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for match in re.finditer(r"<at\b(?P<attrs>[^>]*)>", text, flags=re.IGNORECASE):
        attrs = match.group("attrs")
        for attr in ("id", "user_id"):
            attr_match = re.search(rf"\b{attr}\s*=\s*['\"]?(?P<id>[^'\"\s>]+)", attrs, flags=re.IGNORECASE)
            if attr_match:
                ids.add(attr_match.group("id"))
    return ids


def _looks_like_card_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and (
        "elements" in parsed
        or "header" in parsed
        or parsed.get("msg_type") == "interactive"
    )


def _todo_task_id(item: dict[str, Any]) -> str:
    content = _message_content_json(item)
    return str(content.get("task_id") or item.get("task_id") or "")


def _message_content_json(item: dict[str, Any]) -> dict[str, Any]:
    body = item.get("body") if isinstance(item.get("body"), dict) else {}
    content = body.get("content") or item.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "content"):
            text = _coerce_text(value.get(key))
            if text:
                return text
        return ""
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return _coerce_text(parsed) or stripped
    return stripped


def _deployment_entries_from_task(
    parent: dict[str, Any],
    subtasks: list[dict[str, Any]],
    relative_prefix: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    parent_name = _task_summary(parent) or "task"
    for subtask in subtasks:
        if _task_is_completed(subtask):
            continue
        subtask_guid = str(subtask.get("guid") or subtask.get("task_id") or "")
        raw_text = "\n".join(
            part
            for part in (_task_summary(subtask), _task_description(subtask))
            if part
        )
        for line in _candidate_lines(raw_text):
            weight_path = _extract_weight_path_from_line(line, relative_prefix)
            if not weight_path:
                continue
            model_name = _model_name_from_task_line(line, weight_path, subtask, parent_name)
            entries.append(
                {
                    "subtask_guid": subtask_guid or hashlib.sha1(line.encode("utf-8")).hexdigest()[:12],
                    "weight_path": weight_path,
                    "model_name": model_name,
                    "raw_line": line,
                    "parent": parent,
                    "subtask": subtask,
                }
            )
    return entries


def _deployment_text_from_entry(entry: dict[str, Any]) -> str:
    return "\n".join(
        [
            "deploy_vllm",
            f"weight_path: {entry['weight_path']}",
            f"model_name: {entry['model_name']}",
            f"task_line: {entry['raw_line']}",
        ]
    )


def _task_summary(task: dict[str, Any]) -> str:
    summary = task.get("summary")
    return _rich_text_to_plain(summary).strip()


def _task_description(task: dict[str, Any]) -> str:
    description = task.get("description")
    return _rich_text_to_plain(description).strip()


def _task_is_completed(task: dict[str, Any]) -> bool:
    for key in ("completed_at", "completed_time", "done_at", "finished_at", "finish_time"):
        if _truthy_completion_value(task.get(key)):
            return True
    for key in ("is_completed", "completed", "done", "is_done", "finished"):
        value = task.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "y"}:
            return True
    for key in ("status", "task_status", "complete_status"):
        text = _rich_text_to_plain(task.get(key)).strip().lower()
        if text in {"completed", "complete", "done", "finished", "closed", "已完成", "完成"}:
            return True
    return False


def _truthy_completion_value(value: Any) -> bool:
    if value in (None, "", 0, "0", False):
        return False
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "false"}:
        return False
    return True


def _rich_text_to_plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("title"), str):
            return value["title"]
        if "content" in value:
            return _rich_text_to_plain(value["content"])
        return " ".join(_rich_text_to_plain(v) for v in value.values()).strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _rich_text_to_plain(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(value)


def _candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("-* \t")
        if not line:
            continue
        lines.append(line)
    return lines


def _extract_weight_path_from_line(line: str, relative_prefix: str) -> str:
    explicit = re.search(r"(?:weight_path|model_path|ckpt|checkpoint|权重路径|路径)\s*[:=：]\s*(?P<path>\S+)", line)
    if explicit:
        return explicit.group("path").strip("`'\"，,")
    storage = re.search(
        r"(?:^|\s)(?P<path>(?:/|s3://|oss://|hdfs://|hf://|gs://)[^\s`'\"，,]+)",
        line,
    )
    if storage:
        return storage.group("path").strip("`'\"，,")
    if relative_prefix and _looks_like_relative_model_path(line):
        return relative_prefix.rstrip("/") + "/" + line.strip("`'\"，,").lstrip("/")
    return ""


def _looks_like_relative_model_path(line: str) -> bool:
    if " " in line or "\t" in line:
        return False
    if line.startswith(("http://", "https://")):
        return False
    if "/" not in line:
        return False
    return bool(re.match(r"^[A-Za-z0-9_.@/-]+$", line))


def _model_name_from_task_line(
    line: str,
    weight_path: str,
    subtask: dict[str, Any],
    parent_name: str,
) -> str:
    summary = _task_summary(subtask)
    candidate = summary or line or weight_path.rstrip("/").rsplit("/", 1)[-1] or parent_name
    candidate = candidate.replace(weight_path, "").strip(" -:：")
    if not candidate:
        candidate = weight_path.rstrip("/").rsplit("/", 1)[-1] or parent_name
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate).strip("-") or "model"


def _message_requester(item: dict[str, Any]) -> Requester:
    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    user_id = (
        sender_id.get("open_id")
        or sender_id.get("user_id")
        or sender_id.get("union_id")
        or "unknown"
    )
    display_name = sender.get("sender_name") or sender.get("name") or ""
    return Requester(user_id=str(user_id), display_name=str(display_name))


def _parse_short_review_decision(text: str) -> str:
    stripped = re.sub(r"<at[^>]*>.*?</at>", "", text).strip().lower()
    stripped = stripped.strip("。.!！ ")
    decisions = {
        "retry": "RETRY",
        "重试": "RETRY",
        "cancel": "BLOCK",
        "取消": "BLOCK",
        "block": "BLOCK",
        "阻止": "BLOCK",
    }
    return decisions.get(stripped, "")


def _message_reply_target_ids(item: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for key in ("parent_id", "root_id", "thread_id"):
        value = item.get(key)
        if value:
            targets.add(str(value))
    body = item.get("body") if isinstance(item.get("body"), dict) else {}
    for key in ("parent_id", "root_id"):
        value = body.get(key)
        if value:
            targets.add(str(value))
    return targets


def _message_create_time(item: dict[str, Any]) -> int:
    raw = item.get("create_time") or item.get("update_time") or 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    if value > 10_000_000_000:
        value = value // 1000
    return value


def _looks_like_deploy_intent(text: str) -> bool:
    return any(word in text for word in ("deploy_vllm", "部署模型", "启动模型", "weight_path", "权重路径"))


def _manual_handoff_card(config: AppConfig, title: str, reason: str) -> dict[str, Any]:
    mention = ""
    if config.approval.fallback_mention_open_id:
        name = config.approval.fallback_mention_name or config.approval.fallback_mention_open_id
        mention = f"<at id={config.approval.fallback_mention_open_id}>{name}</at>"
    elif config.approval.fallback_mention_name:
        mention = f"@{config.approval.fallback_mention_name}"
    fields = {"处理人": mention or "人工", "原因": _short_text(reason, 160)}
    return _status_card(title, "orange", fields)


def _manual_handoff_text(config: AppConfig, title: str, reason: str) -> str:
    mention = mention_text(config.approval)
    if not mention:
        return ""
    return f"{mention} {title}\n原因：{_short_text(reason, 220)}"


def _task_title(parent: dict[str, Any], task_id: str) -> str:
    summary = _task_summary(parent)
    if summary:
        return summary
    for key in ("summary", "title", "name"):
        value = _rich_text_to_plain(parent.get(key)).strip()
        if value:
            return value
    return task_id


def _task_entry_key(task_id: str, entry: dict[str, Any]) -> str:
    return f"{task_id}:{entry['subtask_guid']}:{entry['weight_path']}"


def _task_item_status_key(task_key: str, item_key: str) -> str:
    digest = hashlib.sha1(item_key.encode("utf-8")).hexdigest()[:16]
    return f"{task_key}:item:{digest}"


def _task_entry_status_counts(
    entry_states: dict[str, dict[str, object] | None],
    pending_entry_keys: list[str],
) -> dict[str, int]:
    pending = set(pending_entry_keys)
    counts = {
        "new": 0,
        "retry": 0,
        "done": 0,
        "active": 0,
        "other": 0,
    }
    for item_key, state in entry_states.items():
        if state is None:
            if item_key in pending:
                counts["new"] += 1
            continue
        status = str(state.get("status") or "")
        if item_key in pending and status == "retry_waiting":
            counts["retry"] += 1
        elif status in {"deployed", "deploy_failed", "needs_human", "failed_review", "failed_parse"}:
            counts["done"] += 1
        elif status in {"review_pending", "codex_reviewing", "approved", "deploying", "retry_waiting", "submitted"}:
            counts["active"] += 1
        else:
            counts["other"] += 1
    return counts


def _task_detection_detail(total: int, pending: int, counts: dict[str, int]) -> str:
    parts = [f"发现 {total} 个候选子任务", f"本轮处理 {pending} 个"]
    handling_parts = []
    if counts.get("new", 0):
        handling_parts.append(f"新增 {counts['new']}")
    if counts.get("retry", 0):
        handling_parts.append(f"到期重试 {counts['retry']}")
    if handling_parts:
        parts[-1] += f"（{'，'.join(handling_parts)}）"
    if counts.get("done", 0):
        parts.append(f"已处理 {counts['done']} 个")
    if counts.get("active", 0):
        parts.append(f"处理中/等待 {counts['active']} 个")
    if counts.get("other", 0):
        parts.append(f"已跳过 {counts['other']} 个")
    return "；".join(parts) + "。"


def _task_summary_line(task_title: str, weight_path: str) -> str:
    model_id = _model_id_from_weight_path(weight_path)
    if task_title and model_id:
        return f"{task_title} · {model_id}"
    return task_title or model_id or "未命名任务"


def _merge_unique(first: tuple[str, ...], second: tuple[str, ...], *, limit: int = 6) -> tuple[str, ...]:
    merged: list[str] = []
    for item in [*first, *second]:
        item = str(item).strip()
        if item and item not in merged:
            merged.append(item)
        if len(merged) >= limit:
            break
    return tuple(merged)


def _review_plan_row(review: dict[str, object]) -> dict[str, object]:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
    return row


def _retry_setting_key(task_key: str, item_key: str) -> str:
    digest = hashlib.sha1(f"{task_key}:{item_key}".encode("utf-8")).hexdigest()[:20]
    return f"retry_at:{digest}"


def _known_todo_checked_key(task_id: str) -> str:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:20]
    return f"known_todo_checked_at:{digest}"


def _is_no_reusable_worker_error(exc: Exception) -> bool:
    return "no reusable deployed-model row is available" in str(exc)


def _safe_int(value: object) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _model_id_from_weight_path(weight_path: str) -> str:
    return weight_path.rstrip("/").rsplit("/", 1)[-1] if weight_path.strip() else ""


def _plan_address(plan: dict[str, Any]) -> str:
    row = plan.get("row") if isinstance(plan.get("row"), dict) else {}
    ip = str(row.get("ip") or "").strip()
    gpu_count = str(row.get("gpu_count") or "").strip()
    if ip and gpu_count:
        return f"{ip} ({gpu_count}卡)"
    return ip or str(row.get("address") or "").strip()


def _deployment_accept_card(request_id: str, model_name: str, weight_path: str, status: str) -> dict[str, Any]:
    return _status_card(
        "已接收部署任务",
        "blue",
        {
            "request_id": request_id,
            "模型": model_name,
            "状态": status,
            "权重路径": weight_path,
        },
    )


def _todo_accept_card(count: int, request_ids: list[str], *, review_mode: bool = False) -> dict[str, Any]:
    fields = {
        "子任务数": str(count),
        "类型": "待审核部署计划" if review_mode else "部署请求",
        "id": "\n".join(f"- {request_id}" for request_id in request_ids[:10]),
    }
    if len(request_ids) > 10:
        fields["更多"] = f"还有 {len(request_ids) - 10} 个任务未显示"
    return _status_card("已接收任务子项部署", "blue", fields)


def _help_card() -> dict[str, Any]:
    command_lines = [
        "**检测任务** · 扫描最近任务分享和已跟踪子任务",
        "**检测节点** · 查看已部署模型文档里的 worker 可用情况",
        "**codex on/off/status** · 开关或查看 Codex 审核",
        "**回复卡片：重试 / 取消** · 处理失败部署",
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "可用指令"},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": "\n".join(command_lines)},
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "建议在群里 @bot 后发送指令，回复会留在当前群。",
                    }
                ],
            },
        ],
    }


def _node_status_card(rows: list[Any], config: Any) -> dict[str, Any]:
    counts = _node_status_counts(rows, config)
    available = counts["idle"] + counts["reusable"]
    valid_total = max(0, len(rows) - counts["invalid"])
    color = "green" if available else "orange" if rows else "red"
    summary_parts = [
        f"**可用节点** {_tag(str(available), 'green' if available else 'grey')}",
        f"**总节点** {valid_total}",
    ]
    if counts["running"]:
        summary_parts.append(f"运行中 {counts['running']}")
    if counts["fresh"]:
        summary_parts.append(f"待测试 {counts['fresh']}")
    if counts["partial"]:
        summary_parts.append(f"测试未完成 {counts['partial']}")
    if counts["invalid"]:
        summary_parts.append(f"配置异常 {counts['invalid']}")
    if not rows:
        detail = "未从已部署模型文档解析到节点表。"
    else:
        lines = [_node_status_line(row, config) for row in rows[:30]]
        if len(rows) > 30:
            lines.append(f"还有 {len(rows) - 30} 个节点未显示。")
        detail = "**节点明细**\n" + "\n".join(lines)
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "节点状态"},
            "template": color,
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": " · ".join(summary_parts)},
            },
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": detail},
            },
        ],
    }


def _node_status_fallback(rows: list[Any], config: Any) -> str:
    counts = _node_status_counts(rows, config)
    available = counts["idle"] + counts["reusable"]
    return f"节点状态：可用 {available} / 总节点 {max(0, len(rows) - counts['invalid'])}"


def _node_status_counts(rows: list[Any], config: Any) -> dict[str, int]:
    counts = {"idle": 0, "reusable": 0, "running": 0, "fresh": 0, "partial": 0, "invalid": 0}
    for row in rows:
        key, _, _, _ = _node_row_status(row, config)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _node_status_line(row: Any, config: Any) -> str:
    _, label, color, detail = _node_row_status(row, config)
    ip = str(getattr(row, "ip", "") or "").strip()
    gpu_count = int(getattr(row, "gpu_count", 0) or 0)
    address = f"{ip} ({gpu_count}卡)" if ip and gpu_count else _short_text(str(getattr(row, "address", "") or "地址缺失"), 32)
    model_id = str(getattr(row, "model_id", "") or "").strip()
    if not model_id:
        model_id = _short_text(str(getattr(row, "model", "") or "-"), 42)
    return f"{_tag(label, color)} {_md_escape(address)} · {_md_escape(_short_text(model_id or '-', 42))} · {_md_escape(detail)}"


def _node_row_status(row: Any, config: Any) -> tuple[str, str, str, str]:
    ip = str(getattr(row, "ip", "") or "").strip()
    gpu_count = int(getattr(row, "gpu_count", 0) or 0)
    if not ip or gpu_count <= 0:
        return "invalid", "配置异常", "red", "地址或卡数缺失"
    if row.has_running_marker(config.running_marker):
        return "running", "运行中", "blue", "含 running 标记"
    if row.is_idle_empty():
        return "idle", "空闲", "green", "空行可用"
    if row.is_reusable(config):
        finished = "/".join(str(task).strip() for task in config.required_finished_tasks if str(task).strip())
        return "reusable", "可复用", "green", f"已完成 {finished}" if finished else "已测试完成"
    if row.is_fresh_untested():
        return "fresh", "待测试", "orange", "新部署未测试"
    return "partial", "测试未完成", "grey", _short_text(str(getattr(row, "tested_tasks", "") or "未满足 required tasks"), 48)


def _manual_poll_result_card(
    title: str,
    color: str,
    summary: str,
    *,
    task_lines: tuple[str, ...] = (),
    recent_lines: tuple[str, ...] = (),
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": summary},
        }
    ]
    if task_lines:
        content = "\n".join(f"- {_short_text(line, 88)}" for line in task_lines[:4])
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**本轮处理**\n{content}"},
            }
        )
    elif recent_lines:
        content = "\n".join(f"- {_short_text(line, 88)}" for line in recent_lines[:3])
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**最近任务**\n{content}"},
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": elements,
    }


def _status_card(title: str, color: str, fields: dict[str, str]) -> dict[str, Any]:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": key not in {"权重路径", "原因", "request_id"},
                        "text": {"tag": "lark_md", "content": f"**{key}**\n{value}"},
                    }
                    for key, value in fields.items()
                ],
            }
        ],
    }


def _short_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _tag(text: str, color: str) -> str:
    return f"<text_tag color='{color}'>{_md_escape(text)}</text_tag>"


def _md_escape(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\")
    for char in ("*", "_", "~", "`", "[", "]", "(", ")"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def _recent_task_status_lines(store: StateStore, *, limit: int = 3) -> tuple[str, ...]:
    with store._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            """
            SELECT value FROM runtime_settings
            WHERE key LIKE 'task_status:%:item:%'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit * 4,),
        ).fetchall()
    lines: list[str] = []
    for row in rows:
        try:
            state = json.loads(str(row["value"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(state, dict):
            continue
        title = str(state.get("title") or "").strip()
        model_id = str(state.get("model_id") or "").strip()
        if not model_id:
            model_id = _model_id_from_weight_path(str(state.get("model") or ""))
        status = str(state.get("deploy_status") or "").strip()
        parts = [part for part in (title, model_id, status) if part]
        line = " · ".join(parts)
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            break
    return tuple(lines)
