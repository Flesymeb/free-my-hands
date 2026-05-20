from __future__ import annotations

import json
import signal
import subprocess
import time
from typing import Any

from fmh.config import AppConfig, FeishuConfig, PollingConfig, ReusableWorkersConfig, RunnerConfig, StorageConfig, VLLMConfig
from fmh.models import RequestStatus
from fmh.orchestrator import DeploymentOrchestrator
from fmh.poller import FeishuPollingWorker, _task_item_status_key
from fmh.runner import make_runner
from fmh.store import StateStore


class FakeFeishuClient:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        task: dict[str, Any] | None = None,
        subtasks: list[dict[str, Any]] | None = None,
        doc_markdown: str = "",
    ) -> None:
        self.messages = messages
        self.task = task or {}
        self.subtasks = subtasks or []
        self.doc_markdown = doc_markdown
        self.chat_texts: list[tuple[str, str]] = []
        self.sent_cards: list[dict[str, Any]] = []
        self.patched_cards: list[tuple[str, dict[str, Any]]] = []
        self.reactions: list[tuple[str, str]] = []
        self.requested_tasks: list[str] = []
        self._message_counter = 0

    def list_messages(
        self,
        chat_id: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        page_size: int = 50,
        sort_type: str = "ByCreateTimeAsc",
    ) -> list[dict[str, Any]]:
        out = []
        for message in self.messages:
            ts = int(message["create_time"]) // 1000
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue
            out.append(message)
        return out

    def get_document_raw_content(self, document_id: str) -> str:
        return ""

    def get_doc_markdown(self, doc_token: str, doc_type: str = "docx") -> str:
        return self.doc_markdown

    def get_task(self, task_guid: str) -> dict[str, Any]:
        self.requested_tasks.append(task_guid)
        return self.task

    def list_subtasks(self, task_guid: str, page_size: int = 50) -> list[dict[str, Any]]:
        return self.subtasks

    def complete_task(self, task_guid: str) -> None:
        return None

    def send_chat_text(self, chat_id: str, text: str) -> str:
        self.chat_texts.append((chat_id, text))
        return self._next_message_id()

    def send_chat_card(self, chat_id: str, card: dict[str, Any]) -> str:
        self.sent_cards.append(card)
        title = card.get("header", {}).get("title", {}).get("content", "card")
        self.chat_texts.append((chat_id, str(title)))
        return self._next_message_id()

    def reply_text(self, message_id: str, text: str) -> str:
        self.chat_texts.append((message_id, text))
        return self._next_message_id()

    def reply_card(self, message_id: str, card: dict[str, Any]) -> str:
        self.sent_cards.append(card)
        title = card.get("header", {}).get("title", {}).get("content", "card")
        self.chat_texts.append((message_id, str(title)))
        return self._next_message_id()

    def update_card(self, message_id: str, card: dict[str, Any]) -> None:
        self.patched_cards.append((message_id, card))

    def add_message_reaction(self, message_id: str, emoji_type: str) -> str:
        self.reactions.append((message_id, emoji_type))
        return f"reaction_{len(self.reactions)}"

    def send_private_text(self, open_id: str, text: str) -> str:
        return ""

    def send_private_card(self, open_id: str, card: dict[str, object]) -> str:
        return ""

    def update_document_status(self, request, text: str) -> None:
        return None

    def _next_message_id(self) -> str:
        self._message_counter += 1
        return f"om_fake_{self._message_counter}"


def test_polling_submits_chat_deployment_once(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_1",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {
                    "content": '{"text":"deploy_vllm\\nweight_path: /mnt/models/demo\\nmodel_name: demo\\ngpu_count: 1"}'
                },
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    first = worker.poll_once(lookback_sec=60)
    second = worker.poll_once(lookback_sec=60)

    requests = store.list_requests()
    assert first.submitted == 1
    assert second.submitted == 0
    assert len(requests) == 1
    assert requests[0].status == RequestStatus.DRY_RUN_COMPLETE
    assert fake.chat_texts


def test_polling_todo_subtasks_submit_deployments(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_1"}'},
            }
        ],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[
            {
                "guid": "sub_1",
                "summary": "/mnt/models/model-a",
                "description": "",
            },
            {
                "guid": "sub_2",
                "summary": "owner/model-b",
                "description": "",
            },
        ],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(
            chat_ids=["oc_1"],
            notify_chat_on_accept=True,
            relative_weight_path_prefix="/mnt/relative",
        ),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)
    requests = store.list_requests()

    assert stats.submitted == 2
    assert len(requests) == 2
    assert {request.weight_path for request in requests} == {
        "/mnt/models/model-a",
        "/mnt/relative/owner/model-b",
    }
    item_a = "task_1:sub_1:/mnt/models/model-a"
    item_b = "task_1:sub_2:/mnt/relative/owner/model-b"
    task_status_a = store.get_task_status(_task_item_status_key("todo:task_1", item_a))
    task_status_b = store.get_task_status(_task_item_status_key("todo:task_1", item_b))
    assert task_status_a["source_message_id"].startswith("om_fake_")
    assert task_status_b["source_message_id"].startswith("om_fake_")
    assert task_status_a["source_message_id"] != task_status_b["source_message_id"]
    assert fake.patched_cards
    assert fake.reactions == [("om_todo", "SALUTE")]


def test_polling_todo_subtasks_skips_completed_subtasks(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_1"}'},
            }
        ],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[
            {"guid": "sub_done", "summary": "/mnt/models/model-a", "description": "", "completed_at": str(now_ms)},
            {"guid": "sub_active", "summary": "/mnt/models/model-b", "description": "", "completed_at": "0"},
        ],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)
    requests = store.list_requests()

    assert stats.submitted == 1
    assert [request.weight_path for request in requests] == ["/mnt/models/model-b"]
    assert store.get_processed_item("todo:task_1", "task_1:sub_done:/mnt/models/model-a") is None


def test_manual_at_command_scans_recent_task_share(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    old_ms = now_ms - 120_000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(old_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_1"}'},
            },
            {
                "message_id": "om_at",
                "msg_type": "text",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"text":"<at id=\\"ou_bot\\"></at> 检测任务"}'},
            },
        ],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[{"guid": "sub_1", "summary": "/mnt/models/model-a", "description": ""}],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True, manual_poll_lookback_sec=600),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    # Simulate the normal cursor having advanced past the task share but before
    # the manual @ command.
    store.set_cursor("chat:oc_1", str((now_ms // 1000) - 10))
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once()
    requests = store.list_requests()

    assert stats.submitted == 1
    assert [request.weight_path for request in requests] == ["/mnt/models/model-a"]
    assert store.get_processed_item("chat:oc_1", "om_at")["status"] == "manual_poll"
    assert store.get_processed_item("chat:oc_1", "om_todo")["status"] == "submitted"
    assert fake.chat_texts[-1][0] == "om_at"
    assert fake.chat_texts[-1][1] == "任务检查完成"
    manual_card = json.dumps(fake.sent_cards[-1], ensure_ascii=False)
    assert "本轮处理" in manual_card
    assert "deploy batch · model-a" in manual_card
    assert "扫描" not in manual_card
    assert "忽略" not in manual_card


def test_manual_at_command_reports_no_new_tasks(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_at",
                "msg_type": "text",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"text":"@模型部署bot 检测任务"}'},
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True, manual_poll_lookback_sec=600),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    store.set_cursor("chat:oc_1", str((now_ms // 1000) - 10))
    store.set_task_status(
        "todo:task_1:item:model_a",
        {"title": "历史任务", "model_id": "model-a", "deploy_status": "已部署"},
    )
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once()

    assert stats.submitted == 0
    assert stats.ignored == 1
    assert store.get_processed_item("chat:oc_1", "om_at")["status"] == "manual_poll"
    assert fake.chat_texts[-1] == ("om_at", "目前无新任务")
    manual_card = json.dumps(fake.sent_cards[-1], ensure_ascii=False)
    assert "最近任务" in manual_card
    assert "历史任务 · model-a · 已部署" in manual_card
    assert "扫描" not in manual_card
    assert "提交" not in manual_card


def test_manual_at_command_limits_known_task_rescan(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_at",
                "msg_type": "text",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"text":"@模型部署bot 检测任务"}'},
            }
        ],
        task={"guid": "task", "summary": "deploy batch"},
        subtasks=[],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(
            chat_ids=["oc_1"],
            notify_chat_on_accept=True,
            manual_poll_lookback_sec=600,
            known_todo_max_per_tick=1,
        ),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    store.set_cursor("chat:oc_1", str((now_ms // 1000) - 10))
    store.mark_processed_item("todo:task_a", "task_a:sub_a:/mnt/models/a", "deployed")
    store.mark_processed_item("todo:task_b", "task_b:sub_b:/mnt/models/b", "deployed")
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    worker.poll_once()

    assert len(fake.requested_tasks) == 1


def test_polling_known_todo_task_treats_changed_subtask_as_new_card(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_1"}'},
            }
        ],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[
            {
                "guid": "sub_1",
                "summary": "/mnt/models/model-a",
                "description": "",
            }
        ],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    first = worker.poll_once(lookback_sec=60)
    first_item_key = "task_1:sub_1:/mnt/models/model-a"
    first_status_message_id = str(
        store.get_task_status(_task_item_status_key("todo:task_1", first_item_key)).get("source_message_id") or ""
    )
    fake.subtasks[0]["summary"] = "/mnt/models/model-b"
    second = worker.poll_once(lookback_sec=60)
    second_item_key = "task_1:sub_1:/mnt/models/model-b"
    second_status_message_id = str(
        store.get_task_status(_task_item_status_key("todo:task_1", second_item_key)).get("source_message_id") or ""
    )
    third = worker.poll_once(lookback_sec=60)

    requests = store.list_requests()
    assert first.submitted == 1
    assert second.submitted == 1
    assert third.submitted == 0
    assert first_status_message_id
    assert second_status_message_id
    assert second_status_message_id != first_status_message_id
    assert {request.weight_path for request in requests} == {
        "/mnt/models/model-a",
        "/mnt/models/model-b",
    }


def test_polling_known_todo_task_picks_up_new_subtask_as_new_card(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_1"}'},
            }
        ],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[
            {
                "guid": "sub_1",
                "summary": "/mnt/models/model-a",
                "description": "",
            }
        ],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    first = worker.poll_once(lookback_sec=60)
    first_item_key = "task_1:sub_1:/mnt/models/model-a"
    first_status_message_id = str(
        store.get_task_status(_task_item_status_key("todo:task_1", first_item_key)).get("source_message_id") or ""
    )
    fake.subtasks.append({"guid": "sub_2", "summary": "/mnt/models/model-b", "description": ""})
    second = worker.poll_once(lookback_sec=60)
    second_item_key = "task_1:sub_2:/mnt/models/model-b"
    second_status_message_id = str(
        store.get_task_status(_task_item_status_key("todo:task_1", second_item_key)).get("source_message_id") or ""
    )

    assert first.submitted == 1
    assert second.submitted == 1
    assert first_status_message_id
    assert second_status_message_id
    assert second_status_message_id != first_status_message_id


def test_polling_todo_task_skips_done_and_active_entries_processes_new_only(tmp_path) -> None:
    fake = FakeFeishuClient(
        [],
        task={"guid": "task_1", "summary": "deploy batch"},
        subtasks=[
            {"guid": "sub_done", "summary": "/mnt/models/model-a", "description": ""},
            {"guid": "sub_active", "summary": "/mnt/models/model-b", "description": ""},
            {"guid": "sub_new", "summary": "/mnt/models/model-c", "description": ""},
        ],
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        vllm=VLLMConfig(command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    store.create_review(
        "rvw-done",
        "reuse_row_selected",
        "192.0.2.1:model-a",
        {"plan": {"row": {"row_index": 1}}},
    )
    store.decide_review("rvw-done", "deployed", {"summary": "done"})
    store.mark_processed_item(
        "todo:task_1",
        "task_1:sub_done:/mnt/models/model-a",
        "review_pending",
        request_id="rvw-done",
    )
    store.mark_processed_item("todo:task_1", "task_1:sub_active:/mnt/models/model-b", "deploying")
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats, submitted_ids, failed = worker._process_task_subtasks(  # noqa: SLF001
        "task_1",
        "todo:task_1",
        chat_id="oc_1",
    )

    requests = store.list_requests()
    new_item_key = "task_1:sub_new:/mnt/models/model-c"
    detail = store.get_task_status(_task_item_status_key("todo:task_1", new_item_key))["stages"]["detect"]["detail"]
    assert stats.submitted == 1
    assert failed == 0
    assert len(submitted_ids) == 1
    assert [request.weight_path for request in requests] == ["/mnt/models/model-c"]
    assert store.get_processed_item("todo:task_1", "task_1:sub_done:/mnt/models/model-a")["status"] == "deployed"
    assert store.get_processed_item("todo:task_1", "task_1:sub_active:/mnt/models/model-b")["status"] == "deploying"
    assert "本轮处理 1 个（新增 1）" in detail
    assert "已处理 1 个" in detail
    assert "处理中/等待 1 个" in detail


def test_reusable_todo_without_available_worker_retries_later(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    unavailable_doc = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/running</td><td>running</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td>old/running-fullwidth</td><td>running-fullwidth</td><td>192\\.0\\.2\\.15（4卡）</td><td></td><td></td><td></td><td>tau2\nvita（running）</td><td></td></tr>
<tr><td>old/false-positive</td><td>false-positive</td><td>192\\.0\\.2\\.16（4卡）</td><td></td><td></td><td></td><td>tau20\nvitamin</td><td></td></tr>
<tr><td>fresh/model</td><td>model</td><td>192\\.0\\.2\\.13（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>"""
    available_doc = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.14（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>"""
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_todo",
                "msg_type": "todo",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"task_id":"task_retry"}'},
            }
        ],
        task={"guid": "task_retry", "summary": "deploy retry batch"},
        subtasks=[{"guid": "sub_1", "summary": "/mnt/models/model-a", "description": ""}],
        doc_markdown=unavailable_doc,
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(
            chat_ids=["oc_1"],
            notify_chat_on_accept=True,
            reuse_plan_retry_delay_sec=1800,
        ),
        reusable_workers=ReusableWorkersConfig(enabled=True),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    first = worker.poll_once(lookback_sec=60)
    item_key = "task_retry:sub_1:/mnt/models/model-a"
    status_key = _task_item_status_key("todo:task_retry", item_key)
    first_status_message_id = str(store.get_task_status(status_key).get("source_message_id") or "")
    processed = store.get_processed_item("todo:task_retry", item_key)
    retry_settings = [
        key
        for key in _runtime_setting_keys(store)
        if key.startswith("retry_at:")
    ]

    assert first.submitted == 0
    assert first.failed == 0
    assert processed is not None
    assert processed["status"] == "retry_waiting"
    assert store.list_reviews() == []
    assert retry_settings
    state = store.get_task_status(status_key)
    assert state["stages"]["codex"]["status"] == "等待中"
    assert "自动重试" in state["stages"]["codex"]["detail"]

    second = worker.poll_once(lookback_sec=60)
    assert second.submitted == 0
    assert store.get_processed_item("todo:task_retry", item_key)["status"] == "retry_waiting"
    assert store.list_reviews() == []

    store.set_setting(retry_settings[0], "0")
    fake.doc_markdown = available_doc
    third = worker.poll_once(lookback_sec=60)
    third_status_message_id = str(store.get_task_status(status_key).get("source_message_id") or "")
    processed_after_retry = store.get_processed_item("todo:task_retry", item_key)
    reviews = store.list_reviews()

    assert third.submitted == 1
    assert processed_after_retry is not None
    assert processed_after_retry["status"] == "review_pending"
    assert len(reviews) == 1
    assert reviews[0]["status"] == "pending"
    assert reviews[0]["payload"]["plan"]["row"]["ip"] == "192.0.2.14"
    assert third_status_message_id == first_status_message_id


def test_reusable_todo_multiple_new_subtasks_reserve_distinct_rows(tmp_path) -> None:
    doc_markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/a</td><td>a</td><td>192\\.0\\.2\\.14（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
<tr><td>old/b</td><td>b</td><td>192\\.0\\.2\\.15（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    fake = FakeFeishuClient(
        [],
        task={"guid": "task_multi", "summary": "deploy multi"},
        subtasks=[
            {"guid": "sub_1", "summary": "/mnt/models/model-a", "description": ""},
            {"guid": "sub_2", "summary": "/mnt/models/model-b", "description": ""},
        ],
        doc_markdown=doc_markdown,
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        reusable_workers=ReusableWorkersConfig(enabled=True),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats, submitted_ids, failed = worker._process_task_subtasks(  # noqa: SLF001
        "task_multi",
        "todo:task_multi",
        chat_id="oc_1",
    )

    reviews = store.list_reviews(limit=10)
    selected_ips = {review["payload"]["plan"]["row"]["ip"] for review in reviews}
    status_message_ids = {review["payload"]["context"]["status_message_id"] for review in reviews}
    status_task_keys = {review["payload"]["context"]["status_task_key"] for review in reviews}
    assert stats.submitted == 2
    assert failed == 0
    assert len(submitted_ids) == 2
    assert selected_ips == {"192.0.2.14", "192.0.2.15"}
    assert len(status_message_ids) == 2
    assert len(status_task_keys) == 2


def test_reusable_review_notice_uses_source_chat_when_default_chat_differs(tmp_path) -> None:
    doc_markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/a</td><td>a</td><td>192\\.0\\.2\\.14（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    fake = FakeFeishuClient(
        [],
        task={"guid": "task_source", "summary": "deploy source"},
        subtasks=[{"guid": "sub_1", "summary": "/mnt/models/model-a", "description": ""}],
        doc_markdown=doc_markdown,
    )
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_default"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_source"], notify_chat_on_accept=True),
        reusable_workers=ReusableWorkersConfig(enabled=True),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    worker._process_task_subtasks("task_source", "todo:task_source", chat_id="oc_source")  # noqa: SLF001

    assert any(target == "oc_source" and text == "复用常驻 worker 部署审核" for target, text in fake.chat_texts)
    assert not any(target == "oc_default" for target, _ in fake.chat_texts)


def test_reusable_todo_wakes_review_auditor_after_new_review(tmp_path, monkeypatch) -> None:
    doc_markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.14（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>"""
    fake = FakeFeishuClient(
        [],
        task={"guid": "task_wake", "summary": "deploy wake"},
        subtasks=[{"guid": "sub_1", "summary": "/mnt/models/model-a", "description": ""}],
        doc_markdown=doc_markdown,
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs"), tmux_prefix="fmh-test"),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
        reusable_workers=ReusableWorkersConfig(enabled=True),
    )
    kill_calls: list[tuple[int, int]] = []
    tmux_calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        tmux_calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="1234\n", stderr="")

    def fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    monkeypatch.setattr("fmh.poller.subprocess.run", fake_run)
    monkeypatch.setattr("fmh.poller.os.kill", fake_kill)
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats, submitted_ids, failed = worker._process_task_subtasks(  # noqa: SLF001
        "task_wake",
        "todo:task_wake",
        chat_id="oc_1",
    )

    assert stats.submitted == 1
    assert failed == 0
    assert len(submitted_ids) == 1
    assert tmux_calls == [["tmux", "list-panes", "-t", "fmh-test:review-auditor", "-F", "#{pane_pid}"]]
    assert kill_calls == [(1234, signal.SIGUSR1)]


def _runtime_setting_keys(store: StateStore) -> list[str]:
    with store._connect() as conn:  # noqa: SLF001
        rows = conn.execute("SELECT key FROM runtime_settings ORDER BY key").fetchall()
    return [str(row["key"]) for row in rows]


def test_polling_review_command_decides_review(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_review",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_owner"}, "sender_name": "owner"},
                "body": {"content": '{"text":"approve rvw-test"}'},
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
    )
    store = StateStore(config.storage.sqlite_path)
    store.create_review("rvw-test", "reuse_row_selected", "subject", {"review_id": "rvw-test"})
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)
    review = store.get_review("rvw-test")

    assert stats.submitted == 1
    assert review is not None
    assert review["status"] == "approved"
    assert review["decision"]["actor"] == "ou_owner"


def test_polling_reply_retry_infers_review_from_status_card(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_reply",
                "parent_id": "om_status",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_owner"}, "sender_name": "owner"},
                "body": {"content": '{"text":"重试"}'},
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
    )
    store = StateStore(config.storage.sqlite_path)
    store.create_review(
        "rvw-test",
        "reuse_row_selected",
        "subject",
        {
            "review_id": "rvw-test",
            "context": {"status_message_id": "om_status"},
        },
        status="deploy_failed",
    )
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)
    review = store.get_review("rvw-test")

    assert stats.submitted == 1
    assert review is not None
    assert review["status"] == "retry_requested"
    assert review["decision"]["decision"] == "RETRY"


def test_polling_ignores_bot_interactive_card_message(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_bot_card",
                "msg_type": "interactive",
                "create_time": str(now_ms),
                "sender": {"sender_type": "app", "sender_id": {"app_id": "cli_test"}},
                "body": {"content": '{"config":{},"header":{},"elements":[{"tag":"div","text":"/v1/models"}]}'},
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)

    assert stats.ignored == 1
    assert store.list_requests() == []
    assert fake.chat_texts == []


def test_polling_codex_control_command(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": "om_codex",
                "create_time": str(now_ms),
                "sender": {"sender_id": {"open_id": "ou_owner"}, "sender_name": "owner"},
                "body": {"content": '{"text":"codex off"}'},
            }
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(chat_ids=["oc_1"], notify_chat_on_accept=True),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)

    assert stats.submitted == 1
    assert store.get_setting("codex_review_enabled") == "false"
    assert fake.chat_texts[-1][1] == "Codex 审核开关"


def test_parse_failure_handoff_after_threshold(tmp_path) -> None:
    now_ms = int(time.time()) * 1000
    fake = FakeFeishuClient(
        [
            {
                "message_id": f"om_bad_{idx}",
                "create_time": str(now_ms + idx),
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_name": "tester"},
                "body": {"content": '{"text":"deploy_vllm\\nmodel_path"}'},
            }
            for idx in range(4)
        ]
    )
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", log_dir=str(tmp_path / "logs")),
        polling=PollingConfig(
            chat_ids=["oc_1"],
            notify_chat_on_accept=True,
            max_parse_failures_before_handoff=3,
        ),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(config, store, make_runner(config.runner), fake)
    worker = FeishuPollingWorker(config, store, fake, orchestrator)

    stats = worker.poll_once(lookback_sec=60)

    assert stats.failed == 4
    assert [title for _, title in fake.chat_texts].count("部署请求解析失败") == 2
    assert [title for _, title in fake.chat_texts].count("连续解析失败，已停止重复提醒") == 1
