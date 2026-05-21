from __future__ import annotations

from typing import Any

from fmh.config import (
    AppConfig,
    ApprovalConfig,
    FeishuConfig,
    PostDeployNotifyConfig,
    ReusableWorkersConfig,
    StorageConfig,
    WeightConversionConfig,
)
from fmh.reusable_executor import (
    RemoteResult,
    ReusableDeploymentExecutor,
    _gpu_app_pids,
    _row_can_auto_reuse,
    _worker_command,
)
from fmh.store import StateStore


class FakeFeishuClient:
    def __init__(self) -> None:
        self.updated_cards: list[tuple[str, dict[str, Any]]] = []
        self.chat_cards: list[tuple[str, dict[str, Any]]] = []
        self.reply_cards: list[tuple[str, dict[str, Any]]] = []
        self.chat_texts: list[tuple[str, str]] = []
        self.reply_texts: list[tuple[str, str]] = []
        self.completed_tasks: list[str] = []
        self.complete_task_error: Exception | None = None

    def update_card(self, message_id: str, card: dict[str, Any]) -> None:
        self.updated_cards.append((message_id, card))

    def send_chat_card(self, chat_id: str, card: dict[str, Any]) -> str:
        self.chat_cards.append((chat_id, card))
        return "om_new"

    def reply_card(self, message_id: str, card: dict[str, Any]) -> str:
        self.reply_cards.append((message_id, card))
        return "om_reply"

    def send_chat_text(self, chat_id: str, text: str) -> str:
        self.chat_texts.append((chat_id, text))
        return "om_text"

    def reply_text(self, message_id: str, text: str) -> str:
        self.reply_texts.append((message_id, text))
        return "om_reply_text"

    def complete_task(self, task_guid: str) -> None:
        if self.complete_task_error:
            raise self.complete_task_error
        self.completed_tasks.append(task_guid)


def test_executor_updates_existing_source_card_without_extra_success_card(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        reusable_workers=ReusableWorkersConfig(auto_deploy_approved=True),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deployed",
            "summary": "done",
            "endpoint": "http://192.0.2.2:8000",
            "worker": "192.0.2.2",
            "model_id": "model-a",
        },
    )

    assert fake.updated_cards == [("om_status", fake.updated_cards[0][1])]
    assert fake.chat_cards == []
    assert fake.reply_cards == []
    assert fake.chat_texts == []
    assert fake.reply_texts == []


def test_executor_updates_source_card_and_sends_alert_for_human_case(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        approval=ApprovalConfig(fallback_mention_open_id="ou_owner", fallback_mention_name="Owner"),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "needs_human",
            "summary": "worker has existing model",
        },
    )

    assert len(fake.updated_cards) == 1
    assert fake.chat_cards == []
    assert fake.reply_cards == []
    assert fake.reply_texts
    assert "ou_owner" in fake.reply_texts[0][1]


def test_executor_full_card_stays_in_source_chat_when_default_differs(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_default"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        approval=ApprovalConfig(fallback_chat_id="oc_detail"),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()
    review["payload"]["context"].pop("status_message_id")  # type: ignore[index]
    review["payload"]["context"].pop("reply_to_message_id")  # type: ignore[index]

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "needs_human",
            "summary": "worker has existing model",
        },
    )

    assert fake.chat_cards
    assert fake.chat_cards[0][0] == "oc_source"
    assert not any(chat_id in {"oc_default", "oc_detail"} for chat_id, _ in fake.chat_cards)


def test_executor_surfaces_document_write_failure_without_losing_deployed_state(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        reusable_workers=ReusableWorkersConfig(auto_deploy_approved=True),
    )
    store = StateStore(config.storage.sqlite_path)
    review = _review_payload()
    store.create_review(
        str(review["review_id"]),
        str(review["payload"]["stage"]),  # type: ignore[index]
        str(review["subject_id"]),
        review["payload"],  # type: ignore[arg-type]
        status="approved",
    )

    class DocumentFailingExecutor(ReusableDeploymentExecutor):
        def _execute(self, review_id: str, plan: dict[str, Any]) -> dict[str, str]:
            return {"worker": "192.0.2.2", "model_id": "model-a", "endpoint": "http://192.0.2.2:8000"}

        def _write_table_values(self, plan: dict[str, Any], key: str) -> None:
            raise RuntimeError("doc api unavailable")

    executor = DocumentFailingExecutor(config, store, fake)  # type: ignore[arg-type]

    assert executor.execute_if_enabled(review, {"decision": "APPROVE", "summary": "ok"})

    updated = store.get_review("rvw-test")
    decision = updated["decision"]  # type: ignore[index]
    assert updated["status"] == "deployed"
    assert decision["deploy_status"] == "deployed"
    assert decision["document_status"] == "failed"
    rendered = str(fake.updated_cards[-1][1])
    assert "文档回填" in rendered
    assert "需人工" in rendered


def test_executor_sends_manual_subtask_completion_notice_after_success(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_default"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        approval=ApprovalConfig(fallback_mention_open_id="ou_owner", fallback_mention_name="Owner"),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    decision = executor._notify_manual_subtask_completion_if_present(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deployed",
            "summary": "deploy ok",
            "worker": "192.0.2.2",
            "model_id": "model-a",
            "endpoint": "http://192.0.2.2:8000",
        },
    )

    assert fake.completed_tasks == []
    assert fake.chat_texts
    assert fake.chat_texts[0][0] == "oc_source"
    assert "ou_owner" in fake.chat_texts[0][1]
    assert "请手动完成飞书子任务" in fake.chat_texts[0][1]
    assert "model-a" in fake.chat_texts[0][1]
    assert "来源任务: deploy task" in fake.chat_texts[0][1]
    assert "worker:" not in fake.chat_texts[0][1]
    assert "endpoint:" not in fake.chat_texts[0][1]
    assert decision["manual_subtask_completion_notice_status"] == "sent"


def test_executor_marks_task_entry_status(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")))
    store = StateStore(config.storage.sqlite_path)
    store.mark_processed_item(
        "todo:task_1",
        "task_1:sub_1:/mnt/models/model-a",
        "review_pending",
        request_id="rvw-test",
    )
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]

    executor._mark_task_entry_status(_review_payload(), "deployed", summary="done")  # noqa: SLF001

    processed = store.get_processed_item("todo:task_1", "task_1:sub_1:/mnt/models/model-a")
    assert processed is not None
    assert processed["status"] == "deployed"
    assert processed["request_id"] == "rvw-test"
    assert processed["summary"] == "done"


def test_executor_does_not_show_subtask_completion_stage(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deployed",
            "summary": "真实部署完成：model-a 已在 http://192.0.2.2:8000 可见。",
            "approval_summary": "审核通过：复用条件检查通过。",
            "execution_summary": "真实部署完成：model-a 已在 http://192.0.2.2:8000 可见。",
            "endpoint": "http://192.0.2.2:8000",
            "worker": "192.0.2.2",
            "model_id": "model-a",
            "subtask_complete_status": "failed",
            "subtask_complete_error": "no permission",
        },
    )

    state = store.get_task_status("todo:task_1")
    assert "todo" not in state["stages"]


def test_executor_keeps_review_execution_and_todo_details_separate(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deploying",
            "summary": "审核通过：复用条件检查通过。",
            "approval_summary": "审核通过：复用条件检查通过。",
        },
    )
    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deployed",
            "summary": "真实部署完成：model-a 已在 http://192.0.2.2:8000 可见。",
            "approval_summary": "审核通过：复用条件检查通过。",
            "execution_summary": "真实部署完成：model-a 已在 http://192.0.2.2:8000 可见。",
            "endpoint": "http://192.0.2.2:8000",
            "worker": "192.0.2.2",
            "model_id": "model-a",
        },
    )

    state = store.get_task_status("todo:task_1")
    stages = state["stages"]
    assert stages["codex"]["detail"] == "审核通过：复用条件检查通过。"
    assert "真实部署完成" not in stages["codex"]["detail"]
    assert stages["execute"]["detail"] == "真实部署完成：model-a 已在 http://192.0.2.2:8000 可见。"
    assert "todo" not in stages


def test_executor_shows_weight_conversion_as_separate_card_stage(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_source"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()
    review["payload"]["plan"]["weight_conversion"] = {  # type: ignore[index]
        "input_path": "/mnt/gpfs/team/iter_1",
        "output_path": "/mnt/gpfs/team/manual_hf_name",
        "original_weight_path": "/mnt/shared/team/iter_1",
        "output_override": "manual_hf_name",
        "detected_format": "distcp",
        "required": True,
    }

    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deploying",
            "summary": "审核通过：复用条件检查通过。",
            "approval_summary": "审核通过：复用条件检查通过。",
        },
    )
    executor._send_card(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "conversion_done",
            "summary": "权重转换完成：/mnt/gpfs/team/manual_hf_name",
            "execution_summary": "转换完成，正在进入 tmux 启动 vLLM。",
        },
    )

    state = store.get_task_status("todo:task_1")
    stages = state["stages"]
    rendered = str(fake.updated_cards[-1][1])
    assert stages["convert"]["status"] == "完成"
    assert stages["execute"]["status"] == "进行中"
    assert state["deploy_status"] == "vLLM启动中"
    assert "权重转换" in rendered
    assert "执行情况" in rendered


def test_executor_notifies_post_deploy_bot_after_success(tmp_path) -> None:
    fake = FakeFeishuClient()
    config = AppConfig(
        feishu=FeishuConfig(default_chat_id="oc_default"),
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        post_deploy_notify=PostDeployNotifyConfig(
            enabled=True,
            target_open_id="ou_bot",
            target_name="eval-bot",
            chat_id="oc_notify",
        ),
    )
    store = StateStore(config.storage.sqlite_path)
    executor = ReusableDeploymentExecutor(config, store, fake)  # type: ignore[arg-type]
    review = _review_payload()

    decision = executor._notify_post_deploy_bot(  # noqa: SLF001
        review,
        {
            "decision": "APPROVE",
            "deploy_status": "deployed",
            "worker": "192.0.2.2",
            "model_id": "model-a",
            "endpoint": "http://192.0.2.2:8000",
        },
    )

    assert len(fake.chat_cards) == 1
    assert fake.chat_cards[0][0] == "oc_source"
    rendered = str(fake.chat_cards[0][1])
    assert '<at id="ou_bot"></at>' in rendered
    assert "'tag': 'hr'" in rendered
    assert "192.0.2.2" in rendered
    assert "model-a" in rendered
    assert "http://192.0.2.2:8000" in rendered
    assert rendered.index("worker") < rendered.index("endpoint") < rendered.index("model_id")
    assert decision["post_deploy_notify_status"] == "sent"
    assert decision["post_deploy_notify_message_id"] == "om_new"


def test_execute_writes_deploying_marker_before_stopping_worker(tmp_path) -> None:
    order: list[str] = []
    config = AppConfig(storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")))
    store = StateStore(config.storage.sqlite_path)

    class RecordingExecutor(ReusableDeploymentExecutor):
        def _preflight(self, ip: str, session: str, worker_path: str, endpoint: str) -> None:
            order.append("preflight")

        def _write_table_values(self, plan: dict[str, Any], key: str) -> None:
            order.append(f"write:{key}")

        def _stop_existing_vllm(self, ip: str, session: str, endpoint: str) -> None:
            order.append("stop")

        def _send_vllm_command(self, review_id: str, session: str, vllm_command: str) -> None:
            order.append("send")

        def _wait_until_serving(self, session: str, endpoint: str, model_id: str) -> None:
            order.append("wait")

    executor = RecordingExecutor(config, store, FakeFeishuClient())  # type: ignore[arg-type]

    result = executor._execute(  # noqa: SLF001
        "rvw-test",
        {
            "row": {"ip": "192.0.2.2"},
            "path": {"model_id": "model-a", "worker_path": "/mnt/worker-models/model-a"},
            "tmux_session_guess": "ssh_4_gpu_2_2",
            "vllm_command": "python -m vllm.entrypoints.openai.api_server",
            "deploying_table_values": {"模型": "model-a（部署中）"},
        },
    )

    assert order == ["preflight", "write:deploying_table_values", "stop", "send", "wait"]
    assert result == {"worker": "192.0.2.2", "model_id": "model-a", "endpoint": "http://192.0.2.2:8000"}


def test_execute_runs_weight_conversion_before_worker_preflight(tmp_path) -> None:
    order: list[str] = []
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        weight_conversion=WeightConversionConfig(enabled=True, host="converter", conda_env="env", script_path="/trans.sh"),
    )
    store = StateStore(config.storage.sqlite_path)

    class RecordingExecutor(ReusableDeploymentExecutor):
        def _run_weight_conversion(self, conversion: dict[str, Any]) -> None:
            order.append(f"convert:{conversion['output_path']}")

        def _preflight(self, ip: str, session: str, worker_path: str, endpoint: str) -> bool:
            order.append("preflight")
            return True

        def _write_table_values(self, plan: dict[str, Any], key: str) -> None:
            order.append(f"write:{key}")

        def _stop_existing_vllm(self, ip: str, session: str, endpoint: str) -> None:
            order.append("stop")

        def _send_vllm_command(self, review_id: str, session: str, vllm_command: str) -> None:
            order.append("send")

        def _wait_until_serving(self, session: str, endpoint: str, model_id: str) -> None:
            order.append("wait")

    executor = RecordingExecutor(config, store, FakeFeishuClient())  # type: ignore[arg-type]

    result = executor._execute(  # noqa: SLF001
        "rvw-test",
        {
            "row": {"ip": "192.0.2.2"},
            "path": {"model_id": "hf_iter_1", "worker_path": "/mnt/gpfs/team/hf_iter_1"},
            "tmux_session_guess": "ssh_4_gpu_2_2",
            "vllm_command": "python -m vllm.entrypoints.openai.api_server",
            "deploying_table_values": {"模型": "team/hf_iter_1（部署中）"},
            "weight_conversion": {
                "input_path": "/mnt/gpfs/team/iter_1",
                "output_path": "/mnt/gpfs/team/hf_iter_1",
                "original_weight_path": "/mnt/shared/team/iter_1",
                "required": True,
            },
        },
    )

    assert order == [
        "convert:/mnt/gpfs/team/hf_iter_1",
        "preflight",
        "write:deploying_table_values",
        "stop",
        "send",
        "wait",
    ]
    assert result == {"worker": "192.0.2.2", "model_id": "hf_iter_1", "endpoint": "http://192.0.2.2:8000"}


def test_execute_skips_restart_when_target_model_is_already_serving(tmp_path) -> None:
    order: list[str] = []
    config = AppConfig(storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")))
    store = StateStore(config.storage.sqlite_path)

    class RecordingExecutor(ReusableDeploymentExecutor):
        def _endpoint_model_ids(self, endpoint: str) -> list[str]:
            order.append(f"endpoint:{endpoint}")
            return ["model-a"]

        def _preflight(self, ip: str, session: str, worker_path: str, endpoint: str) -> bool:
            raise AssertionError("already-serving deployments should not preflight or restart")

    executor = RecordingExecutor(config, store, FakeFeishuClient())  # type: ignore[arg-type]

    result = executor._execute(  # noqa: SLF001
        "rvw-test",
        {
            "row": {"ip": "192.0.2.2", "model": "team/model-a", "model_id": "model-a"},
            "path": {
                "model_id": "model-a",
                "table_path": "team/model-a",
                "worker_path": "/mnt/worker-models/team/model-a",
            },
            "tmux_session_guess": "ssh_4_gpu_2_2",
            "vllm_command": "python -m vllm.entrypoints.openai.api_server",
        },
    )

    assert order == ["endpoint:http://192.0.2.2:8000"]
    assert result == {"worker": "192.0.2.2", "model_id": "model-a", "endpoint": "http://192.0.2.2:8000"}


def test_run_worker_falls_back_to_tmux_pane_on_worker_ssh_permission_denied(tmp_path) -> None:
    calls: list[str] = []
    config = AppConfig(storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")))
    store = StateStore(config.storage.sqlite_path)

    class FallbackExecutor(ReusableDeploymentExecutor):
        def _run_dev(self, command: str, *, timeout: int, check: bool) -> RemoteResult:
            calls.append(f"dev:{command}")
            return RemoteResult(command=command, returncode=255, stdout="", stderr="Permission denied (publickey,password).")

        def _run_worker_pane(self, session: str, command: str, *, timeout: int) -> RemoteResult:
            calls.append(f"pane:{session}:{command}")
            return RemoteResult(command=command, returncode=0, stdout="OK\n", stderr="")

    executor = FallbackExecutor(config, store, FakeFeishuClient())  # type: ignore[arg-type]

    result = executor._run_worker(  # noqa: SLF001
        "198.51.100.2",
        "echo OK",
        timeout=5,
        check=True,
        tmux_session="ssh_8_gpu_100_2",
    )

    assert result.stdout == "OK\n"
    assert "UpdateHostKeys=no" in calls[0]
    assert calls[1] == "pane:ssh_8_gpu_100_2:echo OK"


def test_worker_command_disables_update_hostkeys() -> None:
    command = _worker_command("198.51.100.2", "true")

    assert "-o UpdateHostKeys=no" in command


def test_auto_reuse_allows_finished_worker_rows() -> None:
    config = AppConfig()

    cases = [
        ({"model": "", "model_id": "", "tested_tasks": ""}, True),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "tau2, vita"}, True),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "TAU2\nVITA"}, True),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "tau2, vita(running)"}, False),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "tau2, vita（running）"}, False),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "tau2, vita ( running )"}, False),
        ({"model": "old/model", "model_id": "old-model", "tested_tasks": "tau20, vitamin"}, False),
        ({"model": "fresh/model", "model_id": "fresh-model", "tested_tasks": ""}, False),
    ]

    for row, expected in cases:
        assert _row_can_auto_reuse(row, config) is expected


def test_gpu_app_pid_parser_handles_nvidia_smi_rows() -> None:
    output = """649316, [Not Found], 134556
649272, python, 134556
bad row
649316, [Not Found], 134556
"""

    assert _gpu_app_pids(output) == ["649316", "649272"]


def _review_payload() -> dict[str, Any]:
    return {
        "review_id": "rvw-test",
        "subject_id": "192.0.2.2:model-a",
        "status": "deployed",
        "payload": {
            "stage": "reuse_row_selected",
            "context": {
                "source_chat_id": "oc_source",
                "reply_to_message_id": "om_parent",
                "status_message_id": "om_status",
                "task_key": "todo:task_1",
                "item_key": "task_1:sub_1:/mnt/models/model-a",
                "task_title": "deploy task",
                "subtask_guid": "sub_1",
            },
            "plan": {
                "path": {"model_id": "model-a"},
                "row": {"ip": "192.0.2.2"},
            },
        },
    }
