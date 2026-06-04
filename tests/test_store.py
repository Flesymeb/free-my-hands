from __future__ import annotations

from fmh.models import DeploymentRequest, Requester, RequestStatus, SourceType
from fmh.store import StateStore


def test_store_roundtrip_and_events(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    request = DeploymentRequest(
        request_id="req-test",
        source_type=SourceType.MANUAL,
        source_ref="unit",
        requester=Requester(user_id="u1"),
        weight_path="/mnt/model",
        model_name="model",
    )

    store.create_request(request)
    store.transition("req-test", RequestStatus.RESOURCE_READY, "ready")

    loaded = store.get_request("req-test")
    assert loaded is not None
    assert loaded.status == RequestStatus.RESOURCE_READY
    assert loaded.weight_path == "/mnt/model"

    events = store.events_for("req-test")
    assert [event.state_to for event in events] == ["pending", "resource_ready"]


def test_store_operator_review(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    payload = {"review_id": "rvw-test", "codex_prompt": "review this"}

    store.create_review("rvw-test", "reuse_row_selected", "subject", payload)
    review = store.get_review("rvw-test")
    assert review is not None
    assert review["status"] == "pending"
    assert review["payload"] == payload

    store.decide_review("rvw-test", "approved", {"decision": "APPROVE"})
    reviews = store.list_reviews()
    assert reviews[0]["status"] == "approved"
    assert reviews[0]["decision"] == {"decision": "APPROVE"}


def test_store_runtime_settings_and_issue_counts(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")

    store.set_setting("codex_review_enabled", "false")
    assert store.get_setting("codex_review_enabled") == "false"

    first = store.increment_issue_count("parse:x", "bad")
    second = store.increment_issue_count("parse:x", "bad again")
    assert first["count"] == 1
    assert second["count"] == 2
    assert not second["alerted"]
    store.mark_issue_alerted("parse:x")
    third = store.increment_issue_count("parse:x", "bad")
    assert third["count"] == 3
    assert third["alerted"]


def test_store_task_status(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")

    store.set_task_status("todo:task_1", {"source_message_id": "om_1", "stages": {"detect": {"status": "完成"}}})

    assert store.get_task_status("todo:task_1")["source_message_id"] == "om_1"
    assert store.get_task_status("missing") == {}


def test_delete_legacy_aggregate_task_statuses_keeps_item_statuses(tmp_path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    store.set_task_status("todo:task_1", {"source_message_id": "om_aggregate"})
    store.set_task_status("todo:task_1:item:abc", {"source_message_id": "om_item"})
    store.set_setting("task_status:chat:oc_1", "{}")

    removed = store.delete_legacy_aggregate_task_statuses()

    assert removed == 1
    assert store.get_task_status("todo:task_1") == {}
    assert store.get_task_status("todo:task_1:item:abc")["source_message_id"] == "om_item"
    assert store.get_setting("task_status:chat:oc_1") == "{}"
