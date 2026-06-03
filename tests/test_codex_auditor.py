from __future__ import annotations

import threading
import time

import pytest

from fmh.codex_auditor import CodexReviewAuditor, deterministic_review_decision, parse_codex_decision
from fmh.config import AppConfig, ReusableWorkersConfig, StorageConfig
from fmh.feishu import NullFeishuClient
from fmh.store import StateStore


def test_parse_codex_decision_from_json() -> None:
    parsed = parse_codex_decision(
        'analysis...\n{"decision":"APPROVE","summary":"ok","risks":[],"next_actions":[]}'
    )

    assert parsed is not None
    assert parsed["decision"] == "APPROVE"
    assert parsed["summary"] == "ok"


def test_parse_codex_decision_from_text() -> None:
    parsed = parse_codex_decision("decision=REQUEST_INFO\n需要人工确认")

    assert parsed is not None
    assert parsed["decision"] == "REQUEST_INFO"


def test_deterministic_review_approves_reusable_doc_row() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "context": {},
        "plan": {
            "row": {
                "tested_tasks": "tau2\nvita",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {
                "worker_path": "/mnt/worker-models/team/model",
            },
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    decision = deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload)

    assert decision is not None
    assert decision["decision"] == "APPROVE"


def test_deterministic_review_approves_idle_empty_row() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "model": "",
                "model_id": "",
                "tested_tasks": "",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/model"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    decision = deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload)

    assert decision is not None
    assert decision["decision"] == "APPROVE"


def test_deterministic_review_keeps_old_tables_compatible() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "reuse": "",
                "reuse_column_present": False,
                "model": "team/model",
                "model_id": "model",
                "tested_tasks": "tau2\nvita",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/next"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    decision = deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload)

    assert decision is not None
    assert decision["decision"] == "APPROVE"


def test_deterministic_review_does_not_approve_reuse_no_row() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "reuse": "no",
                "reuse_column_present": True,
                "model": "team/model",
                "model_id": "model",
                "tested_tasks": "tau2\nvita",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/next"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    assert deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload) is None


def test_deterministic_review_does_not_approve_fresh_untested_row() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "model": "team/model",
                "model_id": "model",
                "tested_tasks": "",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/next"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    assert deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload) is None


def test_deterministic_review_does_not_approve_running_row() -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "tested_tasks": "tau2\nvita(running)",
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/model"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    assert deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload) is None


@pytest.mark.parametrize(
    "tested_tasks",
    [
        "tau2\nvita（running）",
        "tau2\nvita ( running )",
        "tau20\nvitamin",
    ],
)
def test_deterministic_review_uses_robust_reusable_row_classification(tested_tasks: str) -> None:
    payload = {
        "stage": "reuse_row_selected",
        "plan": {
            "row": {
                "model": "team/model",
                "model_id": "model",
                "tested_tasks": tested_tasks,
                "ip": "192.0.2.156",
                "gpu_count": 4,
            },
            "path": {"worker_path": "/mnt/worker-models/team/model"},
            "tmux_session_guess": "ssh_4_gpu_2_156",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }

    assert deterministic_review_decision(AppConfig(reusable_workers=ReusableWorkersConfig()), payload) is None


def test_auditor_schedules_reviews_up_to_parallel_limit(tmp_path, monkeypatch) -> None:
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        reusable_workers=ReusableWorkersConfig(max_parallel_deployments=2),
    )
    store = StateStore(config.storage.sqlite_path)
    for index in range(3):
        store.create_review(
            f"rvw-{index}",
            "reuse_row_selected",
            f"192.0.2.{index}:model-{index}",
            _review_payload(f"192.0.2.{index}", f"model-{index}"),
        )
    auditor = CodexReviewAuditor(config, store, NullFeishuClient())
    started: list[str] = []
    block = threading.Event()

    def fake_process(review: dict[str, object]) -> None:
        started.append(str(review["review_id"]))
        block.wait(2)

    monkeypatch.setattr(auditor, "_process_review", fake_process)
    try:
        count = auditor.process_once()
        _wait_until(lambda: len(started) == 2)

        assert count == 2
        assert len(store.list_reviews(limit=10, status="codex_reviewing")) == 2
        assert len(store.list_reviews(limit=10, status="pending")) == 1
    finally:
        block.set()
        auditor.shutdown()


def test_auditor_does_not_run_same_worker_in_parallel(tmp_path, monkeypatch) -> None:
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        reusable_workers=ReusableWorkersConfig(max_parallel_deployments=2),
    )
    store = StateStore(config.storage.sqlite_path)
    for index in range(2):
        store.create_review(
            f"rvw-{index}",
            "reuse_row_selected",
            f"192.0.2.10:model-{index}",
            _review_payload("192.0.2.10", f"model-{index}"),
        )
    auditor = CodexReviewAuditor(config, store, NullFeishuClient())
    started: list[str] = []
    block = threading.Event()

    def fake_process(review: dict[str, object]) -> None:
        started.append(str(review["review_id"]))
        block.wait(2)

    monkeypatch.setattr(auditor, "_process_review", fake_process)
    try:
        count = auditor.process_once()
        _wait_until(lambda: len(started) == 1)

        assert count == 1
        assert len(store.list_reviews(limit=10, status="codex_reviewing")) == 1
        assert len(store.list_reviews(limit=10, status="pending")) == 1
    finally:
        block.set()
        auditor.shutdown()


def test_auditor_requeues_orphaned_inflight_reviews_after_restart(tmp_path, monkeypatch) -> None:
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        reusable_workers=ReusableWorkersConfig(max_parallel_deployments=2),
    )
    store = StateStore(config.storage.sqlite_path)
    store.create_review(
        "rvw-orphan",
        "reuse_row_selected",
        "192.0.2.20:model-orphan",
        _review_payload("192.0.2.20", "model-orphan"),
        status="deploying",
    )
    auditor = CodexReviewAuditor(config, store, NullFeishuClient())
    started: list[str] = []
    block = threading.Event()

    def fake_process(review: dict[str, object]) -> None:
        started.append(str(review["review_id"]))
        block.wait(2)

    monkeypatch.setattr(auditor, "_process_review", fake_process)
    try:
        count = auditor.process_once()
        _wait_until(lambda: started == ["rvw-orphan"])

        assert count == 1
        review = store.get_review("rvw-orphan")
        assert review is not None
        assert review["status"] == "codex_reviewing"
        decision = review["decision"]
        assert isinstance(decision, dict)
        assert decision["status"] == "claimed"
    finally:
        block.set()
        auditor.shutdown()


def _review_payload(ip: str, model_id: str) -> dict[str, object]:
    return {
        "review_id": f"rvw-{ip}-{model_id}",
        "stage": "reuse_row_selected",
        "title": "复用常驻 worker 部署审核",
        "subject_id": f"{ip}:{model_id}",
        "severity": "warning",
        "summary": "test",
        "plan": {
            "row": {"ip": ip, "gpu_count": 4, "tested_tasks": "tau2\nvita"},
            "path": {"model_id": model_id, "worker_path": f"/mnt/worker-models/team/{model_id}"},
            "tmux_session_guess": "ssh_4_gpu_2_10",
            "vllm_command": "python3 -m vllm.entrypoints.openai.api_server --data-parallel-size 4",
        },
    }


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")
