from __future__ import annotations

from fmh.config import ReusableWorkersConfig
from fmh.config import ApprovalConfig
from fmh.operator_review import ReviewStage, make_error_review, make_reuse_plan_review, parse_review_command, review_card
from fmh.reusable_workers import build_reusable_deployment_plan, parse_deployed_models_table
from tests.test_reusable_workers import MARKDOWN


def test_make_reuse_plan_review_packet() -> None:
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(MARKDOWN)
    plan = build_reusable_deployment_plan(
        MARKDOWN,
        "/mnt/shared-models/new/model",
        config,
    )

    packet = make_reuse_plan_review(
        weight_path="/mnt/shared-models/new/model",
        plan=plan,
        rows=rows,
    )

    assert packet.stage == ReviewStage.REUSE_ROW_SELECTED
    assert packet.review_id.startswith("rvw-")
    assert "decision=APPROVE" in packet.codex_prompt
    assert "已部署模型文档是本阶段的准入依据" in packet.codex_prompt
    assert "不要因为没有实时 SSH/tmux/health 检查而降级" in packet.codex_prompt
    assert packet.plan["path"]["model_id"] == "model"
    card = review_card(packet)
    assert card["header"]["template"] == "orange"
    assert card["elements"][-1]["tag"] == "action"


def test_make_error_review_packet() -> None:
    packet = make_error_review(stage="health_check", subject_id="req-1", error="timeout")

    assert packet.stage == ReviewStage.ERROR
    assert packet.severity == "danger"
    assert "timeout" in packet.summary


def test_review_card_uses_configured_fallback_mention() -> None:
    packet = make_error_review(stage="health_check", subject_id="req-1", error="timeout")
    card = review_card(
        packet,
        ApprovalConfig(fallback_mention_open_id="ou_123", fallback_mention_name="Owner"),
    )

    assert any(
        "<at id=ou_123>Owner</at>" in element.get("text", {}).get("content", "")
        for element in card["elements"]
    )


def test_review_card_hides_actions_when_card_callbacks_disabled() -> None:
    packet = make_error_review(stage="health_check", subject_id="req-1", error="timeout")
    card = review_card(packet, ApprovalConfig(allow_card_actions=False))
    rendered = str(card)

    assert "'decision': 'RETRY'" not in rendered
    assert "'decision': 'BLOCK'" not in rendered
    assert "无公网模式" in rendered
    assert f"retry {packet.review_id}" in rendered


def test_parse_review_command() -> None:
    assert parse_review_command("approve rvw-abc123") == {
        "review_id": "rvw-abc123",
        "decision": "APPROVE",
        "note": "",
    }
    assert parse_review_command("fmh 阻止 rvw-abc123 原因") == {
        "review_id": "rvw-abc123",
        "decision": "BLOCK",
        "note": "原因",
    }
    assert parse_review_command("hello") is None
