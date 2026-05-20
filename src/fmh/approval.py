from __future__ import annotations

from typing import Any

from fmh.operator_review import normalize_decision, review_status_for_decision
from fmh.store import StateStore
from fmh.time_utils import utc_now_iso


def decide_review(
    store: StateStore,
    *,
    review_id: str,
    decision: str,
    actor: str = "",
    source: str = "manual",
    note: str = "",
) -> dict[str, Any]:
    normalized = normalize_decision(decision)
    review = store.get_review(review_id)
    if review is None:
        raise KeyError(f"review not found: {review_id}")
    payload = {
        "decision": normalized,
        "source": source,
        "actor": actor,
        "note": note,
        "decided_at": utc_now_iso(),
    }
    store.decide_review(review_id, review_status_for_decision(normalized), payload)
    updated = store.get_review(review_id)
    if updated is None:
        raise KeyError(f"review not found after decision: {review_id}")
    return updated


def extract_card_action(body: dict[str, Any]) -> dict[str, Any] | None:
    event = body.get("event") if isinstance(body.get("event"), dict) else {}
    action = body.get("action") if isinstance(body.get("action"), dict) else {}
    if not action and isinstance(event.get("action"), dict):
        action = event["action"]
    value = action.get("value") if isinstance(action.get("value"), dict) else {}
    if not value and isinstance(action.get("option"), str):
        value = {"fmh_action": action.get("option")}
    if value.get("fmh_action") != "review_decision":
        return None
    return {
        "review_id": str(value.get("review_id") or ""),
        "decision": str(value.get("decision") or ""),
        "actor": _extract_actor(body),
        "note": str(value.get("note") or ""),
    }


def _extract_actor(body: dict[str, Any]) -> str:
    event = body.get("event") if isinstance(body.get("event"), dict) else {}
    for container in (body, event):
        operator = container.get("operator") if isinstance(container.get("operator"), dict) else {}
        if operator:
            for key in ("open_id", "user_id", "union_id", "tenant_key"):
                if operator.get(key):
                    return str(operator[key])
        operator_id = container.get("operator_id")
        if isinstance(operator_id, dict):
            for key in ("open_id", "user_id", "union_id"):
                if operator_id.get(key):
                    return str(operator_id[key])
        if operator_id:
            return str(operator_id)
    return ""
