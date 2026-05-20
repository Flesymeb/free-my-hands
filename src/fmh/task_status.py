from __future__ import annotations

from typing import Any

from fmh.time_utils import utc_now_iso

STAGE_ORDER = ("detect", "codex", "execute", "document")
STATE_STAGE_ORDER = ("detect", "codex", "execute", "document", "todo")
STAGE_LABELS = {
    "detect": "任务检测",
    "codex": "Codex意见",
    "execute": "执行情况",
    "document": "文档回填",
}


def task_status_with_stage(
    state: dict[str, Any],
    stage: str,
    status: str,
    detail: str = "",
    *,
    title: str = "",
    source_ref: str = "",
    source_chat_id: str = "",
    source_message_id: str = "",
    model_id: str = "",
    model: str = "",
    worker: str = "",
    address: str = "",
    endpoint: str = "",
    deploy_status: str = "",
    review_id: str = "",
) -> dict[str, Any]:
    next_state = dict(state)
    if title:
        next_state["title"] = title
    if source_ref:
        next_state["source_ref"] = source_ref
    if source_chat_id:
        next_state["source_chat_id"] = source_chat_id
    if source_message_id:
        next_state["source_message_id"] = source_message_id
    if model_id:
        next_state["model_id"] = model_id
    if model:
        next_state["model"] = model
    if worker:
        next_state["worker"] = worker
    if address:
        next_state["address"] = address
    if endpoint:
        next_state["endpoint"] = endpoint
    if review_id:
        next_state["review_id"] = review_id
    next_state["deploy_status"] = deploy_status or _next_deploy_status(stage, status, next_state.get("deploy_status", ""))
    stages = dict(next_state.get("stages") or {})
    _drop_downstream_stages(stages, stage)
    stages[stage] = {
        "label": STAGE_LABELS.get(stage, stage),
        "status": status,
        "detail": detail,
        "updated_at": utc_now_iso(),
    }
    next_state["stages"] = stages
    next_state["updated_at"] = utc_now_iso()
    return next_state


def _drop_downstream_stages(stages: dict[str, Any], stage: str) -> None:
    if stage not in STATE_STAGE_ORDER:
        return
    current_index = STATE_STAGE_ORDER.index(stage)
    for downstream in STATE_STAGE_ORDER[current_index + 1 :]:
        stages.pop(downstream, None)


def task_status_card(state: dict[str, Any]) -> dict[str, Any]:
    source = str(state.get("title") or state.get("source_ref") or "部署任务")
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    info_elements: list[dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": _hero_line(state)}},
    ]
    summary_fields: list[dict[str, Any]] = []
    address = str(state.get("address") or state.get("worker") or "")
    if address:
        summary_fields.append(_field("地址", address))
    if state.get("endpoint") and str(state.get("deploy_status") or "") in {"已部署", "错误", "需人工"}:
        summary_fields.append(_field("endpoint", str(state["endpoint"])))
    if summary_fields:
        info_elements.append({"tag": "div", "fields": summary_fields})
    if state.get("model"):
        info_elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**模型**\n{_md_escape(_path_short(str(state['model'])))}",
                },
            }
        )

    elements = [
        *info_elements,
        *_stage_elements(stages),
    ]
    if source:
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"来源任务：{_short(source, 140)}",
                    }
                ],
            }
        )
    if _needs_manual_actions(state):
        review_id = str(state.get("review_id") or "")
        actions_enabled = bool(state.get("card_actions_enabled", True))
        if actions_enabled:
            elements.append(_manual_actions(str(state.get("review_id") or "")))
            note = f"按钮不可用时，回复本卡片：重试 或 取消；也可发送 retry {review_id}"
        else:
            note = f"回复本卡片：重试 或 取消；也可发送 retry {review_id}"
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": note,
                    }
                ],
            }
        )

    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "模型部署任务"},
            "template": _template(stages, str(state.get("deploy_status") or "")),
        },
        "elements": elements,
    }


def _field(
    label: str,
    value: str,
    *,
    is_short: bool = True,
    raw: bool = False,
    multiline: bool = False,
) -> dict[str, Any]:
    separator = "\n" if multiline else " "
    return {
        "is_short": is_short,
        "text": {"tag": "lark_md", "content": f"**{label}**{separator}{value if raw else _md_escape(value)}"},
    }


def _stage_elements(stages: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[str] = []
    for stage in STAGE_ORDER:
        item = stages.get(stage)
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or STAGE_LABELS.get(stage, stage))
        status = str(item.get("status") or "")
        detail = str(item.get("detail") or "")
        suffix = _stage_detail_suffix(status, detail)
        lines.append(f"**{_md_escape(label)}** {_stage_status_tag(status)}{suffix}")
    if not lines:
        lines.append(f"**任务检测** {_stage_status_tag('等待中')}")
    return [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(lines)},
        }
    ]


def _hero_line(state: dict[str, Any]) -> str:
    status = str(state.get("deploy_status") or "准备中")
    model_id = str(state.get("model_id") or "等待模型")
    return f"{_status_tag(status)}  **{_md_escape(_short(model_id, 80))}**"


def _template(stages: dict[str, Any], deploy_status: str = "") -> str:
    statuses = [
        str(item.get("status") or "")
        for stage, item in stages.items()
        if stage != "todo" and isinstance(item, dict)
    ]
    if any(status in {"失败", "错误", "需人工"} for status in statuses):
        return "red"
    if deploy_status in {"错误", "需人工"}:
        return "red"
    if deploy_status == "已部署":
        return "green"
    if deploy_status == "vLLM启动中":
        return "blue"
    if deploy_status == "准备中":
        return "orange"
    if statuses and all(status in {"完成", "通过"} for status in statuses):
        return "green"
    if any(status in {"进行中", "待审核"} for status in statuses):
        return "blue"
    return "orange"


def _next_deploy_status(stage: str, status: str, current: object) -> str:
    if stage == "todo":
        return str(current) if current else "已部署"
    if status in {"失败", "错误"}:
        return "错误"
    if status == "需人工":
        return "需人工"
    if stage in {"detect", "codex"}:
        return "准备中"
    if stage == "execute" and status == "进行中":
        return "vLLM启动中"
    if (stage == "execute" and status == "完成") or (stage == "document" and status == "完成"):
        return "已部署"
    if current:
        return str(current)
    return "准备中"


def _path_short(value: str) -> str:
    value = value.strip()
    if len(value) <= 96:
        return value
    parts = [part for part in value.split("/") if part]
    if len(parts) >= 3:
        suffix = "/".join(parts[-3:])
        return f".../{suffix}"
    return _short(value, 96)


def _status_tag(status: str) -> str:
    return f"<text_tag color='{_status_color(status)}'>{_md_escape(status)}</text_tag>"


def _stage_status_tag(status: str) -> str:
    return f"<text_tag color='{_stage_color(status)}'>{_md_escape(status)}</text_tag>"


def _stage_detail_suffix(status: str, detail: str) -> str:
    text = " ".join(str(detail or "").split())
    if not text:
        return ""
    limit = 140 if status in {"失败", "错误", "需人工"} else 72
    return f" · {_md_escape(_short(text, limit))}"


def _needs_manual_actions(state: dict[str, Any]) -> bool:
    if not state.get("review_id"):
        return False
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    for stage, item in stages.items():
        if stage == "todo" or not isinstance(item, dict):
            continue
        if str(item.get("status") or "") in {"失败", "错误", "需人工"}:
            return True
    return str(state.get("deploy_status") or "") == "错误"


def _manual_actions(review_id: str) -> dict[str, Any]:
    return {
        "tag": "action",
        "actions": [
            _action_button(review_id, "RETRY", "重试", "primary"),
            _action_button(review_id, "BLOCK", "取消", "danger"),
        ],
    }


def _action_button(review_id: str, decision: str, text: str, button_type: str) -> dict[str, Any]:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": button_type,
        "value": {
            "fmh_action": "review_decision",
            "review_id": review_id,
            "decision": decision,
        },
    }


def _status_color(status: str) -> str:
    return {
        "准备中": "orange",
        "vLLM启动中": "blue",
        "已部署": "green",
        "错误": "red",
        "需人工": "carmine",
    }.get(status, "grey")


def _stage_color(status: str) -> str:
    if status in {"完成", "通过"}:
        return "green"
    if status in {"进行中", "待审核"}:
        return "blue"
    if status in {"失败", "错误"}:
        return "red"
    if status == "需人工":
        return "carmine"
    return "grey"


def _md_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for char in ("*", "_", "~", "`", "[", "]", "(", ")"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def _short(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
