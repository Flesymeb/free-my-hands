from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from fmh.config import ApprovalConfig
from fmh.reusable_workers import DeployedModelRow, ReusableDeploymentPlan
from fmh.time_utils import utc_now_iso

DECISIONS = {"APPROVE", "BLOCK", "RETRY", "REQUEST_INFO"}
DECISION_STATUSES = {
    "APPROVE": "approved",
    "BLOCK": "blocked",
    "RETRY": "retry_requested",
    "REQUEST_INFO": "needs_human",
}


class ReviewStage(StrEnum):
    TASK_PARSED = "task_parsed"
    REUSE_ROW_SELECTED = "reuse_row_selected"
    BEFORE_RECONNECT = "before_reconnect"
    BEFORE_STOP_VLLM = "before_stop_vllm"
    BEFORE_START_VLLM = "before_start_vllm"
    AFTER_START_VLLM = "after_start_vllm"
    BEFORE_DOC_WRITE = "before_doc_write"
    NEW_WORKER_REQUIRED = "new_worker_required"
    ERROR = "error"


@dataclass(frozen=True)
class ReviewPacket:
    review_id: str
    stage: ReviewStage
    title: str
    subject_id: str
    severity: str
    summary: str
    context: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    codex_prompt: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewPacket":
        payload = dict(data)
        payload["stage"] = ReviewStage(payload["stage"])
        return cls(**payload)


def make_reuse_plan_review(
    *,
    weight_path: str,
    plan: ReusableDeploymentPlan,
    rows: list[DeployedModelRow],
    conversion: dict[str, Any] | None = None,
) -> ReviewPacket:
    reusable_rows = [row.to_dict() for row in rows if row.ip and row.gpu_count]
    context = {
        "weight_path": weight_path,
        "selected_row": plan.row.to_dict(),
        "candidate_rows": reusable_rows,
        "path": plan.path.to_dict(),
    }
    if conversion:
        context["weight_conversion"] = conversion
    checks = [
        "本阶段以已部署模型文档为准入依据：空闲行，或 required_finished_tasks 均已完成且没有 running 标记，即可批准复用计划。",
        "确认 selected_row 的已经测试完的任务包含 tau2 和 vita，且没有 running 标记。",
        "如果已经测试完的任务为空且模型/模型id也为空，这是空闲 worker；如果已经测试完的任务为空但模型/模型id不为空，这是刚部署未测试，不可停。",
        "确认待部署模型路径已从 /mnt/shared-models 转为 /mnt/worker-models。",
        "确认 tmux_session_guess 与 worker IP 后两段一致。",
        "确认 vLLM 命令里的 data-parallel-size 等于该 worker 的卡数。",
        "确认写回文档前先标记部署中，测试通过后再去掉部署中标记。",
    ]
    risks = [
        "停错 tmux session 会影响其他正在服务的模型。",
        "如果 tau2/vita 状态过期，可能复用仍在使用的节点。",
        "如果模型路径未转换正确，vLLM 会启动失败。",
    ]
    next_actions = [
        "如审核通过：进入对应 tmux session，停旧 vLLM，运行 vllm_command。",
        "如 SSH 断开：先执行 reconnect-plan 或使用 reconnect_command 恢复 worker 连接。",
        "启动后：在 test-model 窗口检查 /v1/models 与模型 id 是否匹配。",
        "测试通过后：写回 final_table_values，并清空已经测试完的任务列。",
    ]
    if conversion:
        checks.append("确认需要先执行权重转换，部署路径应使用转换后的 output_path。")
        next_actions.insert(0, "先在转换机执行权重转换；转换成功后再停旧 vLLM。")
    plan_payload = plan.to_dict()
    if conversion:
        plan_payload["weight_conversion"] = conversion
    packet = _packet(
        stage=ReviewStage.REUSE_ROW_SELECTED,
        title="复用常驻 worker 部署审核",
        subject_id=f"{plan.row.ip}:{plan.path.model_id}",
        severity="warning",
        summary=f"计划复用 {plan.row.ip} 部署 {plan.path.model_id}",
        context=context,
        plan=plan_payload,
        checks=checks,
        risks=risks,
        next_actions=next_actions,
    )
    return packet


def make_error_review(
    *,
    stage: str,
    subject_id: str,
    error: str,
    context: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> ReviewPacket:
    return _packet(
        stage=ReviewStage.ERROR,
        title="部署流程异常审核",
        subject_id=subject_id,
        severity="danger",
        summary=f"{stage}: {error}",
        context=context or {},
        plan=plan or {},
        checks=[
            "确认异常发生阶段和前置状态。",
            "检查是否已经对文档或 tmux 做过部分写入/操作。",
            "给出继续、回滚、重试或人工接管的明确建议。",
        ],
        risks=[
            "重复重试可能导致重复启动 vLLM 或覆盖文档。",
            "未确认当前 tmux 状态时不应继续 stop/start。",
        ],
        next_actions=[
            "BLOCK：如果状态不明确或有误停风险。",
            "RETRY：如果错误是瞬时网络/API/SSH 问题且状态可恢复。",
            "APPROVE：如果计划仍然安全且下一步清晰。",
        ],
    )


def review_card(
    packet: ReviewPacket,
    approval: ApprovalConfig | None = None,
    *,
    footer: str = "",
    include_actions: bool = True,
) -> dict[str, Any]:
    color = {"info": "blue", "warning": "orange", "danger": "red"}.get(packet.severity, "blue")
    fields = _review_fields(packet)
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "fields": [
                {
                    "is_short": key not in {"summary", "review_id"},
                    "text": {"tag": "lark_md", "content": f"**{key}**\n{value}"},
                }
                for key, value in fields.items()
            ],
        },
    ]
    if footer:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": footer}})
    elif approval and approval.fallback_mention_open_id:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"{_mention_md(approval)} 可在 Codex 异常时点击或回复审批。",
                },
            }
        )
    actions_enabled = include_actions and (approval is None or approval.allow_card_actions)
    if actions_enabled:
        elements.append(_review_actions(packet.review_id))
    elif include_actions:
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": (
                            "无公网模式：在群里回复 "
                            f"approve {packet.review_id}、retry {packet.review_id} 或 block {packet.review_id}。"
                        ),
                    }
                ],
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": packet.title},
            "template": color,
        },
        "elements": elements,
    }


def review_result_card(review: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    status = str(review.get("status") or "")
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    title = f"审核结果：{decision.get('decision', status)}"
    color = {
        "approved": "green",
        "deploying": "blue",
        "deployed": "green",
        "deploy_failed": "red",
        "blocked": "red",
        "retry_requested": "orange",
        "needs_human": "orange",
        "codex_failed": "red",
    }.get(status, "blue")
    fields = {
        "review_id": str(review.get("review_id") or ""),
        "worker": _subject_worker(str(review.get("subject_id") or "")),
        "model_id": _subject_model(str(review.get("subject_id") or "")),
        "source": str(decision.get("source") or ""),
        "summary": _short_text(str(decision.get("summary") or payload.get("summary") or ""), 90),
    }
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title}, "template": color},
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": key not in {"summary", "review_id"},
                        "text": {"tag": "lark_md", "content": f"**{key}**\n{value}"},
                    }
                    for key, value in fields.items()
                    if value
                ],
            }
        ],
    }


def human_fallback_footer(approval: ApprovalConfig, error: str) -> str:
    mention = _mention_md(approval)
    prefix = f"{mention} " if mention else ""
    return (
        f"{prefix}Codex 自动审核失败，需要人工确认。\n"
        f"**原因**\n{_short_text(error, 240)}\n"
        "**无公网兜底**\n"
        "在群里回复 `approve <review_id>`、`block <review_id>` 或 `retry <review_id>`。"
    )


def mention_text(approval: ApprovalConfig) -> str:
    if approval.fallback_mention_open_id:
        name = approval.fallback_mention_name or approval.fallback_mention_open_id
        return f'<at user_id="{approval.fallback_mention_open_id}">{name}</at>'
    if approval.fallback_mention_name:
        return f"@{approval.fallback_mention_name}"
    return ""


def parse_review_command(text: str) -> dict[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    decision_words = {
        "approve": "APPROVE",
        "approved": "APPROVE",
        "ok": "APPROVE",
        "同意": "APPROVE",
        "批准": "APPROVE",
        "通过": "APPROVE",
        "block": "BLOCK",
        "blocked": "BLOCK",
        "reject": "BLOCK",
        "rejected": "BLOCK",
        "驳回": "BLOCK",
        "阻止": "BLOCK",
        "拒绝": "BLOCK",
        "retry": "RETRY",
        "重试": "RETRY",
        "cancel": "BLOCK",
        "取消": "BLOCK",
        "info": "REQUEST_INFO",
        "request_info": "REQUEST_INFO",
        "补充信息": "REQUEST_INFO",
    }
    prefix = r"(?:/fmh\s+|fmh\s+)?"
    word_pattern = "|".join(re.escape(word) for word in sorted(decision_words, key=len, reverse=True))
    id_pattern = r"(?P<review_id>rvw-[A-Za-z0-9_-]+)"
    leading = re.match(rf"^{prefix}(?P<word>{word_pattern})\s+{id_pattern}(?:\s+(?P<note>.*))?$", stripped, re.I)
    trailing = re.match(rf"^{id_pattern}\s+(?P<word>{word_pattern})(?:\s+(?P<note>.*))?$", stripped, re.I)
    match = leading or trailing
    if not match:
        return None
    word = match.group("word").lower()
    return {
        "review_id": match.group("review_id"),
        "decision": decision_words[word],
        "note": (match.groupdict().get("note") or "").strip(),
    }


def normalize_decision(decision: str) -> str:
    normalized = decision.strip().upper()
    if normalized not in DECISIONS:
        raise ValueError(f"unknown review decision: {decision}")
    return normalized


def review_status_for_decision(decision: str) -> str:
    return DECISION_STATUSES[normalize_decision(decision)]


def _review_actions(review_id: str) -> dict[str, Any]:
    return {
        "tag": "action",
        "actions": [
            _review_button(review_id, "APPROVE", "同意下一步", "primary"),
            _review_button(review_id, "RETRY", "重试审核", "default"),
            _review_button(review_id, "BLOCK", "阻止", "danger"),
        ],
    }


def _review_fields(packet: ReviewPacket) -> dict[str, str]:
    row = packet.plan.get("row") if isinstance(packet.plan.get("row"), dict) else {}
    path = packet.plan.get("path") if isinstance(packet.plan.get("path"), dict) else {}
    fields = {
        "review_id": packet.review_id,
        "阶段": packet.stage.value,
        "worker": str(row.get("ip") or _subject_worker(packet.subject_id)),
        "模型id": str(path.get("model_id") or _subject_model(packet.subject_id)),
        "状态": _short_text(packet.summary, 90),
    }
    if row.get("gpu_count"):
        fields["卡数"] = str(row["gpu_count"])
    tested = str(row.get("tested_tasks") or "")
    if tested:
        fields["已跑完"] = _short_text(tested.replace("\n", ", "), 40)
    return fields


def _subject_worker(subject: str) -> str:
    return subject.split(":", 1)[0] if ":" in subject else subject


def _subject_model(subject: str) -> str:
    return subject.split(":", 1)[1] if ":" in subject else ""


def _short_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _review_button(review_id: str, decision: str, text: str, button_type: str) -> dict[str, Any]:
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


def _mention_md(approval: ApprovalConfig) -> str:
    if approval.fallback_mention_open_id:
        name = approval.fallback_mention_name or approval.fallback_mention_open_id
        return f"<at id={approval.fallback_mention_open_id}>{name}</at>"
    if approval.fallback_mention_name:
        return f"@{approval.fallback_mention_name}"
    return ""


def _packet(
    *,
    stage: ReviewStage,
    title: str,
    subject_id: str,
    severity: str,
    summary: str,
    context: dict[str, Any],
    plan: dict[str, Any],
    checks: list[str],
    risks: list[str],
    next_actions: list[str],
) -> ReviewPacket:
    base = f"{stage.value}:{subject_id}:{summary}:{utc_now_iso()}"
    review_id = "rvw-" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    draft = ReviewPacket(
        review_id=review_id,
        stage=stage,
        title=title,
        subject_id=subject_id,
        severity=severity,
        summary=summary,
        context=context,
        plan=plan,
        checks=checks,
        risks=risks,
        next_actions=next_actions,
    )
    return ReviewPacket(**{**draft.to_dict(), "codex_prompt": _codex_prompt(draft), "stage": draft.stage})


def _codex_prompt(packet: ReviewPacket) -> str:
    payload = packet.to_dict()
    payload.pop("codex_prompt", None)
    return (
        "你是 free-my-hands 部署流程的审核调控 Codex。请审查下面的阶段审核包。\n"
        "审批边界：reuse_row_selected 阶段只审核“是否可以选这个常驻 worker 进入下一步”。"
        "已部署模型文档是本阶段的准入依据：如果 selected_row 是空闲行（模型、模型id、已经测试完的任务均为空），"
        "或者 selected_row.tested_tasks 同时精确列出 tau2 和 vita 且没有 (running) 标记；"
        "模型路径已正确从 shared-storage 前缀转换到 worker 前缀；"
        "tmux_session_guess 与 worker IP 后两段一致；vLLM 命令卡数与 row.gpu_count 一致，"
        "则应 decision=APPROVE。\n"
        "如果 selected_row.tested_tasks 为空但模型/模型id不为空，表示刚部署好但还没开始测试，不可停，不要批准复用。\n"
        "不要因为没有实时 SSH/tmux/health 检查而降级为 REQUEST_INFO；这些检查属于后续 "
        "before_stop_vllm / after_start_vllm 阶段。只有在表格状态缺失/矛盾、有 running 标记、"
        "路径转换错误、卡数不匹配、没有可复用行时，才 BLOCK 或 REQUEST_INFO。\n"
        "输出必须包含：decision=APPROVE/BLOCK/RETRY/REQUEST_INFO，主要风险，"
        "需要执行的下一步命令或禁止执行的动作。不要假设未提供的信息。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
