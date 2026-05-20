from __future__ import annotations

import os
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from fmh.approval import decide_review, extract_card_action
from fmh.config import AppConfig, load_config
from fmh.feishu import (
    FeishuEventError,
    FeishuEventNormalizer,
    is_url_verification,
    make_feishu_client,
    validate_verification_token,
)
from fmh.models import EventSource, Requester, SourceType
from fmh.orchestrator import DeploymentOrchestrator
from fmh.operator_review import review_result_card
from fmh.parser import ParseError, parse_deployment_request
from fmh.reusable_executor import ReusableDeploymentExecutor
from fmh.runner import make_runner
from fmh.store import StateStore, serialize_requests


class ManualDeployPayload(BaseModel):
    text: str
    source_ref: str = "manual"
    user_id: str = "manual"


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_config(config_path)
    store = StateStore(config.storage.sqlite_path)
    runner = make_runner(config.runner)
    feishu_client = make_feishu_client(config.feishu)
    normalizer = FeishuEventNormalizer()
    orchestrator = DeploymentOrchestrator(config, store, runner, feishu_client)

    app = FastAPI(title="free-my-hands", version="0.1.0")
    app.state.config = config
    app.state.store = store
    app.state.orchestrator = orchestrator
    app.state.normalizer = normalizer
    app.state.feishu_client = feishu_client

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "runner": config.runner.mode}

    @app.post("/webhooks/feishu")
    def feishu_webhook(body: dict[str, Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
        if is_url_verification(body):
            validate_verification_token(body, config.feishu.verification_token)
            return {"challenge": body["challenge"]}

        try:
            validate_verification_token(body, config.feishu.verification_token)
            action = extract_card_action(body)
            if action is not None:
                if not config.approval.allow_card_actions:
                    return {"toast": {"type": "warning", "content": "card actions disabled"}}
                review = decide_review(
                    store,
                    review_id=action["review_id"],
                    decision=action["decision"],
                    actor=action.get("actor", ""),
                    source="feishu_card",
                    note=action.get("note", ""),
                )
                decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
                if str(decision.get("decision") or "").upper() == "APPROVE":
                    executor = ReusableDeploymentExecutor(config, store, feishu_client)
                    background_tasks.add_task(executor.execute_if_enabled, review, decision)
                _try_send_review_result(feishu_client, config, review)
                return {"toast": {"type": "success", "content": "review decision recorded"}}
            source = normalizer.normalize(body)
            request = parse_deployment_request(source)
        except KeyError as exc:
            return {"toast": {"type": "warning", "content": str(exc)}}
        except (FeishuEventError, ParseError) as exc:
            _try_notify_parse_error(feishu_client, body, str(exc))
            return {"accepted": False, "error": str(exc)}

        background_tasks.add_task(orchestrator.submit, request)
        return {"accepted": True, "request_id": request.request_id}

    @app.post("/deployments")
    def create_deployment(payload: ManualDeployPayload, background_tasks: BackgroundTasks) -> dict[str, Any]:
        source = EventSource(
            source_type=SourceType.MANUAL,
            source_ref=payload.source_ref,
            requester=Requester(user_id=payload.user_id),
            text=payload.text,
        )
        try:
            request = parse_deployment_request(source)
        except ParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(orchestrator.submit, request)
        return {"accepted": True, "request_id": request.request_id}

    @app.get("/deployments")
    def list_deployments(limit: int = 50) -> dict[str, Any]:
        return {"deployments": serialize_requests(store.list_requests(limit=limit))}

    @app.get("/deployments/{request_id}")
    def get_deployment(request_id: str) -> dict[str, Any]:
        request = store.get_request(request_id)
        if request is None:
            raise HTTPException(status_code=404, detail="request not found")
        return {
            "deployment": request.to_dict(),
            "events": [event.to_dict() for event in store.events_for(request_id)],
        }

    return app


def _try_notify_parse_error(feishu_client: Any, body: dict[str, Any], error: str) -> None:
    try:
        source = FeishuEventNormalizer().normalize(body)
        feishu_client.send_private_text(source.requester.user_id, f"部署请求解析失败: {error}")
    except Exception:
        return


def _try_send_review_result(feishu_client: Any, config: AppConfig, review: dict[str, Any]) -> None:
    payload = review.get("payload") if isinstance(review.get("payload"), dict) else {}
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    if context.get("status_message_id"):
        return
    chat_id = config.approval.fallback_chat_id or config.feishu.default_chat_id
    if not chat_id:
        return
    try:
        decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
        feishu_client.send_chat_card(chat_id, review_result_card(review, decision))
    except Exception:
        return


app = create_app(os.getenv("FMH_CONFIG"))
