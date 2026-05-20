from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from fmh.config import FeishuConfig
from fmh.models import DeploymentRequest, EventSource, Requester, SourceType


class FeishuEventError(ValueError):
    pass


class FeishuAuthError(RuntimeError):
    pass


class _TemporaryFeishuError(RuntimeError):
    pass


def is_url_verification(body: dict[str, Any]) -> bool:
    return body.get("type") == "url_verification" and "challenge" in body


def validate_verification_token(body: dict[str, Any], expected_token: str) -> None:
    if not expected_token:
        return
    received = body.get("token") or body.get("header", {}).get("token")
    if received != expected_token:
        raise FeishuEventError("invalid Feishu verification token")


class FeishuEventNormalizer:
    def normalize(self, body: dict[str, Any]) -> EventSource:
        event = body.get("event") or body
        message = event.get("message") or {}
        sender = event.get("sender") or {}

        text = _extract_text(event, message)
        if not text:
            raise FeishuEventError("could not extract text from Feishu event")

        source_type = _detect_source_type(event, message)
        source_ref = (
            message.get("message_id")
            or event.get("message_id")
            or event.get("document_id")
            or event.get("obj_token")
            or body.get("event_id")
            or body.get("uuid")
            or "unknown"
        )
        requester = _extract_requester(sender, event)
        return EventSource(
            source_type=source_type,
            source_ref=str(source_ref),
            requester=requester,
            text=text,
            raw_event=body,
        )


@dataclass
class FeishuToken:
    value: str
    expires_at: float


class FeishuOpenAPIClient:
    RETRY_DELAYS = (0.0, 2.0, 6.0, 15.0, 30.0)
    TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, config: FeishuConfig) -> None:
        self.config = config
        self._token: FeishuToken | None = None

    def send_private_text(self, open_id: str, text: str) -> str:
        return self.send_text(open_id, text, receive_id_type="open_id")

    def send_chat_text(self, chat_id: str, text: str) -> str:
        return self.send_text(chat_id, text, receive_id_type="chat_id")

    def send_text(self, receive_id: str, text: str, *, receive_id_type: str) -> str:
        if not receive_id or receive_id == "unknown":
            return ""
        data = self._send_message(
            receive_id,
            receive_id_type=receive_id_type,
            msg_type="text",
            content={"text": text},
        )
        return _message_id(data)

    def send_private_card(self, open_id: str, card: dict[str, Any]) -> str:
        return self.send_card(open_id, card, receive_id_type="open_id")

    def send_chat_card(self, chat_id: str, card: dict[str, Any]) -> str:
        return self.send_card(chat_id, card, receive_id_type="chat_id")

    def reply_text(self, message_id: str, text: str) -> str:
        if not message_id:
            return ""
        data = self._reply_message(message_id, msg_type="text", content={"text": text})
        return _message_id(data)

    def reply_card(self, message_id: str, card: dict[str, Any]) -> str:
        if not message_id:
            return ""
        data = self._reply_message(message_id, msg_type="interactive", content=card)
        return _message_id(data)

    def send_card(self, receive_id: str, card: dict[str, Any], *, receive_id_type: str) -> str:
        if not receive_id or receive_id == "unknown":
            return ""
        data = self._send_message(
            receive_id,
            receive_id_type=receive_id_type,
            msg_type="interactive",
            content=card,
        )
        return _message_id(data)

    def update_card(self, message_id: str, card: dict[str, Any]) -> None:
        if not message_id:
            return
        self._patch(
            f"/im/v1/messages/{message_id}",
            json={"content": json.dumps(card, ensure_ascii=False)},
        )

    def add_message_reaction(self, message_id: str, emoji_type: str) -> str:
        if not message_id or not emoji_type:
            return ""
        data = self._post(
            f"/im/v1/messages/{message_id}/reactions",
            json={"reaction_type": {"emoji_type": emoji_type}},
        )
        reaction_id = data.get("data", {}).get("reaction_id")
        return str(reaction_id or "")

    def list_chats(self, page_size: int = 50) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if page_token:
                params["page_token"] = page_token
            data = self._get("/im/v1/chats", params=params)
            body = data.get("data", {})
            items.extend(body.get("items", []))
            if not body.get("has_more"):
                return items
            page_token = body.get("page_token", "")

    def list_messages(
        self,
        chat_id: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        page_size: int = 50,
        sort_type: str = "ByCreateTimeAsc",
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {
                "container_id_type": "chat",
                "container_id": chat_id,
                "sort_type": sort_type,
                "page_size": page_size,
            }
            if start_time is not None:
                params["start_time"] = start_time
            if end_time is not None:
                params["end_time"] = end_time
            if page_token:
                params["page_token"] = page_token
            data = self._get("/im/v1/messages", params=params)
            body = data.get("data", {})
            items.extend(body.get("items", []))
            if not body.get("has_more"):
                return items
            page_token = body.get("page_token", "")

    def get_document_raw_content(self, document_id: str) -> str:
        data = self._get(f"/docx/v1/documents/{document_id}/raw_content")
        body = data.get("data", {})
        content = body.get("content") or body.get("raw_content") or body.get("text")
        if content is None:
            return ""
        return str(content)

    def get_doc_markdown(self, doc_token: str, doc_type: str = "docx") -> str:
        data = self._get(
            "/docs/v1/content",
            params={
                "doc_token": doc_token,
                "doc_type": doc_type,
                "content_type": "markdown",
            },
        )
        content = data.get("data", {}).get("content")
        return str(content or "")

    def get_wiki_node(self, node_token: str) -> dict[str, Any]:
        data = self._get("/wiki/v2/spaces/get_node", params={"token": node_token})
        node = data.get("data", {}).get("node")
        return node if isinstance(node, dict) else {}

    def get_task(self, task_guid: str) -> dict[str, Any]:
        data = self._get(
            f"/task/v2/tasks/{task_guid}",
            params={"user_id_type": "open_id"},
        )
        task = data.get("data", {}).get("task")
        return task if isinstance(task, dict) else {}

    def list_subtasks(self, task_guid: str, page_size: int = 50) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {
                "user_id_type": "open_id",
                "page_size": page_size,
            }
            if page_token:
                params["page_token"] = page_token
            data = self._get(f"/task/v2/tasks/{task_guid}/subtasks", params=params)
            body = data.get("data", {})
            items.extend(body.get("items", []))
            if not body.get("has_more"):
                return items
            page_token = body.get("page_token", "")

    def complete_task(self, task_guid: str) -> None:
        if not task_guid:
            return
        user_access_token = self.config.user_access_token.strip()
        if not user_access_token:
            raise FeishuAuthError("Feishu user_access_token is required to complete tasks; tenant token is not used")
        self._patch(
            f"/task/v2/tasks/{task_guid}",
            params={"user_id_type": "open_id"},
            json={
                "task": {"completed_at": str(int(time.time() * 1000))},
                "update_fields": ["completed_at"],
            },
            access_token=user_access_token,
        )

    def update_document_status(self, request: DeploymentRequest, text: str) -> None:
        meta = request.metadata.get("feishu", {})
        if not isinstance(meta, dict):
            return None
        document_id = str(meta.get("document_id") or "")
        status_block_id = str(meta.get("status_block_id") or "")
        if not document_id or not status_block_id:
            return None
        payload = {
            "update_text_elements": {
                "elements": [
                    {
                        "text_run": {
                            "content": text,
                        }
                    }
                ]
            }
        }
        self._patch(
            f"/docx/v1/documents/{document_id}/blocks/{status_block_id}",
            json=payload,
        )
        return None

    def _tenant_access_token(self) -> str:
        if self._token and self._token.expires_at > time.time() + 60:
            return self._token.value
        if not self.config.app_id or not self.config.app_secret:
            raise FeishuAuthError("missing Feishu app_id/app_secret")
        data = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            auth=False,
            json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
        )
        if data.get("code") != 0:
            raise FeishuAuthError(f"Feishu auth failed: {data.get('msg') or data}")
        token = data["tenant_access_token"]
        expires_in = int(data.get("expire", 7200))
        self._token = FeishuToken(value=token, expires_at=time.time() + expires_in)
        return token

    def _send_message(
        self,
        receive_id: str,
        *,
        receive_id_type: str,
        msg_type: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        return self._post(
            "/im/v1/messages",
            json=payload,
            params={"receive_id_type": receive_id_type},
        )

    def _reply_message(
        self,
        message_id: str,
        *,
        msg_type: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        return self._post(f"/im/v1/messages/{message_id}/reply", json=payload)

    def _post(
        self,
        path: str,
        *,
        json: dict[str, Any],
        params: dict[str, Any] | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            path,
            auth=True,
            json=json,
            params=params,
            access_token=access_token,
        )

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        return self._request("GET", path, auth=True, params=params, access_token=access_token)

    def _patch(
        self,
        path: str,
        *,
        json: dict[str, Any],
        params: dict[str, Any] | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            path,
            auth=True,
            json=json,
            params=params,
            access_token=access_token,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth: bool,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        elif auth:
            headers["Authorization"] = f"Bearer {self._tenant_access_token()}"

        last_exc: Exception | None = None
        for attempt, delay in enumerate(self.RETRY_DELAYS, start=1):
            if delay:
                time.sleep(delay)
            try:
                response = httpx.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=15,
                )
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                if response.status_code in self.TRANSIENT_STATUS_CODES:
                    detail = _feishu_error_detail(response, data)
                    last_exc = _TemporaryFeishuError(detail)
                    if attempt == len(self.RETRY_DELAYS):
                        break
                    continue
                if response.is_error:
                    detail = _feishu_error_detail(response, data)
                    raise RuntimeError(detail)
                if data.get("code") not in (None, 0):
                    raise RuntimeError(_feishu_api_error_detail(data))
                return data
            except (httpx.HTTPError, _TemporaryFeishuError) as exc:
                last_exc = exc
                if attempt == len(self.RETRY_DELAYS):
                    break
        raise RuntimeError(f"Feishu request failed: {last_exc}") from last_exc


def _feishu_error_detail(response: httpx.Response, data: dict[str, Any]) -> str:
    if data:
        return f"Feishu HTTP {response.status_code}: {_feishu_api_error_detail(data)}"
    body = response.text.strip()
    if len(body) > 500:
        body = body[:499] + "…"
    return f"Feishu HTTP {response.status_code}: {body or response.reason_phrase}"


def _feishu_api_error_detail(data: dict[str, Any]) -> str:
    code = data.get("code")
    msg = data.get("msg") or data.get("message") or data
    return f"code={code}, msg={msg}"


class NullFeishuClient:
    def send_private_text(self, open_id: str, text: str) -> str:
        return ""

    def send_private_card(self, open_id: str, card: dict[str, Any]) -> str:
        return ""

    def send_chat_text(self, chat_id: str, text: str) -> str:
        return ""

    def send_chat_card(self, chat_id: str, card: dict[str, Any]) -> str:
        return ""

    def reply_text(self, message_id: str, text: str) -> str:
        return ""

    def reply_card(self, message_id: str, card: dict[str, Any]) -> str:
        return ""

    def update_card(self, message_id: str, card: dict[str, Any]) -> None:
        return None

    def add_message_reaction(self, message_id: str, emoji_type: str) -> str:
        return ""

    def list_chats(self, page_size: int = 50) -> list[dict[str, Any]]:
        return []

    def list_messages(
        self,
        chat_id: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        page_size: int = 50,
        sort_type: str = "ByCreateTimeAsc",
    ) -> list[dict[str, Any]]:
        return []

    def get_document_raw_content(self, document_id: str) -> str:
        return ""

    def get_doc_markdown(self, doc_token: str, doc_type: str = "docx") -> str:
        return ""

    def get_wiki_node(self, node_token: str) -> dict[str, Any]:
        return {}

    def get_task(self, task_guid: str) -> dict[str, Any]:
        return {}

    def list_subtasks(self, task_guid: str, page_size: int = 50) -> list[dict[str, Any]]:
        return []

    def complete_task(self, task_guid: str) -> None:
        return None

    def update_document_status(self, request: DeploymentRequest, text: str) -> None:
        return None


def make_feishu_client(config: FeishuConfig) -> FeishuOpenAPIClient | NullFeishuClient:
    if config.send_notifications:
        return FeishuOpenAPIClient(config)
    return NullFeishuClient()


def _message_id(response: dict[str, Any]) -> str:
    data = response.get("data") if isinstance(response, dict) else {}
    if not isinstance(data, dict):
        return ""
    value = data.get("message_id") or data.get("message", {}).get("message_id")
    return str(value or "")


def _extract_text(event: dict[str, Any], message: dict[str, Any]) -> str:
    for key in ("doc_text", "text", "content"):
        value = event.get(key)
        text = _coerce_message_text(value)
        if text:
            return text
    return _coerce_message_text(message.get("content"))


def _coerce_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "content"):
            nested = _coerce_message_text(value.get(key))
            if nested:
                return nested
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return _coerce_message_text(parsed) or stripped
        return stripped
    return str(value)


def _detect_source_type(event: dict[str, Any], message: dict[str, Any]) -> SourceType:
    if event.get("document_id") or event.get("obj_token") or event.get("doc_text"):
        return SourceType.DOC
    if message or event.get("chat_id") or event.get("message_id"):
        return SourceType.GROUP_MESSAGE
    return SourceType.MANUAL


def _extract_requester(sender: dict[str, Any], event: dict[str, Any]) -> Requester:
    sender_id = sender.get("sender_id") if isinstance(sender, dict) else {}
    if not isinstance(sender_id, dict):
        sender_id = {}
    user_id = (
        sender_id.get("open_id")
        or sender_id.get("user_id")
        or sender_id.get("union_id")
        or event.get("operator_id")
        or event.get("user_id")
        or "unknown"
    )
    return Requester(user_id=str(user_id), display_name=str(sender.get("sender_name", "")))
