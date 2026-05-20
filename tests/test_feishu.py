from __future__ import annotations

from typing import Any

import httpx
import pytest

from fmh.config import FeishuConfig, load_config
from fmh.feishu import FeishuAuthError, FeishuEventNormalizer, FeishuOpenAPIClient
from fmh.models import SourceType


class FakeResponse:
    status_code = 200
    is_error = False
    reason_phrase = "OK"
    text = ""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def json(self) -> dict[str, Any]:
        return self._data


def test_normalizer_accepts_message_receive_v1_payload() -> None:
    source = FeishuEventNormalizer().normalize(
        {
            "schema": "2.0",
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "verification-token",
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_sender"},
                    "sender_name": "sender",
                },
                "message": {
                    "message_id": "om_message",
                    "chat_id": "oc_chat",
                    "message_type": "text",
                    "content": "{\"text\":\"deploy_vllm\\nweight_path: /mnt/models/demo\"}",
                },
            },
        }
    )

    assert source.source_type == SourceType.GROUP_MESSAGE
    assert source.source_ref == "om_message"
    assert source.requester.user_id == "ou_sender"
    assert source.text == "deploy_vllm\nweight_path: /mnt/models/demo"


def test_complete_task_uses_user_access_token(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse({"code": 0, "msg": "success", "data": {}})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    client = FeishuOpenAPIClient(
        FeishuConfig(
            base_url="https://example.feishu.test/open-apis",
            user_access_token="u-test-token",
        )
    )

    client.complete_task("task-guid")

    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "PATCH"
    assert call["url"].endswith("/task/v2/tasks/task-guid")
    assert call["headers"] == {"Authorization": "Bearer u-test-token"}
    assert call["params"] == {"user_id_type": "open_id"}
    assert call["json"]["update_fields"] == ["completed_at"]
    assert set(call["json"]["task"]) == {"completed_at"}


def test_complete_task_requires_user_access_token(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse({"code": 0, "msg": "success", "data": {}})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    client = FeishuOpenAPIClient(FeishuConfig(base_url="https://example.feishu.test/open-apis"))

    with pytest.raises(FeishuAuthError, match="user_access_token is required"):
        client.complete_task("task-guid")

    assert calls == []


def test_add_message_reaction_uses_tenant_token(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return FakeResponse({"code": 0, "tenant_access_token": "t-test-token", "expire": 7200})
        return FakeResponse({"code": 0, "msg": "success", "data": {"reaction_id": "r_1"}})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    client = FeishuOpenAPIClient(
        FeishuConfig(
            base_url="https://example.feishu.test/open-apis",
            app_id="app",
            app_secret="secret",
        )
    )

    reaction_id = client.add_message_reaction("om_1", "SALUTE")

    assert reaction_id == "r_1"
    call = calls[-1]
    assert call["method"] == "POST"
    assert call["url"].endswith("/im/v1/messages/om_1/reactions")
    assert call["headers"] == {"Authorization": "Bearer t-test-token"}
    assert call["json"] == {"reaction_type": {"emoji_type": "SALUTE"}}


def test_get_bot_open_id_accepts_legacy_bot_info_shape(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return FakeResponse({"code": 0, "tenant_access_token": "t-test-token", "expire": 7200})
        return FakeResponse({"code": 0, "msg": "ok", "bot": {"open_id": "ou_bot"}})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    client = FeishuOpenAPIClient(
        FeishuConfig(
            base_url="https://example.feishu.test/open-apis",
            app_id="app",
            app_secret="secret",
        )
    )

    assert client.get_bot_open_id() == "ou_bot"
    assert calls[-1]["method"] == "GET"
    assert calls[-1]["url"].endswith("/bot/v3/info")


def test_request_retries_temporary_network_errors(monkeypatch) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append(url)
        if len(calls) < 5:
            raise httpx.ConnectError("dns failed")
        return FakeResponse({"code": 0, "msg": "success", "data": {"ok": True}})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    monkeypatch.setattr("fmh.feishu.time.sleep", lambda delay: sleeps.append(delay))
    client = FeishuOpenAPIClient(FeishuConfig(base_url="https://example.feishu.test/open-apis"))

    assert client._request("GET", "/ping", auth=False)["data"] == {"ok": True}
    assert len(calls) == 5
    assert sleeps == [2.0, 6.0, 15.0, 30.0]


def test_request_does_not_retry_permission_errors(monkeypatch) -> None:
    calls: list[str] = []

    class ErrorResponse(FakeResponse):
        status_code = 403
        is_error = True
        reason_phrase = "Forbidden"

    def fake_request(method: str, url: str, **kwargs: Any) -> ErrorResponse:
        calls.append(url)
        return ErrorResponse({"code": 99991672, "msg": "forbidden"})

    monkeypatch.setattr("fmh.feishu.httpx.request", fake_request)
    monkeypatch.setattr("fmh.feishu.time.sleep", lambda delay: None)
    client = FeishuOpenAPIClient(FeishuConfig(base_url="https://example.feishu.test/open-apis"))

    with pytest.raises(RuntimeError, match="Feishu HTTP 403"):
        client._request("GET", "/forbidden", auth=False)

    assert len(calls) == 1


def test_feishu_user_access_token_env_override(monkeypatch) -> None:
    monkeypatch.delenv("FMH_CONFIG", raising=False)
    monkeypatch.setenv("FEISHU_USER_ACCESS_TOKEN", "u-env-token")

    config = load_config()

    assert config.feishu.user_access_token == "u-env-token"
