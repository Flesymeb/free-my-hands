from __future__ import annotations

from fastapi.testclient import TestClient

from fmh.app import create_app


def test_feishu_message_webhook_processes_message_event_through_poller(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[storage]
sqlite_path = "{tmp_path / 'state.sqlite3'}"

[runner]
mode = "dry-run"
log_dir = "{tmp_path / 'logs'}"

[vllm]
command_template = "echo vllm {{weight_path}} {{port}}"
""",
        encoding="utf-8",
    )
    app = create_app(str(config_path))
    client = TestClient(app)

    response = client.post(
        "/webhooks/feishu",
        json={
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_sender"}, "sender_name": "tester"},
                "message": {
                    "message_id": "om_message",
                    "chat_id": "oc_chat",
                    "message_type": "text",
                    "create_time": "1780540000000",
                    "content": '{"text":"deploy_vllm\\nweight_path: /mnt/models/demo\\nmodel_name: demo"}',
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["source"] == "message_event"
    assert app.state.store.get_processed_item("chat:oc_chat", "om_message")["status"] == "submitted"
    deployments = app.state.store.list_requests()
    assert len(deployments) == 1
    assert deployments[0].weight_path == "/mnt/models/demo"
