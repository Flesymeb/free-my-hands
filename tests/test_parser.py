from __future__ import annotations

import pytest

from fmh.models import EventSource, Requester, SourceType
from fmh.parser import ParseError, parse_deployment_request


def test_parse_structured_request() -> None:
    text = """deploy_vllm
weight_path: /mnt/checkpoints/qwen
model_name: qwen-test
gpu_count: 2
gpu_type: A100
extra_args: --max-model-len 32768
document_id: doc_test
status_block_id: blk_test
"""
    request = parse_deployment_request(_source(text))

    assert request.weight_path == "/mnt/checkpoints/qwen"
    assert request.model_name == "qwen-test"
    assert request.gpu_count == 2
    assert request.gpu_type == "A100"
    assert request.extra_args == "--max-model-len 32768"
    assert request.metadata["feishu"]["document_id"] == "doc_test"
    assert request.metadata["feishu"]["status_block_id"] == "blk_test"
    assert request.request_id.startswith("req-")


def test_parse_chinese_aliases() -> None:
    request = parse_deployment_request(
        _source(
            """部署模型
权重路径: /mnt/models/demo
卡数: 1
卡型: H100
"""
        )
    )

    assert request.weight_path == "/mnt/models/demo"
    assert request.model_name == "demo"
    assert request.gpu_type == "H100"


def test_rejects_implicit_path_without_trigger() -> None:
    with pytest.raises(ParseError):
        parse_deployment_request(_source("please check /mnt/models/demo later"))


def _source(text: str) -> EventSource:
    return EventSource(
        source_type=SourceType.GROUP_MESSAGE,
        source_ref="om_test",
        requester=Requester(user_id="ou_test"),
        text=text,
    )
