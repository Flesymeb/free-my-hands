from __future__ import annotations

import pytest

from fmh.config import ReusableWorkersConfig
from fmh.reusable_workers import (
    build_new_worker_row_plan,
    build_reconnect_plan,
    build_reusable_deployment_plan,
    choose_reusable_row,
    normalize_model_path,
    parse_deployed_models_table,
)


MARKDOWN = """# 已部署模型

<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/running</td><td>running</td><td>198\\.51\\.100\\.44（8卡）</td><td>qwen3\\_coder</td><td>qwen3</td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td>old/free</td><td>free</td><td>192\\.0\\.2\\.156（4卡）</td><td>qwen3\\_coder</td><td>qwen3</td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>
"""


def test_parse_and_choose_reusable_row() -> None:
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(MARKDOWN)

    assert len(rows) == 2
    assert rows[0].ip == "198.51.100.44"
    assert rows[0].gpu_count == 8
    assert not rows[0].is_reusable(config)
    assert rows[1].is_reusable(config)


def test_idle_empty_row_is_reusable_but_fresh_untested_is_not() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>new/model</td><td>model</td><td>192\\.0\\.2\\.2（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.3（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>old/free</td><td>free</td><td>192\\.0\\.2\\.4（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert not rows[0].is_reusable(config)
    assert rows[1].is_reusable(config)
    assert rows[2].is_reusable(config)
    assert choose_reusable_row(rows, config) == rows[1]


def test_reusable_worker_selection_covers_idle_finished_running_and_fresh_rows() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.10（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>old/finished</td><td>finished</td><td>192\\.0\\.2\\.11（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
<tr><td>old/running</td><td>running</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td>fresh/model</td><td>model</td><td>192\\.0\\.2\\.13（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert [row.is_reusable(config) for row in rows] == [True, True, False, False]
    assert choose_reusable_row(rows, config) == rows[0]


@pytest.mark.parametrize(
    ("tested_tasks", "expected"),
    [
        ("TAU2\nVITA", True),
        ("tau2, vita", True),
        ("tau2 done\nvita done", True),
        ("tau2\nvita\\(running\\)", False),
        ("tau2\nvita（running）", False),
        ("tau2\nvita ( running )", False),
        ("tau20\nvitamin", False),
    ],
)
def test_tested_tasks_parsing_uses_exact_tasks_and_running_marker_variants(
    tested_tasks: str,
    expected: bool,
) -> None:
    markdown = f"""<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/model</td><td>model</td><td>192\\.0\\.2\\.20（4卡）</td><td></td><td></td><td></td><td>{tested_tasks}</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert rows[0].is_reusable(config) is expected


def test_reusable_worker_selection_skips_marker_variants_and_task_name_false_positives() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/fullwidth-running</td><td>running-a</td><td>192\\.0\\.2\\.20（4卡）</td><td></td><td></td><td></td><td>tau2\nvita（running）</td><td></td></tr>
<tr><td>old/spaced-running</td><td>running-b</td><td>192\\.0\\.2\\.21（4卡）</td><td></td><td></td><td></td><td>tau2\nvita ( running )</td><td></td></tr>
<tr><td>old/false-positive</td><td>false-positive</td><td>192\\.0\\.2\\.22（4卡）</td><td></td><td></td><td></td><td>tau20\nvitamin</td><td></td></tr>
<tr><td>old/finished</td><td>finished</td><td>192\\.0\\.2\\.23（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert [row.is_reusable(config) for row in rows] == [False, False, False, True]
    assert choose_reusable_row(rows, config) == rows[3]


def test_reusable_worker_selection_uses_finished_row_when_no_idle_row_exists() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>fresh/model</td><td>model</td><td>192\\.0\\.2\\.13（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>old/running</td><td>running</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td>old/finished</td><td>finished</td><td>192\\.0\\.2\\.11（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert choose_reusable_row(rows, config) == rows[2]


def test_reusable_worker_selection_skips_one_running_row_and_uses_available_finished_row() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/running</td><td>running</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td>old/finished</td><td>finished</td><td>192\\.0\\.2\\.11（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert [row.is_reusable(config) for row in rows] == [False, True]
    assert choose_reusable_row(rows, config) == rows[1]


def test_reusable_worker_selection_skips_two_running_rows_and_uses_idle_row() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>old/running-a</td><td>running-a</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\\(running\\)\nvita</td><td></td></tr>
<tr><td>old/running-b</td><td>running-b</td><td>192\\.0\\.2\\.13（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.14（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert [row.is_reusable(config) for row in rows] == [False, False, True]
    assert choose_reusable_row(rows, config) == rows[2]


def test_reusable_worker_selection_rejects_only_running_and_fresh_rows() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td>fresh/model</td><td>model</td><td>192\\.0\\.2\\.13（4卡）</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>old/running</td><td>running</td><td>192\\.0\\.2\\.12（4卡）</td><td></td><td></td><td></td><td>tau2\nvita\\(running\\)</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert choose_reusable_row(rows, config) is None
    with pytest.raises(ValueError, match="no reusable deployed-model row"):
        build_reusable_deployment_plan(
            markdown,
            "/mnt/shared-models/team/foo/bar",
            config,
        )


def test_doc_parser_finds_deployed_models_table_after_unrelated_table_and_th_headers() -> None:
    markdown = """# 文档说明
<table><tbody>
<tr><th>项目</th><th>备注</th></tr>
<tr><td>说明</td><td>这不是节点表</td></tr>
</tbody></table>

<table><tbody>
<tr><th>模型</th><th>模型id</th><th>地址</th><th>推理工具调用解析器</th><th>推理解析器</th><th>SSH转发命令</th><th>已经测试完的任务</th><th>vpn排除命令</th></tr>
<tr><td>old/finished</td><td>finished</td><td>192\\.0\\.2\\.11（4卡）</td><td></td><td></td><td></td><td>tau2<br/>vita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()

    rows = parse_deployed_models_table(markdown)
    plan = build_reusable_deployment_plan(
        markdown,
        "/mnt/shared-models/team/foo/bar",
        config,
    )

    assert len(rows) == 1
    assert rows[0].ip == "192.0.2.11"
    assert rows[0].tested_tasks == "tau2\nvita"
    assert plan.row == rows[0]


def test_doc_parser_returns_no_rows_when_deployed_models_table_is_missing() -> None:
    markdown = """<table><tbody>
<tr><th>项目</th><th>备注</th></tr>
<tr><td>说明</td><td>这不是节点表</td></tr>
</tbody></table>"""

    assert parse_deployed_models_table(markdown) == []


def test_reusable_worker_selection_ignores_rows_without_valid_node_address_or_gpu_count() -> None:
    markdown = """<table><tbody>
<tr><td>模型</td><td>模型id</td><td>地址</td><td>推理工具调用解析器</td><td>推理解析器</td><td>SSH转发命令</td><td>已经测试完的任务</td><td>vpn排除命令</td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td>192\\.0\\.2\\.21</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>old/free</td><td>free</td><td>192\\.0\\.2\\.22（4卡）</td><td></td><td></td><td></td><td>tau2\nvita</td><td></td></tr>
</tbody></table>"""
    config = ReusableWorkersConfig()
    rows = parse_deployed_models_table(markdown)

    assert [row.ip for row in rows] == ["192.0.2.21", "192.0.2.22"]
    assert [row.gpu_count for row in rows] == [0, 4]
    assert choose_reusable_row(rows, config) == rows[1]


def test_normalize_model_path() -> None:
    config = ReusableWorkersConfig()
    info = normalize_model_path(
        "/mnt/shared-models/team/foo/bar",
        config,
    )

    assert info.worker_path == "/mnt/worker-models/team/foo/bar"
    assert info.table_path == "team/foo/bar"
    assert info.model_id == "bar"


def test_build_reuse_plan() -> None:
    config = ReusableWorkersConfig()
    plan = build_reusable_deployment_plan(
        MARKDOWN,
        "/mnt/shared-models/team/foo/bar",
        config,
    )

    assert plan.row.ip == "192.0.2.156"
    assert plan.tmux_session_guess == "ssh_4_gpu_2_156"
    assert "192.0.2.156" in plan.health_check_command
    assert "--model /mnt/worker-models/team/foo/bar" in plan.vllm_command
    assert plan.final_table_values["模型"] == "team/foo/bar"
    assert plan.final_table_values["模型id"] == "bar"
    assert plan.deploying_table_values["已经测试完的任务"] == ""
    assert plan.final_table_values["已经测试完的任务"] == ""


def test_build_reconnect_plan_adds_keepalive() -> None:
    config = ReusableWorkersConfig()
    row = parse_deployed_models_table(MARKDOWN)[1]
    plan = build_reconnect_plan(
        row,
        config,
        windows=["ssh", "test-model"],
        pane_history="ssh root@192.0.2.156\n",
    )

    assert plan.tmux_session == "ssh_4_gpu_2_156"
    assert "ServerAliveInterval=60" in plan.reconnect_command
    assert "ServerAliveCountMax=3" in plan.reconnect_command
    assert "UpdateHostKeys=no" in plan.reconnect_command
    assert "root@192.0.2.156" in plan.reconnect_command


def test_build_new_worker_row_plan() -> None:
    config = ReusableWorkersConfig()
    plan = build_new_worker_row_plan(
        "/mnt/shared-models/new/model",
        "198.51.100.2",
        8,
        config,
    )

    assert plan.tmux_session == "ssh_8_gpu_100_2"
    assert plan.row_values["模型"] == "new/model"
    assert plan.row_values["模型id"] == "model"
    assert "198.51.100.2" in plan.row_values["SSH转发命令"]
    assert "UpdateHostKeys=no" in plan.row_values["SSH转发命令"]
