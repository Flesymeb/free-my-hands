from __future__ import annotations

from fmh.task_status import task_status_card, task_status_with_stage


def test_task_status_card_uses_table_like_fields_and_status_tags() -> None:
    state = {}
    state = task_status_with_stage(
        state,
        "detect",
        "完成",
        "发现 1 个候选子任务。",
        title="测试任务",
        source_ref="task_1",
        model_id="model-a",
        model="team/very/long/path/to/model-a",
        address="192.0.2.2 (4卡)",
    )
    state = task_status_with_stage(state, "execute", "进行中", "正在启动 vLLM。")

    card = task_status_card(state)
    rendered = str(card)
    progress = _rendered_text(card)

    assert "**父任务**" not in rendered
    assert "来源任务：测试任务" in rendered
    assert "**模型**\\nteam/very/long/path/to/model-a" in rendered
    assert "**模型ID**" not in rendered
    assert "**模型**" in rendered
    assert "**地址**" in rendered
    assert "<text_tag color='blue'>vLLM启动中</text_tag>" in rendered
    assert "endpoint" not in rendered
    assert "流程进度" not in rendered
    assert "子任务完成 <text_tag color='grey'>等待中</text_tag>" not in rendered
    assert "**任务检测**" in progress
    assert "**执行情况**" in progress
    assert "<text_tag color='blue'>进行中</text_tag>" in progress
    assert "<text_tag color='green'>完成</text_tag>" in progress
    assert "`进行中`" not in progress
    assert card["header"]["template"] == "blue"
    assert len(card["elements"]) == 5


def test_task_status_card_marks_deployed_green() -> None:
    state = task_status_with_stage(
        {},
        "document",
        "完成",
        "已回填文档。",
        deploy_status="已部署",
    )

    assert task_status_card(state)["header"]["template"] == "green"


def test_task_status_card_ignores_legacy_subtask_completion_stage() -> None:
    state = task_status_with_stage(
        {},
        "execute",
        "完成",
        "模型已通过检查。",
        deploy_status="已部署",
    )
    state = task_status_with_stage(state, "todo", "需人工", "子任务完成失败。")

    assert state["deploy_status"] == "已部署"
    card = task_status_card(state)
    rendered = str(card)
    assert card["header"]["template"] == "green"
    assert "子任务完成" not in rendered
    assert "子任务完成失败" not in rendered
    assert "重试" not in rendered


def test_task_status_clears_stale_downstream_stages_on_new_progress() -> None:
    state = task_status_with_stage({}, "execute", "完成", "old model ready", deploy_status="已部署")
    state = task_status_with_stage(state, "document", "完成", "old doc updated")
    state = task_status_with_stage(state, "todo", "需人工", "old subtask failure")

    state = task_status_with_stage(
        state,
        "codex",
        "通过",
        "new review approved",
        model_id="new-model",
        model="team/new-model",
    )

    assert set(state["stages"]) == {"codex"}
    assert state["deploy_status"] == "准备中"
    assert "execute" not in state["stages"]
    assert "document" not in state["stages"]
    assert "todo" not in state["stages"]
    rendered = str(task_status_card(state))
    assert "old doc updated" not in rendered
    assert "old subtask failure" not in rendered


def test_task_status_card_adds_retry_and_cancel_actions_for_errors() -> None:
    state = task_status_with_stage(
        {},
        "execute",
        "失败",
        "worker GPUs are still busy",
        review_id="rvw-test",
    )

    card = task_status_card(state)
    rendered = str(card)

    assert "重试" in rendered
    assert "取消" in rendered
    assert "'decision': 'RETRY'" in rendered
    assert "'decision': 'BLOCK'" in rendered
    assert "回复本卡片：重试 或 取消" in rendered
    assert "retry rvw-test" in rendered


def test_task_status_card_hides_actions_when_card_callbacks_disabled() -> None:
    state = task_status_with_stage(
        {},
        "execute",
        "失败",
        "worker GPUs are still busy",
        review_id="rvw-test",
    )
    state["card_actions_enabled"] = False

    card = task_status_card(state)
    rendered = str(card)

    assert "'decision': 'RETRY'" not in rendered
    assert "'decision': 'BLOCK'" not in rendered
    assert "回复本卡片：重试 或 取消" in rendered
    assert "retry rvw-test" in rendered


def test_task_status_card_keeps_long_conversion_path_tail_visible() -> None:
    output_path = (
        "/mnt/gpfs/ma4agi-gpu/team_alpha/project/model_ckpt/run_group_0514_1280/"
        "example-run-0514-1280-20260520_031524/hf_iter_0000005"
    )
    state = task_status_with_stage(
        {},
        "convert",
        "进行中",
        f"正在转换到 {output_path}。",
        model_id="hf_iter_0000005",
        model=output_path,
    )

    rendered = _rendered_text(task_status_card(state))

    assert "正在转换到 .../" in rendered
    assert "hf\\_iter\\_0000005。" in rendered
    assert "run_group_05…" not in rendered


def _rendered_text(card: dict[str, object]) -> str:
    chunks: list[str] = []
    for element in card.get("elements", []):
        if not isinstance(element, dict):
            continue
        text = element.get("text")
        if isinstance(text, dict):
            chunks.append(str(text.get("content") or ""))
        fields = element.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if isinstance(field, dict) and isinstance(field.get("text"), dict):
                    chunks.append(str(field["text"].get("content") or ""))
    return "\n".join(chunks)
