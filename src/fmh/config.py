from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str = ""
    app_secret: str = ""
    user_access_token: str = ""
    verification_token: str = ""
    base_url: str = "https://open.feishu.cn/open-apis"
    send_notifications: bool = False
    default_chat_id: str = ""


@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: str = "data/free-my-hands.sqlite3"


@dataclass(frozen=True)
class RunnerConfig:
    mode: str = "dry-run"
    workdir: str = "."
    log_dir: str = "logs"
    tmux_prefix: str = "fmh"


@dataclass(frozen=True)
class RLaunchConfig:
    command_template: str = ""
    setup_commands: list[str] = field(
        default_factory=lambda: ["source ~/.bashrc", "conda activate vllm"]
    )


@dataclass(frozen=True)
class VLLMConfig:
    host: str = "0.0.0.0"
    public_host: str = "127.0.0.1"
    port_start: int = 18000
    health_timeout_sec: int = 180
    health_interval_sec: int = 5
    default_extra_args: str = ""
    command_template: str = (
        "python -m vllm.entrypoints.openai.api_server "
        "--model {weight_path} "
        "--served-model-name {model_name} "
        "--host {host} "
        "--port {port} "
        "{extra_args}"
    )


@dataclass(frozen=True)
class PollingConfig:
    interval_sec: int = 20
    page_size: int = 50
    chat_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    initial_lookback_sec: int = 0
    process_existing_on_first_run: bool = False
    notify_chat_on_accept: bool = True
    deploy_todo_subtasks: bool = True
    watch_known_todo_tasks: bool = True
    wake_review_auditor_on_submit: bool = True
    manual_poll_lookback_sec: int = 600
    known_todo_check_interval_sec: int = 600
    known_todo_max_per_tick: int = 1
    relative_weight_path_prefix: str = ""
    max_parse_failures_before_handoff: int = 3
    ignore_self_messages: bool = True
    reuse_plan_retry_delay_sec: int = 1800
    task_detected_reaction_emoji: str = "SALUTE"


@dataclass(frozen=True)
class ReusableWorkersConfig:
    enabled: bool = False
    auto_deploy_approved: bool = False
    deployed_models_doc_token: str = ""
    tutorial_doc_token: str = ""
    dev_host: str = ""
    source_model_prefix: str = "/mnt/shared-models"
    worker_model_prefix: str = "/mnt/worker-models"
    table_model_prefix: str = "/mnt/worker-models"
    running_marker: str = "(running)"
    deploying_marker: str = "（部署中）"
    required_finished_tasks: list[str] = field(default_factory=lambda: ["tau2", "vita"])
    default_port: int = 8000
    default_tool_parser: str = "qwen3_coder"
    default_reasoning_parser: str = "qwen3"
    default_max_model_len: int = 262144
    default_gpu_memory_utilization: float = 0.90
    default_api_server_count: int = 4
    deploy_timeout_sec: int = 1800
    deploy_poll_interval_sec: int = 15
    max_parallel_deployments: int = 2


@dataclass(frozen=True)
class ApprovalConfig:
    fallback_chat_id: str = ""
    fallback_mention_open_id: str = ""
    fallback_mention_name: str = ""
    allow_card_actions: bool = False
    allow_group_commands: bool = True


@dataclass(frozen=True)
class PostDeployNotifyConfig:
    enabled: bool = False
    target_open_id: str = ""
    target_name: str = ""
    chat_id: str = ""
    card_enabled: bool = True
    message_template: str = (
        '<at user_id="{target_open_id}">{target_name}</at> 新模型已部署，请处理。\n'
        "worker: {worker}\n"
        "model_id: {model_id}\n"
        "endpoint: {endpoint}"
    )


@dataclass(frozen=True)
class CodexReviewConfig:
    enabled: bool = False
    command: list[str] | str = field(
        default_factory=lambda: [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
        ]
    )
    cd: str = "."
    interval_sec: int = 10
    timeout_sec: int = 300
    max_output_chars: int = 12000
    send_result_cards: bool = True
    runtime_toggle_key: str = "codex_review_enabled"


@dataclass(frozen=True)
class AppConfig:
    feishu: FeishuConfig = field(default_factory=FeishuConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    rlaunch: RLaunchConfig = field(default_factory=RLaunchConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    reusable_workers: ReusableWorkersConfig = field(default_factory=ReusableWorkersConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    post_deploy_notify: PostDeployNotifyConfig = field(default_factory=PostDeployNotifyConfig)
    codex_review: CodexReviewConfig = field(default_factory=CodexReviewConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    config_path = path or os.getenv("FMH_CONFIG")
    if config_path:
        with Path(config_path).expanduser().open("rb") as f:
            raw = tomllib.load(f)
        config = _merge_config(config, raw)
    return _apply_env_overrides(config)


def _merge_config(config: AppConfig, raw: dict[str, Any]) -> AppConfig:
    return replace(
        config,
        feishu=replace(config.feishu, **raw.get("feishu", {})),
        storage=replace(config.storage, **raw.get("storage", {})),
        runner=replace(config.runner, **raw.get("runner", {})),
        rlaunch=replace(config.rlaunch, **raw.get("rlaunch", {})),
        vllm=replace(config.vllm, **raw.get("vllm", {})),
        polling=replace(config.polling, **raw.get("polling", {})),
        reusable_workers=replace(config.reusable_workers, **raw.get("reusable_workers", {})),
        approval=replace(config.approval, **raw.get("approval", {})),
        post_deploy_notify=replace(config.post_deploy_notify, **raw.get("post_deploy_notify", {})),
        codex_review=replace(config.codex_review, **raw.get("codex_review", {})),
    )


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    feishu = config.feishu
    storage = config.storage
    runner = config.runner
    reusable_workers = config.reusable_workers
    approval = config.approval

    if os.getenv("FEISHU_APP_ID"):
        feishu = replace(feishu, app_id=os.environ["FEISHU_APP_ID"])
    if os.getenv("FEISHU_APP_SECRET"):
        feishu = replace(feishu, app_secret=os.environ["FEISHU_APP_SECRET"])
    if os.getenv("FEISHU_USER_ACCESS_TOKEN"):
        feishu = replace(feishu, user_access_token=os.environ["FEISHU_USER_ACCESS_TOKEN"])
    if os.getenv("FEISHU_VERIFICATION_TOKEN"):
        feishu = replace(feishu, verification_token=os.environ["FEISHU_VERIFICATION_TOKEN"])
    if os.getenv("FEISHU_CHAT_ID"):
        feishu = replace(feishu, default_chat_id=os.environ["FEISHU_CHAT_ID"])
    if os.getenv("FMH_SQLITE_PATH"):
        storage = replace(storage, sqlite_path=os.environ["FMH_SQLITE_PATH"])
    if os.getenv("FMH_RUNNER_MODE"):
        runner = replace(runner, mode=os.environ["FMH_RUNNER_MODE"])
    if os.getenv("FMH_DEPLOYED_MODELS_DOC_TOKEN"):
        reusable_workers = replace(
            reusable_workers,
            deployed_models_doc_token=os.environ["FMH_DEPLOYED_MODELS_DOC_TOKEN"],
        )
    if os.getenv("FMH_TUTORIAL_DOC_TOKEN"):
        reusable_workers = replace(reusable_workers, tutorial_doc_token=os.environ["FMH_TUTORIAL_DOC_TOKEN"])
    if os.getenv("FMH_REUSABLE_WORKERS_DEV_HOST"):
        reusable_workers = replace(reusable_workers, dev_host=os.environ["FMH_REUSABLE_WORKERS_DEV_HOST"])
    if os.getenv("FMH_APPROVAL_FALLBACK_CHAT_ID"):
        approval = replace(approval, fallback_chat_id=os.environ["FMH_APPROVAL_FALLBACK_CHAT_ID"])
    if os.getenv("FMH_APPROVAL_MENTION_OPEN_ID"):
        approval = replace(approval, fallback_mention_open_id=os.environ["FMH_APPROVAL_MENTION_OPEN_ID"])
    if os.getenv("FMH_APPROVAL_MENTION_NAME"):
        approval = replace(approval, fallback_mention_name=os.environ["FMH_APPROVAL_MENTION_NAME"])

    post_deploy_notify = config.post_deploy_notify
    if os.getenv("FMH_POST_DEPLOY_NOTIFY_ENABLED"):
        enabled = os.environ["FMH_POST_DEPLOY_NOTIFY_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
        post_deploy_notify = replace(post_deploy_notify, enabled=enabled)
    if os.getenv("FMH_POST_DEPLOY_NOTIFY_OPEN_ID"):
        post_deploy_notify = replace(post_deploy_notify, target_open_id=os.environ["FMH_POST_DEPLOY_NOTIFY_OPEN_ID"])
    if os.getenv("FMH_POST_DEPLOY_NOTIFY_NAME"):
        post_deploy_notify = replace(post_deploy_notify, target_name=os.environ["FMH_POST_DEPLOY_NOTIFY_NAME"])
    if os.getenv("FMH_POST_DEPLOY_NOTIFY_CHAT_ID"):
        post_deploy_notify = replace(post_deploy_notify, chat_id=os.environ["FMH_POST_DEPLOY_NOTIFY_CHAT_ID"])
    if os.getenv("FMH_POST_DEPLOY_NOTIFY_CARD_ENABLED"):
        card_enabled = os.environ["FMH_POST_DEPLOY_NOTIFY_CARD_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
        post_deploy_notify = replace(post_deploy_notify, card_enabled=card_enabled)

    return replace(
        config,
        feishu=feishu,
        storage=storage,
        runner=runner,
        reusable_workers=reusable_workers,
        approval=approval,
        post_deploy_notify=post_deploy_notify,
    )
