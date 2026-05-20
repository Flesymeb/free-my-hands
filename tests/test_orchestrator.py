from __future__ import annotations

from fmh.config import AppConfig, RLaunchConfig, RunnerConfig, StorageConfig, VLLMConfig
from fmh.feishu import NullFeishuClient
from fmh.models import EventSource, RequestStatus, Requester, SourceType
from fmh.orchestrator import DeploymentOrchestrator
from fmh.parser import parse_deployment_request
from fmh.runner import make_runner
from fmh.store import StateStore


def test_orchestrator_dry_run(tmp_path) -> None:
    config = AppConfig(
        storage=StorageConfig(sqlite_path=str(tmp_path / "state.sqlite3")),
        runner=RunnerConfig(mode="dry-run", workdir=str(tmp_path), log_dir=str(tmp_path / "logs")),
        rlaunch=RLaunchConfig(
            command_template="echo rlaunch --gpu {gpu_count} --type {gpu_type}",
            setup_commands=["echo setup"],
        ),
        vllm=VLLMConfig(port_start=19000, command_template="echo vllm {weight_path} {port}"),
    )
    store = StateStore(config.storage.sqlite_path)
    orchestrator = DeploymentOrchestrator(
        config=config,
        store=store,
        runner=make_runner(config.runner),
        notifier=NullFeishuClient(),
    )
    request = parse_deployment_request(
        EventSource(
            source_type=SourceType.MANUAL,
            source_ref="unit",
            requester=Requester(user_id="u1"),
            text="deploy_vllm\nweight_path: /mnt/model\ngpu_type: A100",
        )
    )

    result = orchestrator.submit(request)

    assert result.status == RequestStatus.DRY_RUN_COMPLETE
    assert result.tmux_session.startswith("fmh-")
    assert result.endpoint.startswith("http://127.0.0.1:")
    assert (tmp_path / "logs").exists()
