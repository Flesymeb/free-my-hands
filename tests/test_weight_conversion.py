from __future__ import annotations

import subprocess
from typing import Any

import pytest

from fmh.config import ReusableWorkersConfig, WeightConversionConfig
from fmh.weight_conversion import plan_weight_conversion, run_weight_conversion


def test_plan_weight_conversion_uses_normalized_worker_path() -> None:
    plan = plan_weight_conversion(
        "/mnt/shared-storage-user/ma4agi-gpu/zhangchen/vita/model_ckpt/run/iter_0000005",
        WeightConversionConfig(
            enabled=True,
            source_prefixes=["/mnt/gpfs/ma4agi-gpu/zhangchen"],
            output_basename_prefix="hf_",
        ),
        ReusableWorkersConfig(
            source_model_prefix="/mnt/shared-storage-user/ma4agi-gpu",
            worker_model_prefix="/mnt/gpfs/ma4agi-gpu",
            table_model_prefix="/mnt/gpfs/ma4agi-gpu",
        ),
    )

    assert plan is not None
    assert plan.original_weight_path == "/mnt/shared-storage-user/ma4agi-gpu/zhangchen/vita/model_ckpt/run/iter_0000005"
    assert plan.input_path == "/mnt/gpfs/ma4agi-gpu/zhangchen/vita/model_ckpt/run/iter_0000005"
    assert plan.output_path == "/mnt/gpfs/ma4agi-gpu/zhangchen/vita/model_ckpt/run/hf_iter_0000005"


def test_plan_weight_conversion_ignores_disabled_unmatched_and_already_converted_paths() -> None:
    reusable = ReusableWorkersConfig(
        source_model_prefix="/mnt/shared-storage-user/ma4agi-gpu",
        worker_model_prefix="/mnt/gpfs/ma4agi-gpu",
    )
    config = WeightConversionConfig(enabled=True, source_prefixes=["/mnt/gpfs/ma4agi-gpu/zhangchen"])

    assert plan_weight_conversion("/mnt/gpfs/ma4agi-gpu/zhangbo/run/iter_1", config, reusable) is None
    assert plan_weight_conversion("/mnt/gpfs/ma4agi-gpu/zhangchen/run/hf_iter_1", config, reusable) is None
    assert plan_weight_conversion(
        "/mnt/gpfs/ma4agi-gpu/zhangchen/run/iter_1",
        WeightConversionConfig(enabled=False, source_prefixes=["/mnt/gpfs/ma4agi-gpu/zhangchen"]),
        reusable,
    ) is None


def test_run_weight_conversion_patches_remote_script_and_invokes_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("fmh.weight_conversion.subprocess.run", fake_run)

    result = run_weight_conversion(
        {
            "input_path": "/mnt/gpfs/team/run/iter_1",
            "output_path": "/mnt/gpfs/team/run/hf_iter_1",
            "original_weight_path": "/mnt/shared/team/run/iter_1",
        },
        WeightConversionConfig(
            enabled=True,
            host="converter@example.com",
            conda_env="smile",
            script_path="/opt/trans.sh",
            timeout_sec=123,
        ),
    )

    assert result.output_path == "/mnt/gpfs/team/run/hf_iter_1"
    call = calls[0]
    assert call["args"] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=3",
        "-CAXY",
        "converter@example.com",
        "bash",
        "-s",
    ]
    assert call["timeout"] == 123
    script = str(call["input"])
    assert "SCRIPT=/opt/trans.sh" in script
    assert "CONDA_ENV=smile" in script
    assert "INPUT_DIR=/mnt/gpfs/team/run/iter_1" in script
    assert "OUTPUT_DIR=/mnt/gpfs/team/run/hf_iter_1" in script
    assert "--input-dir" in script
    assert "--output-dir" in script
