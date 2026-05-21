from __future__ import annotations

import subprocess
from typing import Any

import pytest

from fmh.config import ReusableWorkersConfig, WeightConversionConfig
from fmh.weight_conversion import detect_weight_format, plan_weight_conversion, run_weight_conversion


def test_plan_weight_conversion_uses_normalized_worker_path() -> None:
    plan = plan_weight_conversion(
        "/mnt/shared-storage-user/ma4agi-gpu/team_alpha/vita/model_ckpt/run/iter_0000005",
        WeightConversionConfig(
            enabled=True,
            source_prefixes=["/mnt/gpfs/ma4agi-gpu/team_alpha"],
            output_basename_prefix="hf_",
        ),
        ReusableWorkersConfig(
            source_model_prefix="/mnt/shared-storage-user/ma4agi-gpu",
            worker_model_prefix="/mnt/gpfs/ma4agi-gpu",
            table_model_prefix="/mnt/gpfs/ma4agi-gpu",
        ),
    )

    assert plan is not None
    assert plan.original_weight_path == "/mnt/shared-storage-user/ma4agi-gpu/team_alpha/vita/model_ckpt/run/iter_0000005"
    assert plan.input_path == "/mnt/gpfs/ma4agi-gpu/team_alpha/vita/model_ckpt/run/iter_0000005"
    assert plan.output_path == "/mnt/gpfs/ma4agi-gpu/team_alpha/vita/model_ckpt/run/hf_iter_0000005"


def test_plan_weight_conversion_handles_requested_checkpoint_shape() -> None:
    raw_path = (
        "/mnt/shared-storage-user/ma4agi-gpu/team_alpha/vita/model_ckpt/"
        "training_family_0514_1280/"
        "model-run-0514-1280-20260520_031524/"
        "iter_0000005"
    )
    expected_input = (
        "/mnt/gpfs/ma4agi-gpu/team_alpha/vita/model_ckpt/"
        "training_family_0514_1280/"
        "model-run-0514-1280-20260520_031524/"
        "iter_0000005"
    )
    expected_output = (
        "/mnt/gpfs/ma4agi-gpu/team_alpha/vita/model_ckpt/"
        "training_family_0514_1280/"
        "model-run-0514-1280-20260520_031524/"
        "hf_iter_0000005"
    )

    plan = plan_weight_conversion(
        raw_path,
        WeightConversionConfig(
            enabled=True,
            source_prefixes=["/mnt/gpfs/ma4agi-gpu/team_alpha"],
            output_basename_prefix="hf_",
        ),
        ReusableWorkersConfig(
            source_model_prefix="/mnt/shared-storage-user/ma4agi-gpu",
            worker_model_prefix="/mnt/gpfs/ma4agi-gpu",
            table_model_prefix="/mnt/gpfs/ma4agi-gpu",
        ),
    )

    assert plan is not None
    assert plan.original_weight_path == raw_path
    assert plan.input_path == expected_input
    assert plan.output_path == expected_output
    assert plan.to_dict() == {
        "input_path": expected_input,
        "output_path": expected_output,
        "original_weight_path": raw_path,
        "detected_format": "",
        "required": True,
    }


def test_plan_weight_conversion_uses_format_detection(tmp_path) -> None:
    distcp_dir = tmp_path / "distcp_iter"
    distcp_dir.mkdir()
    (distcp_dir / "__0_0.distcp").write_text("shard")
    (distcp_dir / "common.pt").write_text("common")
    (distcp_dir / "metadata.json").write_text("{}")

    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "model.safetensors.index.json").write_text("{}")

    config = WeightConversionConfig(
        enabled=True,
        source_prefixes=[str(tmp_path)],
        format_detection_enabled=True,
        remote_format_detection=False,
    )

    plan = plan_weight_conversion(str(distcp_dir), config)

    assert detect_weight_format(str(distcp_dir), config) == "distcp"
    assert plan is not None
    assert plan.detected_format == "distcp"
    assert plan.output_path == str(tmp_path / "hf_distcp_iter")
    assert detect_weight_format(str(hf_dir), config) == "hf"
    assert plan_weight_conversion(str(hf_dir), config) is None


def test_plan_weight_conversion_rejects_unknown_format_when_detection_is_required(tmp_path) -> None:
    unknown_dir = tmp_path / "unknown_iter"
    unknown_dir.mkdir()
    config = WeightConversionConfig(
        enabled=True,
        source_prefixes=[str(tmp_path)],
        format_detection_enabled=True,
        remote_format_detection=False,
        format_detection_required=True,
    )

    with pytest.raises(RuntimeError, match="unsupported or unknown weight format"):
        plan_weight_conversion(str(unknown_dir), config)


def test_plan_weight_conversion_ignores_disabled_unmatched_and_already_converted_paths() -> None:
    reusable = ReusableWorkersConfig(
        source_model_prefix="/mnt/shared-storage-user/ma4agi-gpu",
        worker_model_prefix="/mnt/gpfs/ma4agi-gpu",
    )
    config = WeightConversionConfig(enabled=True, source_prefixes=["/mnt/gpfs/ma4agi-gpu/team_alpha"])

    assert plan_weight_conversion("/mnt/gpfs/ma4agi-gpu/zhangbo/run/iter_1", config, reusable) is None
    assert plan_weight_conversion("/mnt/gpfs/ma4agi-gpu/team_alpha/run/hf_iter_1", config, reusable) is None
    assert plan_weight_conversion(
        "/mnt/gpfs/ma4agi-gpu/team_alpha/run/iter_1",
        WeightConversionConfig(enabled=False, source_prefixes=["/mnt/gpfs/ma4agi-gpu/team_alpha"]),
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
            conda_sh="/opt/miniconda3/etc/profile.d/conda.sh",
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
    assert 'SCRIPT_DIR=$(dirname "$SCRIPT")' in script
    assert 'TMP_SCRIPT="$SCRIPT_DIR/.fmh_$(basename "$SCRIPT").$$.sh"' in script
    assert 'cp "$SCRIPT" "$TMP_SCRIPT"' in script
    assert "python3 - \"$TMP_SCRIPT\" \"$INPUT_DIR\" \"$OUTPUT_DIR\"" in script
    assert 'cd "$SCRIPT_DIR"' in script
    assert "CONDA_ENV=smile" in script
    assert "source /opt/miniconda3/etc/profile.d/conda.sh" in script
    assert "INPUT_DIR=/mnt/gpfs/team/run/iter_1" in script
    assert "OUTPUT_DIR=/mnt/gpfs/team/run/hf_iter_1" in script
    assert "conversion output exists but is not a complete HF checkpoint" in script
    assert "model.safetensors.index.json" in script
    assert "--input-dir" in script
    assert "--output-dir" in script
