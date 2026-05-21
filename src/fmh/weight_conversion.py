from __future__ import annotations

import posixpath
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from fmh.config import ReusableWorkersConfig, WeightConversionConfig
from fmh.reusable_workers import normalize_model_path


@dataclass(frozen=True)
class WeightConversionPlan:
    input_path: str
    output_path: str
    original_weight_path: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightConversionResult:
    input_path: str
    output_path: str
    returncode: int
    stdout: str
    stderr: str


def plan_weight_conversion(
    weight_path: str,
    config: WeightConversionConfig,
    reusable_config: ReusableWorkersConfig | None = None,
) -> WeightConversionPlan | None:
    if not config.enabled:
        return None
    source_prefixes = [prefix.rstrip("/") for prefix in config.source_prefixes if prefix.strip()]
    if not source_prefixes:
        return None

    original = str(weight_path or "").strip().strip("`'\"，,")
    if not original:
        return None
    input_path = normalize_model_path(original, reusable_config).worker_path if reusable_config else original
    if not any(input_path == prefix or input_path.startswith(prefix + "/") for prefix in source_prefixes):
        return None

    output_prefix = config.output_basename_prefix or "hf_"
    parent, basename = posixpath.split(input_path.rstrip("/"))
    if not basename or basename.startswith(output_prefix):
        return None
    output_path = posixpath.join(parent, output_prefix + basename)
    return WeightConversionPlan(
        input_path=input_path,
        output_path=output_path,
        original_weight_path=original,
    )


def run_weight_conversion(
    plan: WeightConversionPlan | dict[str, Any],
    config: WeightConversionConfig,
) -> WeightConversionResult:
    conversion = _coerce_plan(plan)
    _validate_config(config)
    script = _remote_conversion_script(
        script_path=config.script_path,
        conda_env=config.conda_env,
        input_path=conversion.input_path,
        output_path=conversion.output_path,
    )
    args = [
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
        config.host,
        "bash",
        "-s",
    ]
    result = subprocess.run(
        args,
        input=script,
        text=True,
        capture_output=True,
        timeout=config.timeout_sec,
        check=False,
    )
    out = WeightConversionResult(
        input_path=conversion.input_path,
        output_path=conversion.output_path,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"weight conversion failed ({result.returncode}): {_short(detail, 800)}")
    return out


def _coerce_plan(plan: WeightConversionPlan | dict[str, Any]) -> WeightConversionPlan:
    if isinstance(plan, WeightConversionPlan):
        return plan
    return WeightConversionPlan(
        input_path=str(plan.get("input_path") or ""),
        output_path=str(plan.get("output_path") or ""),
        original_weight_path=str(plan.get("original_weight_path") or ""),
        required=bool(plan.get("required", True)),
    )


def _validate_config(config: WeightConversionConfig) -> None:
    missing = [
        name
        for name, value in {
            "host": config.host,
            "conda_env": config.conda_env,
            "script_path": config.script_path,
        }.items()
        if not str(value or "").strip()
    ]
    if missing:
        raise RuntimeError(f"weight conversion config missing: {', '.join(missing)}")


def _remote_conversion_script(*, script_path: str, conda_env: str, input_path: str, output_path: str) -> str:
    return f"""set -euo pipefail
SCRIPT={shlex.quote(script_path)}
CONDA_ENV={shlex.quote(conda_env)}
INPUT_DIR={shlex.quote(input_path)}
OUTPUT_DIR={shlex.quote(output_path)}

test -f "$SCRIPT"
if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "conversion output already exists: $OUTPUT_DIR"
  exit 0
fi

python3 - "$SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" <<'PY'
import pathlib
import re
import sys

script_path, input_dir, output_dir = sys.argv[1:4]
path = pathlib.Path(script_path)
text = path.read_text()
text, input_count = re.subn(r"(--input-dir\\s+)(\\S+)", lambda m: m.group(1) + input_dir, text, count=1)
text, output_count = re.subn(r"(--output-dir\\s+)(\\S+)", lambda m: m.group(1) + output_dir, text, count=1)
if input_count != 1 or output_count != 1:
    raise SystemExit(f"failed to patch conversion script flags: input={{input_count}}, output={{output_count}}")
path.write_text(text)
PY

source ~/.bashrc >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
bash "$SCRIPT"
test -d "$OUTPUT_DIR"
"""


def _short(text: str, limit: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."
