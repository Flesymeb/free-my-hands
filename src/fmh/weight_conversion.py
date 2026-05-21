from __future__ import annotations

import posixpath
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fmh.config import ReusableWorkersConfig, WeightConversionConfig
from fmh.reusable_workers import normalize_model_path


@dataclass(frozen=True)
class WeightConversionPlan:
    input_path: str
    output_path: str
    original_weight_path: str
    output_override: str = ""
    detected_format: str = ""
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
    *,
    output_override: str = "",
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

    detected_format = ""
    if config.format_detection_enabled:
        detected_format = detect_weight_format(input_path, config)
        if detected_format == "hf":
            return None
        if detected_format != "distcp":
            if config.format_detection_required:
                raise RuntimeError(f"unsupported or unknown weight format for conversion: {input_path} ({detected_format})")
            return None

    output_path = _conversion_output_path(
        input_path,
        output_override,
        output_prefix,
        reusable_config,
    )
    return WeightConversionPlan(
        input_path=input_path,
        output_path=output_path,
        original_weight_path=original,
        output_override=output_override.strip(),
        detected_format=detected_format,
    )


def detect_weight_format(path: str, config: WeightConversionConfig | None = None) -> str:
    local = _detect_local_format(path)
    if local != "missing":
        return local
    if config and config.remote_format_detection and config.host:
        return _detect_remote_format(path, config)
    return local


def run_weight_conversion(
    plan: WeightConversionPlan | dict[str, Any],
    config: WeightConversionConfig,
) -> WeightConversionResult:
    conversion = _coerce_plan(plan)
    _validate_config(config)
    script = _remote_conversion_script(
        script_path=config.script_path,
        conda_env=config.conda_env,
        conda_sh=config.conda_sh,
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
        output_override=str(plan.get("output_override") or ""),
        detected_format=str(plan.get("detected_format") or ""),
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


def _conversion_output_path(
    input_path: str,
    output_override: str,
    output_prefix: str,
    reusable_config: ReusableWorkersConfig | None,
) -> str:
    parent, basename = posixpath.split(input_path.rstrip("/"))
    override = str(output_override or "").strip().strip("`'\"，,")
    if not override:
        return posixpath.join(parent, output_prefix + basename)
    if override.startswith("/"):
        return normalize_model_path(override, reusable_config).worker_path if reusable_config else override
    if "/" in override:
        return posixpath.join(parent, override.strip("/"))
    return posixpath.join(parent, override)


def _detect_local_format(path: str) -> str:
    directory = Path(path)
    if not directory.is_dir():
        return "missing"
    try:
        names = {item.name for item in directory.iterdir()}
    except OSError:
        return "unknown"
    return _classify_names(names)


def _detect_remote_format(path: str, config: WeightConversionConfig) -> str:
    script = f"""set -e
DIR={shlex.quote(path)}
if [ ! -d "$DIR" ]; then
  echo missing
  exit 0
fi
has_config=0
has_hf_weight=0
has_distcp=0
has_distcp_meta=0
[ -f "$DIR/config.json" ] && has_config=1
if [ -n "$(find "$DIR" -maxdepth 1 -type f \\( -name '*.safetensors' -o -name 'model.safetensors.index.json' -o -name 'pytorch_model*.bin' \\) -print -quit 2>/dev/null)" ]; then
  has_hf_weight=1
fi
if [ -n "$(find "$DIR" -maxdepth 1 -type f -name '*.distcp' -print -quit 2>/dev/null)" ]; then
  has_distcp=1
fi
if [ -f "$DIR/common.pt" ] || [ -f "$DIR/metadata.json" ] || [ -f "$DIR/.metadata" ]; then
  has_distcp_meta=1
fi
if [ "$has_config" = 1 ] && [ "$has_hf_weight" = 1 ]; then
  echo hf
elif [ "$has_distcp" = 1 ] && [ "$has_distcp_meta" = 1 ]; then
  echo distcp
elif [ "$has_config" = 1 ]; then
  echo hf
else
  echo unknown
fi
"""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
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
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=config.detect_timeout_sec,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"weight format detection failed ({result.returncode}): {_short(detail, 500)}")
    detected = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"
    return detected if detected in {"hf", "distcp", "missing", "unknown"} else "unknown"


def _classify_names(names: set[str]) -> str:
    has_config = "config.json" in names
    has_hf_weight = any(
        name.endswith(".safetensors") or name == "model.safetensors.index.json" or name.startswith("pytorch_model")
        for name in names
    )
    has_distcp = any(name.endswith(".distcp") for name in names)
    has_distcp_meta = bool({"common.pt", "metadata.json", ".metadata"} & names)
    if has_config and has_hf_weight:
        return "hf"
    if has_distcp and has_distcp_meta:
        return "distcp"
    if has_config:
        return "hf"
    return "unknown"


def _remote_conversion_script(
    *,
    script_path: str,
    conda_env: str,
    conda_sh: str,
    input_path: str,
    output_path: str,
) -> str:
    conda_sh_command = f"source {shlex.quote(conda_sh)}" if conda_sh else ""
    return f"""set -eo pipefail
SCRIPT={shlex.quote(script_path)}
CONDA_ENV={shlex.quote(conda_env)}
INPUT_DIR={shlex.quote(input_path)}
OUTPUT_DIR={shlex.quote(output_path)}
SCRIPT_DIR=$(dirname "$SCRIPT")
TMP_SCRIPT="$SCRIPT_DIR/.fmh_$(basename "$SCRIPT").$$.sh"

test -f "$SCRIPT"
if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  if [ -f "$OUTPUT_DIR/config.json" ] && [ -n "$(find "$OUTPUT_DIR" -maxdepth 1 -type f \\( -name '*.safetensors' -o -name 'model.safetensors.index.json' -o -name 'pytorch_model*.bin' \\) -print -quit 2>/dev/null)" ]; then
    echo "conversion output already exists: $OUTPUT_DIR"
    exit 0
  fi
  echo "conversion output exists but is not a complete HF checkpoint: $OUTPUT_DIR" >&2
  exit 2
fi

cp "$SCRIPT" "$TMP_SCRIPT"
trap 'rm -f "$TMP_SCRIPT"' EXIT
python3 - "$TMP_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" <<'PY'
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
{conda_sh_command}
if ! command -v conda >/dev/null 2>&1 && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif ! command -v conda >/dev/null 2>&1 && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"
cd "$SCRIPT_DIR"
bash "$TMP_SCRIPT"
test -d "$OUTPUT_DIR"
"""


def _short(text: str, limit: int) -> str:
    compact = str(text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."
