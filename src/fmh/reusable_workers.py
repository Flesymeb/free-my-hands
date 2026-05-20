from __future__ import annotations

import html
import re
import shlex
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Iterable

from fmh.config import ReusableWorkersConfig


@dataclass(frozen=True)
class ModelPathInfo:
    original_path: str
    worker_path: str
    table_path: str
    model_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DeployedModelRow:
    row_index: int
    model: str
    model_id: str
    address: str
    tool_parser: str
    reasoning_parser: str
    ssh_forward_command: str
    tested_tasks: str
    no_proxy_command: str

    @property
    def ip(self) -> str:
        match = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", self.address)
        return match.group(0) if match else ""

    @property
    def gpu_count(self) -> int:
        match = re.search(r"(\d+)\s*卡", self.address)
        return int(match.group(1)) if match else 0

    def has_running_marker(self, marker: str) -> bool:
        return tested_tasks_has_running_marker(self.tested_tasks, marker)

    def has_finished_tasks(self, required_tasks: list[str]) -> bool:
        return tested_tasks_has_finished_tasks(self.tested_tasks, required_tasks)

    def is_idle_empty(self) -> bool:
        return not self.model.strip() and not self.model_id.strip() and not self.tested_tasks.strip()

    def is_fresh_untested(self) -> bool:
        return bool(self.model.strip() or self.model_id.strip()) and not self.tested_tasks.strip()

    def is_reusable(self, config: ReusableWorkersConfig) -> bool:
        return is_reusable_worker_state(self.model, self.model_id, self.tested_tasks, config)

    def to_dict(self, config: ReusableWorkersConfig | None = None) -> dict[str, Any]:
        data = asdict(self)
        data["ip"] = self.ip
        data["gpu_count"] = self.gpu_count
        if config is not None:
            data["reusable"] = self.is_reusable(config)
        return data


@dataclass(frozen=True)
class ReusableDeploymentPlan:
    path: ModelPathInfo
    row: DeployedModelRow
    ssh_command: str
    tmux_session_guess: str
    tmux_attach_command: str
    tmux_test_window: str
    tmux_session_patterns: list[str]
    stop_hint: str
    vllm_command: str
    health_check_command: str
    final_table_values: dict[str, str]
    deploying_table_values: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "row": self.row.to_dict(),
            "ssh_command": self.ssh_command,
            "tmux_session_guess": self.tmux_session_guess,
            "tmux_attach_command": self.tmux_attach_command,
            "tmux_test_window": self.tmux_test_window,
            "tmux_session_patterns": self.tmux_session_patterns,
            "stop_hint": self.stop_hint,
            "vllm_command": self.vllm_command,
            "health_check_command": self.health_check_command,
            "deploying_table_values": self.deploying_table_values,
            "final_table_values": self.final_table_values,
        }


@dataclass(frozen=True)
class WorkerReconnectPlan:
    row: DeployedModelRow
    tmux_session: str
    windows: list[str]
    history_ssh_command: str
    reconnect_command: str
    reconnect_in_tmux_command: str
    keepalive_options: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row.to_dict(),
            "tmux_session": self.tmux_session,
            "windows": self.windows,
            "history_ssh_command": self.history_ssh_command,
            "reconnect_command": self.reconnect_command,
            "reconnect_in_tmux_command": self.reconnect_in_tmux_command,
            "keepalive_options": self.keepalive_options,
        }


@dataclass(frozen=True)
class NewWorkerRowPlan:
    path: ModelPathInfo
    ip: str
    gpu_count: int
    tmux_session: str
    rlaunch_command: str
    ssh_worker_command: str
    row_values: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "ip": self.ip,
            "gpu_count": self.gpu_count,
            "tmux_session": self.tmux_session,
            "rlaunch_command": self.rlaunch_command,
            "ssh_worker_command": self.ssh_worker_command,
            "row_values": self.row_values,
        }


def parse_deployed_models_table(markdown: str) -> list[DeployedModelRow]:
    table = _extract_deployed_models_html_table(markdown)
    if not table:
        return []
    rows = _TableParser.parse(table)
    if not rows:
        return []
    header = [_clean_cell(cell) for cell in rows[0]]
    out: list[DeployedModelRow] = []
    for index, cells in enumerate(rows[1:], start=1):
        row = _row_dict(header, cells)
        if not any(row.values()):
            continue
        out.append(
            DeployedModelRow(
                row_index=index,
                model=row.get("模型", ""),
                model_id=row.get("模型id", ""),
                address=row.get("地址", ""),
                tool_parser=row.get("推理工具调用解析器", ""),
                reasoning_parser=row.get("推理解析器", ""),
                ssh_forward_command=row.get("SSH转发命令", ""),
                tested_tasks=row.get("已经测试完的任务", ""),
                no_proxy_command=row.get("vpn排除命令", ""),
            )
        )
    return out


def choose_reusable_row(
    rows: list[DeployedModelRow],
    config: ReusableWorkersConfig,
    *,
    required_gpu_count: int = 0,
    excluded_row_indices: Iterable[int] | None = None,
) -> DeployedModelRow | None:
    excluded = {int(index) for index in (excluded_row_indices or []) if int(index) > 0}
    candidates = [row for row in rows if row.ip and row.gpu_count and row.is_reusable(config)]
    if excluded:
        candidates = [row for row in candidates if row.row_index not in excluded]
    if required_gpu_count > 0:
        candidates = [row for row in candidates if row.gpu_count >= required_gpu_count]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (not row.is_idle_empty(), row.gpu_count or 999, row.row_index))[0]


def normalize_tested_tasks(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", text)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tested_tasks_has_running_marker(tested_tasks: str, marker: str) -> bool:
    text = normalize_tested_tasks(tested_tasks)
    marker_text = normalize_tested_tasks(marker)
    compact_text = re.sub(r"\s+", "", text).lower()
    compact_marker = re.sub(r"\s+", "", marker_text).lower()
    if compact_marker and compact_marker in compact_text:
        return True
    return bool(re.search(r"\(\s*running\s*\)", text, flags=re.IGNORECASE))


def tested_tasks_has_finished_tasks(tested_tasks: str, required_tasks: list[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_.-]*", normalize_tested_tasks(tested_tasks).lower()))
    return all(str(task).strip().lower() in tokens for task in required_tasks if str(task).strip())


def is_reusable_worker_state(
    model: str,
    model_id: str,
    tested_tasks: str,
    config: ReusableWorkersConfig,
) -> bool:
    model_text = str(model or "").strip()
    model_id_text = str(model_id or "").strip()
    tested_text = str(tested_tasks or "").strip()
    if tested_tasks_has_running_marker(tested_text, config.running_marker):
        return False
    has_model = bool(model_text or model_id_text)
    if not has_model and not tested_text:
        return True
    if has_model and not tested_text:
        return False
    return tested_tasks_has_finished_tasks(tested_text, config.required_finished_tasks)


def normalize_model_path(raw_path: str, config: ReusableWorkersConfig) -> ModelPathInfo:
    original = raw_path.strip().strip("`'\"，,")
    worker_path = original
    source_prefix = config.source_model_prefix.rstrip("/")
    worker_prefix = config.worker_model_prefix.rstrip("/")
    table_prefix = config.table_model_prefix.rstrip("/")

    if original.startswith(source_prefix + "/"):
        worker_path = worker_prefix + original[len(source_prefix) :]
    elif original.startswith(worker_prefix + "/"):
        worker_path = original
    elif original.startswith("/"):
        worker_path = original
    else:
        worker_path = worker_prefix + "/" + original.lstrip("/")

    if worker_path.startswith(table_prefix + "/"):
        table_path = worker_path[len(table_prefix) + 1 :]
    else:
        table_path = worker_path.lstrip("/")
    model_id = table_path.rstrip("/").rsplit("/", 1)[-1] or "model"
    return ModelPathInfo(
        original_path=original,
        worker_path=worker_path,
        table_path=table_path,
        model_id=model_id,
    )


def build_reusable_deployment_plan(
    markdown: str,
    raw_model_path: str,
    config: ReusableWorkersConfig,
    *,
    required_gpu_count: int = 0,
    excluded_row_indices: Iterable[int] | None = None,
) -> ReusableDeploymentPlan:
    rows = parse_deployed_models_table(markdown)
    row = choose_reusable_row(
        rows,
        config,
        required_gpu_count=required_gpu_count,
        excluded_row_indices=excluded_row_indices,
    )
    if row is None:
        raise ValueError("no reusable deployed-model row is available")
    path = normalize_model_path(raw_model_path, config)
    return build_plan_for_row(path, row, config)


def build_plan_for_row(
    path: ModelPathInfo,
    row: DeployedModelRow,
    config: ReusableWorkersConfig,
) -> ReusableDeploymentPlan:
    if not row.ip:
        raise ValueError(f"selected row has no IP address: row {row.row_index}")
    tool_parser = row.tool_parser or config.default_tool_parser
    reasoning_parser = row.reasoning_parser or config.default_reasoning_parser
    gpu_count = row.gpu_count or 1
    ssh_forward = _ssh_forward_command(row.ip, config)
    no_proxy = _no_proxy_command(row.ip)
    vllm = _vllm_command(path, gpu_count, tool_parser, reasoning_parser, config)
    tmux_session = _tmux_session_guess(row.ip, gpu_count)
    return ReusableDeploymentPlan(
        path=path,
        row=row,
        ssh_command=f"ssh -CAXY {config.dev_host}",
        tmux_session_guess=tmux_session,
        tmux_attach_command=f"tmux attach -t {tmux_session}",
        tmux_test_window=f"{tmux_session}:test-model",
        tmux_session_patterns=_tmux_patterns_for_ip(row.ip),
        stop_hint=(
            (
                f"Attach {tmux_session}, verify the idle worker has no active vLLM on port {config.default_port}, "
                "run vllm_command, then use the test-model window for health/model checks."
            )
            if row.is_idle_empty()
            else (
                f"Attach {tmux_session}, stop the existing vLLM process in the ssh window with Ctrl-C, "
                "run vllm_command, then use the test-model window for health/model checks."
            )
        ),
        vllm_command=vllm,
        health_check_command=f"curl -s http://{row.ip}:{config.default_port}/v1/models",
        deploying_table_values={
            "模型": f"{path.table_path}{config.deploying_marker}",
            "模型id": f"{path.model_id}{config.deploying_marker}",
            "已经测试完的任务": "",
        },
        final_table_values={
            "模型": path.table_path,
            "模型id": path.model_id,
            "地址": row.address,
            "推理工具调用解析器": tool_parser,
            "推理解析器": reasoning_parser,
            "SSH转发命令": ssh_forward,
            "已经测试完的任务": "",
            "vpn排除命令": no_proxy,
        },
    )


def build_reconnect_plan(
    row: DeployedModelRow,
    config: ReusableWorkersConfig,
    *,
    windows: list[str] | None = None,
    pane_history: str = "",
) -> WorkerReconnectPlan:
    gpu_count = row.gpu_count or 1
    session = _tmux_session_guess(row.ip, gpu_count)
    history_command = _find_ssh_command(pane_history, row.ip)
    reconnect = _with_keepalive(history_command or f"ssh {row.ip}")
    return WorkerReconnectPlan(
        row=row,
        tmux_session=session,
        windows=windows or [],
        history_ssh_command=history_command,
        reconnect_command=reconnect,
        reconnect_in_tmux_command=f"tmux send-keys -t {session}:ssh {shlex.quote(reconnect)} C-m",
        keepalive_options="-o ServerAliveInterval=60 -o ServerAliveCountMax=3",
    )


def build_new_worker_row_plan(
    raw_model_path: str,
    ip: str,
    gpu_count: int,
    config: ReusableWorkersConfig,
    *,
    rlaunch_command: str = "",
) -> NewWorkerRowPlan:
    path = normalize_model_path(raw_model_path, config)
    session = _tmux_session_guess(ip, gpu_count)
    tool_parser = config.default_tool_parser
    reasoning_parser = config.default_reasoning_parser
    address = f"{ip}（{gpu_count}卡）"
    row = DeployedModelRow(
        row_index=0,
        model=path.table_path,
        model_id=path.model_id,
        address=address,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        ssh_forward_command=_ssh_forward_command(ip, config),
        tested_tasks="",
        no_proxy_command=_no_proxy_command(ip),
    )
    return NewWorkerRowPlan(
        path=path,
        ip=ip,
        gpu_count=gpu_count,
        tmux_session=session,
        rlaunch_command=rlaunch_command,
        ssh_worker_command=_with_keepalive(f"ssh {ip}"),
        row_values=build_plan_for_row(path, row, config).final_table_values,
    )


def _extract_deployed_models_html_table(markdown: str) -> str:
    tables = re.findall(r"<table\b.*?</table>", markdown, flags=re.IGNORECASE | re.DOTALL)
    for table in tables:
        rows = _TableParser.parse(table)
        if not rows:
            continue
        header = {_clean_cell(cell) for cell in rows[0]}
        if {"模型", "模型id", "地址", "已经测试完的任务"}.issubset(header):
            return table
    return ""


def _row_dict(header: list[str], cells: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for index, name in enumerate(header):
        out[name] = _clean_cell(cells[index]) if index < len(cells) else ""
    return out


def _clean_cell(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", text)
    text = text.replace("\\&#34;", '"').replace("&#34;", '"')
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _tmux_patterns_for_ip(ip: str) -> list[str]:
    parts = ip.split(".")
    if len(parts) != 4:
        return [ip]
    return [
        f"{parts[2]}.{parts[3]}",
        f"{parts[2]}_{parts[3]}",
        f"{parts[2]}-{parts[3]}",
        parts[3],
    ]


def _tmux_session_guess(ip: str, gpu_count: int) -> str:
    parts = ip.split(".")
    if len(parts) != 4:
        return f"ssh_{gpu_count}_gpu"
    return f"ssh_{gpu_count}_gpu_{parts[2]}_{parts[3]}"


def _ssh_forward_command(ip: str, config: ReusableWorkersConfig) -> str:
    port = config.default_port
    return (
        "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o UpdateHostKeys=no "
        f"-L {port}:{ip}:{port} {config.dev_host} -N"
    )


def _no_proxy_command(ip: str) -> str:
    return "\n".join(
        [
            f'export no_proxy="${{no_proxy:+$no_proxy,}}{ip}"',
            f'export NO_PROXY="${{NO_PROXY:+$NO_PROXY,}}{ip}"',
        ]
    )


def _vllm_command(
    path: ModelPathInfo,
    gpu_count: int,
    tool_parser: str,
    reasoning_parser: str,
    config: ReusableWorkersConfig,
) -> str:
    args = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--served-model-name",
        path.model_id,
        "--model",
        path.worker_path,
        "--data-parallel-size",
        str(gpu_count),
        "--api-server-count",
        str(config.default_api_server_count),
        "--gpu-memory-utilization",
        f"{config.default_gpu_memory_utilization:.2f}",
        "--max-model-len",
        str(config.default_max_model_len),
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 0}',
        "--host",
        "0.0.0.0",
        "--port",
        str(config.default_port),
        "--reasoning-parser",
        reasoning_parser,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_parser,
    ]
    return shlex.join(args)


def _find_ssh_command(history: str, ip: str) -> str:
    candidates: list[str] = []
    for raw in history.splitlines():
        line = raw.strip()
        if not line or "ssh" not in line:
            continue
        if line.startswith(("$", "#")):
            line = line[1:].strip()
        if re.search(r"(^|\s)ssh(\s|$)", line):
            candidates.append(line)
    for command in reversed(candidates):
        if ip and ip in command:
            return command
    return candidates[-1] if candidates else ""


def _with_keepalive(command: str) -> str:
    parts = shlex.split(command)
    if not parts:
        return command
    if parts[0] != "ssh":
        parts.insert(0, "ssh")
    if "ServerAliveInterval" not in command:
        parts[1:1] = ["-o", "ServerAliveInterval=60"]
    if "ServerAliveCountMax" not in command:
        insert_at = 3 if len(parts) >= 3 and parts[1:3] == ["-o", "ServerAliveInterval=60"] else 1
        parts[insert_at:insert_at] = ["-o", "ServerAliveCountMax=3"]
    if "UpdateHostKeys" not in command:
        insert_at = 1
        parts[insert_at:insert_at] = ["-o", "UpdateHostKeys=no"]
    return shlex.join(parts)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    @classmethod
    def parse(cls, html_table: str) -> list[list[str]]:
        parser = cls()
        parser.feed(html_table)
        return parser.rows

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"}:
            self._current_cell = []
        elif tag.lower() == "br" and self._current_cell is not None:
            self._current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append("".join(self._current_cell))
            self._current_cell = None
        elif tag.lower() == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(f"&#{name};")
