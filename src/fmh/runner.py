from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path

from fmh.config import RunnerConfig
from fmh.time_utils import utc_now_iso


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    log_path: str = ""


class RunnerError(RuntimeError):
    pass


class BaseRunner:
    def ensure_session(self, session_name: str, workdir: str) -> CommandResult:
        raise NotImplementedError

    def run(self, session_name: str, window_name: str, command: str) -> CommandResult:
        raise NotImplementedError


class DryRunRunner(BaseRunner):
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def ensure_session(self, session_name: str, workdir: str) -> CommandResult:
        return self._record(session_name, "session", f"tmux new-session -d -s {session_name} -c {workdir}")

    def run(self, session_name: str, window_name: str, command: str) -> CommandResult:
        return self._record(session_name, window_name, command)

    def _record(self, session_name: str, window_name: str, command: str) -> CommandResult:
        timestamp = utc_now_iso().replace(":", "").replace("+", "Z")
        safe_session = _safe_name(session_name)
        safe_window = _safe_name(window_name)
        path = self.log_dir / f"{timestamp}-{safe_session}-{safe_window}.log"
        path.write_text(f"[dry-run]\nwindow={window_name}\ncommand={command}\n", encoding="utf-8")
        return CommandResult(command=command, returncode=0, stdout="dry-run", log_path=str(path))


class TmuxRunner(BaseRunner):
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def ensure_session(self, session_name: str, workdir: str) -> CommandResult:
        has_session = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            text=True,
            capture_output=True,
            check=False,
        )
        if has_session.returncode == 0:
            return CommandResult(
                command=f"tmux has-session -t {session_name}",
                returncode=0,
                stdout="session exists",
            )

        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "-c", workdir],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or "failed to create tmux session")
        return CommandResult(
            command=f"tmux new-session -d -s {session_name} -c {workdir}",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def run(self, session_name: str, window_name: str, command: str) -> CommandResult:
        self._ensure_window(session_name, window_name)
        wrapped_command = self._wrap_command(session_name, window_name, command)
        result = subprocess.run(
            ["tmux", "send-keys", "-t", f"{session_name}:{window_name}", wrapped_command, "C-m"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RunnerError(result.stderr.strip() or f"failed to send command to {window_name}")
        return CommandResult(
            command=command,
            returncode=0,
            stdout=result.stdout,
            stderr=result.stderr,
            log_path=self._log_path(session_name, window_name),
        )

    def _ensure_window(self, session_name: str, window_name: str) -> None:
        list_windows = subprocess.run(
            ["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if list_windows.returncode != 0:
            raise RunnerError(list_windows.stderr.strip() or "failed to list tmux windows")
        if window_name in set(list_windows.stdout.splitlines()):
            return
        new_window = subprocess.run(
            ["tmux", "new-window", "-t", session_name, "-n", window_name],
            text=True,
            capture_output=True,
            check=False,
        )
        if new_window.returncode != 0:
            raise RunnerError(new_window.stderr.strip() or f"failed to create window {window_name}")

    def _wrap_command(self, session_name: str, window_name: str, command: str) -> str:
        log_path = self._log_path(session_name, window_name)
        return f"({command}) 2>&1 | tee -a {shlex.quote(log_path)}"

    def _log_path(self, session_name: str, window_name: str) -> str:
        safe_session = _safe_name(session_name)
        safe_window = _safe_name(window_name)
        return str(self.log_dir / f"{safe_session}-{safe_window}.log")


def make_runner(config: RunnerConfig) -> BaseRunner:
    if config.mode == "dry-run":
        return DryRunRunner(config.log_dir)
    if config.mode == "tmux":
        return TmuxRunner(config.log_dir)
    raise ValueError(f"unsupported runner mode: {config.mode}")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)
