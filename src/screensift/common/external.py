from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screensift.common.io import ensure_dir


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout_path: str
    stderr_path: str
    status: str
    error_message: str = ""


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return shlex.split(str(value))


def docker_prefix(
    image: str,
    workdir: str | Path = "/work",
    mount: str | Path = ".",
    docker_bin: str = "docker",
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
) -> list[str]:
    command = [docker_bin, "run", "--rm"]
    if gpu and str(gpu).lower() not in {"false", "none", "cpu", "0"}:
        command.extend(["--gpus", "all" if str(gpu).lower() in {"true", "gpu", "cuda"} else str(gpu)])
    command.extend(as_list(docker_args))
    host_mount = Path(mount).resolve()
    command.extend(["-v", f"{host_mount}:{workdir}", "-w", str(workdir), image])
    return command


def tool_command(
    binary: str,
    args: list[str] | None = None,
    mode: str = "native",
    image: str | None = None,
    workdir: str | Path = "/work",
    mount: str | Path = ".",
    docker_bin: str = "docker",
    gpu: str | bool | None = None,
    docker_args: list[str] | str | None = None,
) -> list[str]:
    inner = [binary] + list(args or [])
    if mode == "native":
        return inner
    if mode == "docker":
        if not image:
            raise ValueError("Docker mode requires an image.")
        return docker_prefix(image, workdir=workdir, mount=mount, docker_bin=docker_bin, gpu=gpu, docker_args=docker_args) + inner
    raise ValueError(f"Unsupported execution mode: {mode!r}. Use 'native' or 'docker'.")


def run_command(
    command: list[str],
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_seconds: int = 1800,
    dry_run: bool = False,
) -> CommandResult:
    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    ensure_dir(stdout.parent)
    ensure_dir(stderr.parent)
    if dry_run:
        stdout.write_text("", encoding="utf-8")
        stderr.write_text(shell_join(command) + "\n", encoding="utf-8")
        return CommandResult(command=command, returncode=None, stdout_path=str(stdout), stderr_path=str(stderr), status="planned")
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
        stdout.write_text(completed.stdout, encoding="utf-8")
        stderr.write_text(completed.stderr, encoding="utf-8")
        status = "complete" if completed.returncode == 0 else "failed"
        error = "" if completed.returncode == 0 else f"returncode={completed.returncode}; stderr={completed.stderr.strip()[:1000]}"
        return CommandResult(command=command, returncode=completed.returncode, stdout_path=str(stdout), stderr_path=str(stderr), status=status, error_message=error)
    except Exception as exc:
        stdout.write_text("", encoding="utf-8")
        stderr.write_text(str(exc), encoding="utf-8")
        return CommandResult(command=command, returncode=None, stdout_path=str(stdout), stderr_path=str(stderr), status="failed", error_message=str(exc))
