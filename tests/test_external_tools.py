from pathlib import Path

from screensift.common.external import docker_prefix, run_command, shell_join, tool_command


def test_docker_command_builder_includes_gpu_mount_and_image(tmp_path: Path) -> None:
    command = tool_command(
        "gnina",
        ["--score_only", "-r", "rec.pdb", "-l", "lig.pdb"],
        mode="docker",
        image="gnina/gnina",
        gpu="all",
        mount=tmp_path,
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in command
    assert "gnina/gnina" in command
    assert command[-5:] == ["--score_only", "-r", "rec.pdb", "-l", "lig.pdb"]


def test_run_command_dry_run_writes_command(tmp_path: Path) -> None:
    result = run_command(["echo", "hello world"], tmp_path / "out.log", tmp_path / "err.log", dry_run=True)

    assert result.status == "planned"
    assert result.returncode is None
    assert "echo" in Path(result.stderr_path).read_text(encoding="utf-8")
    assert shell_join(["echo", "hello world"]) in Path(result.stderr_path).read_text(encoding="utf-8")


def test_docker_prefix_accepts_extra_args(tmp_path: Path) -> None:
    command = docker_prefix("image", mount=tmp_path, docker_args=["--ipc=host"])

    assert "--ipc=host" in command
