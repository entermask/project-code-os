from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sglang.sh"


def _fake_env(tmp_path: Path, port: int, wait_seconds: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "sgl-omni-args"
    fake = bin_dir / "sgl-omni"
    fake.write_text(f'#!/usr/bin/env bash\necho "$@" > "{marker}"\n')
    fake.chmod(0o755)
    python = bin_dir / "python3"
    python.symlink_to(sys.executable)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_")}
    env.update(
        PATH=f"{bin_dir}:{env.get('PATH', '')}",
        MODEL_PATH="test-model",
        SGLANG_HOST="127.0.0.1",
        SGLANG_PORT=str(port),
        SGLANG_PORT_WAIT_SECONDS=wait_seconds,
    )
    return env


def test_refuses_to_start_on_random_port_while_port_busy(tmp_path: Path) -> None:
    """31/10-01/11/2026: restart lúc port còn bận → sgl-omni lặng lẽ nhảy port, wrapper mất engine."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=_fake_env(tmp_path, port, "1"),
            capture_output=True, text=True, timeout=30,
        )
    finally:
        holder.close()
    assert result.returncode == 1
    assert "still busy" in result.stderr
    assert not (tmp_path / "sgl-omni-args").exists()


def test_starts_sgl_omni_on_configured_port_once_free(tmp_path: Path) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=_fake_env(tmp_path, port, "5"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "sgl-omni-args").read_text()
    assert f"--port {port}" in args
