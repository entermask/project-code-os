from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_api.sh"


def _uvicorn_args(tmp_path: Path, **env_overrides: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "uvicorn-args"
    fake = bin_dir / "uvicorn"
    fake.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{marker}"\n')
    fake.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HOST", "PORT", "APP_MODULE") and not k.startswith("UVICORN_")
    }
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env.update(env_overrides)
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, timeout=10)
    return marker.read_text().splitlines()


def test_defaults_to_stdlib_asyncio_loop(tmp_path: Path) -> None:
    args = _uvicorn_args(tmp_path, APP_MODULE="drain_wrapper:app", PORT="6006")

    assert args == [
        "drain_wrapper:app",
        "--host",
        "0.0.0.0",
        "--port",
        "6006",
        "--loop",
        "asyncio",
    ]


def test_uvicorn_loop_env_overrides_default(tmp_path: Path) -> None:
    args = _uvicorn_args(tmp_path, UVICORN_LOOP="uvloop")

    assert args[-2:] == ["--loop", "uvloop"]
