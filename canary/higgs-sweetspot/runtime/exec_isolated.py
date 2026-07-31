#!/usr/bin/env python3
"""Enter a new session, publish exact process identity, then exec a command."""

from __future__ import annotations

import argparse
import os
import re
import signal
import tempfile
from pathlib import Path


def start_ticks() -> int:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    fields_after_comm = raw.rsplit(")", 1)[1].split()
    return int(fields_after_comm[19])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity_file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing command after --")

    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    launch_token = os.environ.get("HIGGS_TEST_LAUNCH_TOKEN", "")
    if re.fullmatch(r"[0-9a-f]{64}", launch_token) is None:
        raise RuntimeError("Missing or invalid HIGGS_TEST_LAUNCH_TOKEN")
    pid = os.getpid()
    pgid = os.getpgid(0)
    if pgid != pid:
        raise RuntimeError(f"Failed to isolate process group: pid={pid} pgid={pgid}")

    args.identity_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.identity_file.name}.",
        dir=args.identity_file.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{pid} {start_ticks()} {pgid} {launch_token}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, args.identity_file)
    finally:
        temporary_path.unlink(missing_ok=True)

    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
