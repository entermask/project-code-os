#!/usr/bin/env python3
"""Terminate only processes that still carry one controller launch token."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import re
import signal
import time
from pathlib import Path


if platform.machine() != "x86_64":
    raise RuntimeError("pidfd syscall wrapper is pinned to x86_64")

_LIBC = ctypes.CDLL(None, use_errno=True)
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434


def pidfd_open(pid: int) -> int:
    fd = _LIBC.syscall(
        ctypes.c_long(_SYS_PIDFD_OPEN),
        ctypes.c_int(pid),
        ctypes.c_uint(0),
    )
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def pidfd_send_signal(pidfd: int, signum: int) -> None:
    result = _LIBC.syscall(
        ctypes.c_long(_SYS_PIDFD_SEND_SIGNAL),
        ctypes.c_int(pidfd),
        ctypes.c_int(signum),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def matching_pidfds(token: str, pgid: int) -> list[tuple[int, int]]:
    expected = f"HIGGS_TEST_LAUNCH_TOKEN={token}".encode()
    matches: list[tuple[int, int]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            pidfd = pidfd_open(pid)
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ESRCH}:
                continue
            raise
        try:
            if os.getpgid(pid) != pgid:
                os.close(pidfd)
                continue
            environ = (entry / "environ").read_bytes().split(b"\0")
            if expected not in environ:
                os.close(pidfd)
                continue
        except OSError as exc:
            os.close(pidfd)
            if exc.errno in {errno.ENOENT, errno.ESRCH}:
                continue
            raise
        matches.append((pid, pidfd))
    return matches


def signal_matches(token: str, pgid: int, signum: int) -> int:
    matches = matching_pidfds(token, pgid)
    try:
        for _, pidfd in matches:
            try:
                pidfd_send_signal(pidfd, signum)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise
    finally:
        for _, pidfd in matches:
            os.close(pidfd)
    return len(matches)


def live_count(token: str, pgid: int) -> int:
    matches = matching_pidfds(token, pgid)
    for _, pidfd in matches:
        os.close(pidfd)
    return len(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--token")
    parser.add_argument("--pgid", type=int)
    parser.add_argument("--label")
    parser.add_argument("--term-timeout", type=float, default=90.0)
    parser.add_argument("--kill-timeout", type=float, default=3.0)
    args = parser.parse_args()
    if args.probe:
        fd = pidfd_open(os.getpid())
        try:
            pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
        print("pidfd syscalls available")
        return
    if args.token is None or args.pgid is None or args.label is None:
        parser.error("--token, --pgid, and --label are required unless --probe is used")
    if re.fullmatch(r"[0-9a-f]{64}", args.token) is None:
        raise ValueError("Invalid launch token")
    if args.pgid <= 1:
        raise ValueError("Invalid process group")

    deadline = time.monotonic() + args.term_timeout
    while True:
        count = signal_matches(args.token, args.pgid, signal.SIGTERM)
        if count == 0:
            print(f"{args.label}: launch token has no live processes")
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)

    deadline = time.monotonic() + args.kill_timeout
    while True:
        count = signal_matches(args.token, args.pgid, signal.SIGKILL)
        if count == 0:
            print(f"{args.label}: launch token terminated after SIGKILL")
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    remaining = live_count(args.token, args.pgid)
    raise RuntimeError(
        f"{args.label}: {remaining} launch-token process(es) survived SIGKILL"
    )


if __name__ == "__main__":
    main()
