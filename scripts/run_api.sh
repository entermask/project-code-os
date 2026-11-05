#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
APP_MODULE="${APP_MODULE:-app:app}"
# Stdlib asyncio, not uvloop: libuv spawns subprocesses with fork() and blocks the
# event loop until the child execs. The wrapper spawns ffmpeg once or twice per chunk
# from a ~1 GB process, which starved the engine in production (2026-11-05).
UVICORN_LOOP="${UVICORN_LOOP:-asyncio}"

exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --loop "$UVICORN_LOOP"
