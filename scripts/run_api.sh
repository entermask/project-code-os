#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
APP_MODULE="${APP_MODULE:-app:app}"

exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"

