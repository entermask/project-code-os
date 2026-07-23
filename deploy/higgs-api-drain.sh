#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/root/autodl-tmp/Fish-Audio
cd "$APP_ROOT"

set -a
. "$APP_ROOT/.env"
set +a

export APP_MODULE=drain_wrapper:app
export PYTHONUNBUFFERED=1

. "$APP_ROOT/.venv/bin/activate"
exec "$APP_ROOT/scripts/run_api.sh"
