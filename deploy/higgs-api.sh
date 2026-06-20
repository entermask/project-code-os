#!/bin/bash
set -euo pipefail

cd /workspace/Fish-Audio

set -a
. /workspace/Fish-Audio/.env
set +a

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8080}"
export PYTHONUNBUFFERED=1

. /workspace/Fish-Audio/.venv/bin/activate
exec /workspace/Fish-Audio/scripts/run_api.sh
