#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Fish-Audio
set -a
. /root/autodl-tmp/Fish-Audio/.env
set +a
export PYTHONUNBUFFERED=1
. /root/autodl-tmp/Fish-Audio/.venv/bin/activate
exec /root/autodl-tmp/Fish-Audio/scripts/run_api.sh
