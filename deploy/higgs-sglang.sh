#!/bin/bash
set -euo pipefail

cd /workspace/Fish-Audio

set -a
. /workspace/Fish-Audio/.env
set +a

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTHONUNBUFFERED=1

. /workspace/sglang-omni/.venv/bin/activate
exec /workspace/Fish-Audio/scripts/run_sglang.sh
