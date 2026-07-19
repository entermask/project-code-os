#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Fish-Audio
set -a
. /root/autodl-tmp/Fish-Audio/.env
set +a
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.hf_home}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-0}"
. /root/autodl-tmp/sglang-omni/.venv/bin/activate
exec /root/autodl-tmp/Fish-Audio/scripts/run_sglang.sh
