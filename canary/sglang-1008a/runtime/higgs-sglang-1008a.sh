#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/root/autodl-tmp/Fish-Audio
SOURCE_ROOT=/root/autodl-tmp/sglang-omni-1008a-canary
VENV_ROOT=/root/autodl-tmp/sglang-omni
cd "$APP_ROOT"

set -a
. "$APP_ROOT/.env"
. /root/autodl-tmp/prod-sim/runtime.env
set +a

unset HIGGS_VOCODER_CANARY_POLICY
unset HIGGS_VOCODER_CANARY_DUMP_DIR
unset HIGGS_VOCODER_CANARY_FADE_OUT_MS
unset HIGGS_VOCODER_CANARY_FADE_CURVE

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$VENV_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
CCCL_INCLUDE="$VENV_ROOT/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export CPATH="$CCCL_INCLUDE${CPATH:+:$CPATH}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/hf-cache}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-0}"

ulimit -n 65534

. "$VENV_ROOT/.venv/bin/activate"
exec "$APP_ROOT/scripts/run_sglang.sh"
