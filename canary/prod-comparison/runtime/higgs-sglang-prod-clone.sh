#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/root/autodl-tmp/Fish-Audio
SOURCE_ROOT=/root/autodl-tmp/sglang-omni-prod-sim
VENV_ROOT=/root/autodl-tmp/sglang-omni
EXPECTED_HEAD=df62e91a00d383e6f73ab9604386ffac6c520529
EXPECTED_DIFF_SHA=9a3ba2d6f6b8459e631b488b76eb5a9a96432ed32edc5dcab770789dd4ef6ad4
EXPECTED_STATUS_SHA=426a175a8fbe005d450e253f8ff0a4769d30b090a0860532c3cf301896740584
MODEL_SNAPSHOT=/root/autodl-tmp/hf-cache/models--bosonai--higgs-audio-v3-tts-4b/snapshots/a7f70853f163c4cccbdd27ce9a80dd97961fc581
MODEL_PUBLIC_NAME=bosonai/higgs-audio-v3-tts-4b
EXPECTED_MODEL_LINK=../../blobs/2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a
EXPECTED_MODEL_BYTES=9309834930

if [[ $(readlink "$MODEL_SNAPSHOT/model.safetensors") != "$EXPECTED_MODEL_LINK" \
  || $(stat -Lc '%s' "$MODEL_SNAPSHOT/model.safetensors") != "$EXPECTED_MODEL_BYTES" ]]; then
  echo "pinned model snapshot mismatch" >&2
  exit 2
fi

actual_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
actual_diff_sha=$(git -C "$SOURCE_ROOT" diff HEAD --no-ext-diff | sha256sum | awk '{print $1}')
actual_status_sha=$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all | sha256sum | awk '{print $1}')
if [[ "$actual_head" != "$EXPECTED_HEAD" || "$actual_diff_sha" != "$EXPECTED_DIFF_SHA" || "$actual_status_sha" != "$EXPECTED_STATUS_SHA" ]]; then
  echo "prod clone SGLang fingerprint mismatch" >&2
  exit 2
fi

cd "$APP_ROOT"
set -a
. "$APP_ROOT/.env"
. /root/autodl-tmp/prod-sim/runtime.env
set +a

export SGLANG_EXTRA_ARGS="--stages.2.factory_args.server_args_overrides.attention_backend triton --model-name $MODEL_PUBLIC_NAME"
export MODEL_PATH="$MODEL_SNAPSHOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset PREFILL_COALESCE_REQUESTS
unset PREFILL_COALESCE_WAIT_MS
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
