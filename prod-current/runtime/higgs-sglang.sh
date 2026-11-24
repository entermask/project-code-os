#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/root/autodl-tmp/Fish-Audio
SOURCE_ROOT=${SGLANG_SOURCE_ROOT:-/root/autodl-tmp/sglang-omni-k16}
VENV_ROOT=/root/autodl-tmp/sglang-omni
MODEL_SNAPSHOT=${HIGGS_MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/snapshots/a7f70853f163c4cccbdd27ce9a80dd97961fc581}
MODEL_PUBLIC_NAME=bosonai/higgs-audio-v3-tts-4b
EXPECTED_MODEL_LINK=../../blobs/2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a
EXPECTED_MODEL_BYTES=9309834930

if [[ ! -d "$SOURCE_ROOT/sglang_omni" ]]; then
  echo "candidate source root is missing: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$(readlink "$MODEL_SNAPSHOT/model.safetensors")" != "$EXPECTED_MODEL_LINK" ]]; then
  echo "pinned model symlink mismatch: $MODEL_SNAPSHOT" >&2
  exit 2
fi
if [[ "$(stat -Lc '%s' "$MODEL_SNAPSHOT/model.safetensors")" != "$EXPECTED_MODEL_BYTES" ]]; then
  echo "pinned model size mismatch: $MODEL_SNAPSHOT" >&2
  exit 2
fi

cd "$APP_ROOT"
set -a
. "$APP_ROOT/.env"
set +a

export MODEL_PATH="$MODEL_SNAPSHOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-} --model-name $MODEL_PUBLIC_NAME --stages.2.factory_args.prefill_coalesce_requests 16 --stages.2.factory_args.prefill_coalesce_wait_ms 60"

# Worker-side SRT owns the current 60 ms tail fade. Keep every experimental
# vocoder fade policy disabled in production.
unset HIGGS_VOCODER_CANARY_POLICY
unset HIGGS_VOCODER_CANARY_DUMP_DIR
unset HIGGS_VOCODER_CANARY_FADE_OUT_MS
unset HIGGS_VOCODER_CANARY_FADE_CURVE

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-0}"

ulimit -n 65534

. "$VENV_ROOT/.venv/bin/activate"
exec "$APP_ROOT/scripts/run_sglang.sh"
