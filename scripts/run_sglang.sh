#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-bosonai/higgs-audio-v3-tts-4b}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_ALLOWED_LOCAL_MEDIA_PATH="${SGLANG_ALLOWED_LOCAL_MEDIA_PATH:-${TTS_CACHE_DIR:-}}"

if ! command -v sgl-omni >/dev/null 2>&1; then
  echo "sgl-omni not found. Install SGLang-Omni first, then activate that environment." >&2
  exit 1
fi

args=(
  serve
  --model-path "$MODEL_PATH"
  --host "$SGLANG_HOST"
  --port "$SGLANG_PORT"
)

if [ -n "${SGLANG_CONFIG:-}" ]; then
  args+=(--config "$SGLANG_CONFIG")
fi

if [ -n "$SGLANG_ALLOWED_LOCAL_MEDIA_PATH" ]; then
  args+=(--allowed-local-media-path "$SGLANG_ALLOWED_LOCAL_MEDIA_PATH")
fi
if [ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]; then
  args+=(--mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC")
fi
if [ -n "${SGLANG_CPU_OFFLOAD_GB:-}" ]; then
  args+=(--cpu-offload-gb "$SGLANG_CPU_OFFLOAD_GB")
fi
if [ -n "${SGLANG_EXTRA_ARGS:-}" ]; then
  read -r -a extra_args <<< "$SGLANG_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

exec sgl-omni "${args[@]}"
