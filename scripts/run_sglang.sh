#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fishaudio/s2-pro}"
SGLANG_OMNI_DIR="${SGLANG_OMNI_DIR:-$HOME/sglang-omni}"
SGLANG_CONFIG="${SGLANG_CONFIG:-$SGLANG_OMNI_DIR/examples/configs/s2pro_tts.yaml}"
SGLANG_PORT="${SGLANG_PORT:-8000}"

if ! command -v sgl-omni >/dev/null 2>&1; then
  echo "sgl-omni not found. Install SGLang-Omni first, then activate that environment." >&2
  exit 1
fi

exec sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config "$SGLANG_CONFIG" \
  --port "$SGLANG_PORT"

