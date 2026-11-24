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

# sgl-omni thay port bi chiem bang port ngau nhien ma KHONG bao loi -> wrapper goi
# SGLANG_PORT khong bao gio thay engine. supervisord restart higgs_sglang ngay sau
# crash, luc port con TIME_WAIT (~60s) -> 31/10-01/11 hai box chet lam sang 8-12h.
# Cho toi khi bind kieu vanilla duoc (dung kieu kiem tra cua sgl-omni); het han
# thi thoat loi de supervisord thu lai, khong bao gio chay tren port khac.
port_wait_seconds="${SGLANG_PORT_WAIT_SECONDS:-180}"
port_deadline=$((SECONDS + port_wait_seconds))
until python3 -c 'import socket, sys; s = socket.socket(); s.bind((sys.argv[1], int(sys.argv[2]))); s.close()' \
    "$SGLANG_HOST" "$SGLANG_PORT" 2>/dev/null; do
  if (( SECONDS >= port_deadline )); then
    echo "port $SGLANG_HOST:$SGLANG_PORT still busy after ${port_wait_seconds}s; refusing to start sgl-omni on a random port" >&2
    exit 1
  fi
  sleep 2
done

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
