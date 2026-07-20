#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=/root/autodl-tmp/Fish-Audio
PROD_SOURCE_ROOT="$APP_ROOT/prod-current"
VENV_ROOT="$APP_ROOT/.venv"
EXPECTED_APP_SHA=328851b1d77dacf921963376fe15c16255335e68fca75863b981fb11d45cf033

actual_app_sha=$(sha256sum "$PROD_SOURCE_ROOT/app.py" | awk '{print $1}')
if [[ "$actual_app_sha" != "$EXPECTED_APP_SHA" ]]; then
  echo "prod clone app hash mismatch: $actual_app_sha" >&2
  exit 2
fi

set -a
. "$APP_ROOT/.env"
set +a
api_token=${API_TOKEN:?API_TOKEN is required}

mkdir -p /root/autodl-tmp/tts-cache/prod-compare
cd "$PROD_SOURCE_ROOT"

exec env -i \
  API_TOKEN="$api_token" \
  PATH="$VENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  PYTHONUNBUFFERED=1 \
  HOST=0.0.0.0 \
  PORT=6006 \
  MODEL_PATH=bosonai/higgs-audio-v3-tts-4b \
  TTS_BACKEND_NAME=bosonai/higgs-audio-v3-tts-4b \
  SPEECH_MODEL= \
  SGLANG_BASE_URL=http://127.0.0.1:8000 \
  TTS_CACHE_DIR=/root/autodl-tmp/tts-cache/prod-compare \
  MAX_CONCURRENT_CHUNKS=96 \
  MAX_IN_FLIGHT_CHUNKS_PER_JOB=10 \
  BUSY_BACKLOG_CHUNKS=2000 \
  SHORT_RESERVED_CHUNKS=4 \
  SHORT_REQUEST_MAX_CHARS=1000 \
  SHORT_REQUEST_MAX_CHUNKS=4 \
  DOWNLOAD_TIMEOUT=60 \
  REQUEST_TIMEOUT=600 \
  JOB_TTL_SECONDS=3600 \
  JOB_CLEANUP_INTERVAL_SECONDS=30 \
  STREAMED_JOB_TTL_SECONDS=60 \
  STREAM_CHUNK_SIZE_BYTES=4194304 \
  CHUNK_RETRY_ATTEMPTS=3 \
  CHUNK_RETRY_BASE_DELAY=1.0 \
  CHUNK_MIN_BYTES=512 \
  FFMPEG_BIN=ffmpeg \
  LOG_LEVEL=INFO \
  "$VENV_ROOT/bin/uvicorn" app:app --host 127.0.0.1 --port 16007
