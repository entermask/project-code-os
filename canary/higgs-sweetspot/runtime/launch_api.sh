#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=${APP_ROOT:?Missing APP_ROOT}
launch_token=${HIGGS_TEST_LAUNCH_TOKEN:?Missing HIGGS_TEST_LAUNCH_TOKEN}
pcm_stats_audioop=${HIGGS_TEST_PCM_STATS_AUDIOOP:?Missing HIGGS_TEST_PCM_STATS_AUDIOOP}
lane_admission_mode=${HIGGS_TEST_LANE_ADMISSION_MODE:?Missing HIGGS_TEST_LANE_ADMISSION_MODE}
ffmpeg_timing=${HIGGS_TEST_FFMPEG_TIMING:?Missing HIGGS_TEST_FFMPEG_TIMING}
max_concurrent_chunks=${HIGGS_TEST_MAX_CONCURRENT_CHUNKS:?Missing HIGGS_TEST_MAX_CONCURRENT_CHUNKS}
cd "$APP_ROOT"
set -a
. "$APP_ROOT/.env"
set +a
export HIGGS_TEST_LAUNCH_TOKEN="$launch_token"
export HIGGS_PCM_STATS_AUDIOOP="$pcm_stats_audioop"
export HIGGS_LANE_ADMISSION_MODE="$lane_admission_mode"
export HIGGS_FFMPEG_TIMING="$ffmpeg_timing"
export MAX_CONCURRENT_CHUNKS="$max_concurrent_chunks"

exec "$APP_ROOT/.venv/bin/uvicorn" app:app --host 127.0.0.1 --port 6006
