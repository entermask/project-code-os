#!/usr/bin/env bash
set -euo pipefail

# Test-host-only screen for the bridge's production MP3+loudnorm post-process
# pool. SGLang stays on the active K16 arm; this script restarts only the
# loopback API and verifies the candidate environment in /proc.

APP_ROOT=${APP_ROOT:-/root/autodl-tmp/Fish-Audio}
CONTROL="$APP_ROOT/canary/higgs-sweetspot/runtime/higgs-test-control.sh"
RESULT_ROOT="$APP_ROOT/canary/higgs-sweetspot/results/20260730/round2/ffmpeg"
STATE_DIR=${STATE_DIR:-/root/autodl-tmp/higgs-test-control}

ARM=${1:?Usage: $0 ARM CONCURRENCY NICE REPEATS}
CONCURRENCY=${2:?Usage: $0 ARM CONCURRENCY NICE REPEATS}
NICE_VALUE=${3:?Usage: $0 ARM CONCURRENCY NICE REPEATS}
REPEATS=${4:?Usage: $0 ARM CONCURRENCY NICE REPEATS}

[[ "$ARM" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "Invalid arm label" >&2
  exit 2
}
[[ "$CONCURRENCY" =~ ^[0-9]+$ && "$CONCURRENCY" -gt 0 ]] || {
  echo "Invalid concurrency" >&2
  exit 2
}
[[ "$NICE_VALUE" =~ ^(off|0|[0-9]+)$ ]] || {
  echo "Invalid nice value" >&2
  exit 2
}
[[ "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -gt 0 ]] || {
  echo "Invalid repeat count" >&2
  exit 2
}

AF_FILTER='aresample=44100,acompressor=threshold=-18dB:ratio=3,loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=level_in=1:level_out=1:limit=0.95'
COMMON_ARGS=(
  --api-base-url http://127.0.0.1:6006
  --env-file .env
  --ref-dir /root/autodl-tmp/bench-assets
  --ref-port 18081
  --ref-profile hot
  --ref-file ref_341c9392.mp3
  --ref-text "The people who are crazy enough to think they can change the world are the ones who do."
  --audio-format mp3
  --af-filter "$AF_FILTER"
  --unique-text
  --poll-interval 0.1
  --gpu-poll-interval 0.25
  --job-timeout 900
  --http-timeout 900
  --seed 20260730
)

mkdir -p "$RESULT_ROOT/$ARM"
cd "$APP_ROOT"

"$CONTROL" stop-api
FFMPEG_POST_CONCURRENCY="$CONCURRENCY" \
FFMPEG_POST_NICE="$NICE_VALUE" \
FFMPEG_THREADS=1 \
TEST_PCM_STATS_AUDIOOP=all \
  "$CONTROL" start-api

read -r api_pid _ <"$STATE_DIR/api.pid"
for expected in \
  "HIGGS_PCM_STATS_AUDIOOP=all" \
  "FFMPEG_POST_CONCURRENCY=$CONCURRENCY" \
  "FFMPEG_POST_NICE=$NICE_VALUE" \
  "FFMPEG_THREADS=1"; do
  tr '\0' '\n' <"/proc/$api_pid/environ" | grep -Fxq -- "$expected" || {
    echo "API PID $api_pid missing expected environment: $expected" >&2
    exit 2
  }
done

.venv/bin/python scripts/bench_production_tts.py \
  "${COMMON_ARGS[@]}" \
  --label "$ARM-warm-long" \
  --jobs 4 \
  --chunks-per-job 10 \
  --text-profile long-en \
  --submit-concurrency 4 \
  --poll-concurrency 4 \
  --output "$RESULT_ROOT/$ARM/warm-long.json"

for ((run = 0; run < REPEATS; run++)); do
  .venv/bin/python scripts/bench_production_tts.py \
    "${COMMON_ARGS[@]}" \
    --label "$ARM-long-r$run" \
    --jobs 10 \
    --chunks-per-job 10 \
    --text-profile long-en \
    --submit-concurrency 10 \
    --poll-concurrency 10 \
    --output "$RESULT_ROOT/$ARM/long-r$run.json"

  .venv/bin/python scripts/bench_production_tts.py \
    "${COMMON_ARGS[@]}" \
    --label "$ARM-true-short4-r$run" \
    --jobs 24 \
    --chunks-per-job 4 \
    --text-profile short-en \
    --submit-concurrency 24 \
    --poll-concurrency 24 \
    --output "$RESULT_ROOT/$ARM/true-short4-r$run.json"
done

curl --fail --silent --show-error http://127.0.0.1:6006/health
echo
