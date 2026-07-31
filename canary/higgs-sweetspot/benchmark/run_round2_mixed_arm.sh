#!/usr/bin/env bash
set -euo pipefail

# Test-host-only synchronized mixed-load benchmark. The active SGLang engine is
# held constant; only the loopback bridge is restarted to select lane admission.
#
# Usage: run_round2_mixed_arm.sh ARM MODE REPEATS [START_RUN]
# START_RUN may also be supplied through the environment and defaults to zero.
# MIXED_TOP_K defaults to one so paired arms use deterministic greedy sampling.
# Set MIXED_TOP_K=off (or an explicitly empty value) to omit --top-k and use
# the API's production sampling configuration.

APP_ROOT=${APP_ROOT:-/root/autodl-tmp/Fish-Audio}
CONTROL="$APP_ROOT/canary/higgs-sweetspot/runtime/higgs-test-control.sh"
RESULT_ROOT="$APP_ROOT/canary/higgs-sweetspot/results/20260730/round2/admission"
STATE_DIR=${STATE_DIR:-/root/autodl-tmp/higgs-test-control}

ARM=${1:?Usage: $0 ARM MODE REPEATS [START_RUN]}
MODE=${2:?Usage: $0 ARM MODE REPEATS [START_RUN]}
REPEATS=${3:?Usage: $0 ARM MODE REPEATS [START_RUN]}
START_RUN=${4:-${START_RUN:-0}}
MIXED_TOP_K=${MIXED_TOP_K-1}

[[ "$#" -le 4 ]] || {
  echo "Usage: $0 ARM MODE REPEATS [START_RUN]" >&2
  exit 2
}

[[ "$ARM" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "Invalid arm label" >&2
  exit 2
}
[[ "$MODE" =~ ^(dual|soft_reserved)$ ]] || {
  echo "Invalid lane admission mode" >&2
  exit 2
}
[[ "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -gt 0 ]] || {
  echo "Invalid repeat count" >&2
  exit 2
}
[[ "$START_RUN" =~ ^[0-9]+$ ]] || {
  echo "Invalid start run" >&2
  exit 2
}

# Force base-10 arithmetic so values such as START_RUN=08 remain valid.
REPEATS=$((10#$REPEATS))
START_RUN=$((10#$START_RUN))
last_run=$((START_RUN + REPEATS - 1))
((last_run <= 9999999)) || {
  echo "Run index exceeds the 8-character marker range" >&2
  exit 2
}

INCLUDE_TOP_K=0
if [[ -n "$MIXED_TOP_K" && "$MIXED_TOP_K" != "off" ]]; then
  [[ "$MIXED_TOP_K" =~ ^[0-9]+$ && "$MIXED_TOP_K" -gt 0 ]] || {
    echo "Invalid MIXED_TOP_K" >&2
    exit 2
  }
  MIXED_TOP_K=$((10#$MIXED_TOP_K))
  INCLUDE_TOP_K=1
fi

COMMON_ARGS=(
  --api-base-url http://127.0.0.1:6006
  --env-file .env
  --ref-dir /root/autodl-tmp/bench-assets
  --ref-profile hot
  --ref-file ref_341c9392.mp3
  --ref-text "The people who are crazy enough to think they can change the world are the ones who do."
  --audio-format wav
  --unique-text
  --poll-interval 0.05
  --gpu-poll-interval 0.25
  --job-timeout 900
  --http-timeout 900
  --seed 20260730
)
if ((INCLUDE_TOP_K)); then
  COMMON_ARGS+=(--top-k "$MIXED_TOP_K")
fi

mkdir -p "$RESULT_ROOT/$ARM"
cd "$APP_ROOT"

"$CONTROL" stop-api
FFMPEG_POST_CONCURRENCY=10 \
FFMPEG_POST_NICE=19 \
FFMPEG_THREADS=1 \
TEST_PCM_STATS_AUDIOOP=all \
TEST_LANE_ADMISSION_MODE="$MODE" \
TEST_FFMPEG_TIMING=0 \
  "$CONTROL" start-api

read -r api_pid _ <"$STATE_DIR/api.pid"
for expected in \
  "HIGGS_LANE_ADMISSION_MODE=$MODE" \
  "HIGGS_PCM_STATS_AUDIOOP=all" \
  "FFMPEG_POST_CONCURRENCY=10" \
  "FFMPEG_POST_NICE=19" \
  "FFMPEG_THREADS=1"; do
  tr '\0' '\n' <"/proc/$api_pid/environ" | grep -Fxq -- "$expected" || {
    echo "API PID $api_pid missing expected environment: $expected" >&2
    exit 2
  }
done

.venv/bin/python scripts/bench_production_tts.py \
  "${COMMON_ARGS[@]}" \
  --label "$ARM-warm-long" \
  --run-marker WARM000L \
  --ref-port 18081 \
  --jobs 4 \
  --chunks-per-job 10 \
  --text-profile long-en \
  --submit-concurrency 4 \
  --poll-concurrency 4 \
  --output "$RESULT_ROOT/$ARM/warm-long.json"

.venv/bin/python scripts/bench_production_tts.py \
  "${COMMON_ARGS[@]}" \
  --label "$ARM-warm-short" \
  --run-marker WARM000S \
  --ref-port 18082 \
  --jobs 4 \
  --chunks-per-job 4 \
  --text-profile short-en \
  --submit-concurrency 4 \
  --poll-concurrency 4 \
  --output "$RESULT_ROOT/$ARM/warm-short.json"

for ((run = START_RUN; run <= last_run; run++)); do
  # This marker depends only on the absolute run index. Dual and soft_reserved
  # therefore receive identical text at the same run index, regardless of ARM.
  printf -v run_marker 'R%07d' "$run"
  [[ "$run_marker" =~ ^[a-zA-Z0-9_-]{8}$ ]] || {
    echo "Generated invalid run marker: $run_marker" >&2
    exit 2
  }
  target_ns=$(
    .venv/bin/python -c 'import time; print(time.monotonic_ns() + 5_000_000_000)'
  )
  short_target_ns=$((target_ns + 750000000))
  long_output="$RESULT_ROOT/$ARM/mixed-long-r$run.json"
  short_output="$RESULT_ROOT/$ARM/mixed-short-r$run.json"

  .venv/bin/python scripts/bench_production_tts.py \
    "${COMMON_ARGS[@]}" \
    --label "$ARM-mixed-long-r$run" \
    --run-marker "$run_marker" \
    --start-at-monotonic-ns "$target_ns" \
    --ref-port 18081 \
    --jobs 10 \
    --chunks-per-job 10 \
    --text-profile long-en \
    --submit-concurrency 10 \
    --poll-concurrency 10 \
    --output "$long_output" \
    >"$RESULT_ROOT/$ARM/mixed-long-r$run.stdout" 2>&1 &
  long_pid=$!

  .venv/bin/python scripts/bench_production_tts.py \
    "${COMMON_ARGS[@]}" \
    --label "$ARM-mixed-short-r$run" \
    --run-marker "$run_marker" \
    --start-at-monotonic-ns "$short_target_ns" \
    --ref-port 18082 \
    --jobs 4 \
    --chunks-per-job 4 \
    --text-profile short-en \
    --submit-concurrency 4 \
    --poll-concurrency 4 \
    --output "$short_output" \
    >"$RESULT_ROOT/$ARM/mixed-short-r$run.stdout" 2>&1 &
  short_pid=$!

  wait "$long_pid"
  wait "$short_pid"
  .venv/bin/python \
    canary/higgs-sweetspot/benchmark/validate_round2_mixed.py \
    "$long_output" \
    "$short_output" \
    "$RESULT_ROOT/$ARM/mixed-summary-r$run.json"
done

curl --fail --silent --show-error http://127.0.0.1:6006/health
echo
