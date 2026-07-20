#!/usr/bin/env bash
set -euo pipefail

arm=${1:-}
if [[ "$arm" != "candidate" && "$arm" != "prod-clone" ]]; then
  echo "usage: $0 candidate|prod-clone" >&2
  exit 2
fi

RUNTIME_ROOT=/root/autodl-tmp/prod-compare/runtime
SUPERVISOR_CONF=/root/autodl-tmp/prod-sim/supervisord.conf
SUPERVISORCTL=/usr/bin/supervisorctl
PYTHON=/root/autodl-tmp/Fish-Audio/.venv/bin/python3
ACTIVE_LAUNCHER=/root/autodl-tmp/prod-sim/higgs-sglang-prod-sim.sh
ACTIVE_GATE_ENV=/root/autodl-tmp/prod-sim/prefill-gate.env
ROLLBACK_ROOT=/root/autodl-tmp/prod-compare/rollback
DIRTY_SENTINEL=/root/autodl-tmp/prod-sim/prefill-benchmark-dirty.json
LOCK_FILE=/root/autodl-tmp/prod-compare/switch-sglang.lock
CANDIDATE_SHA=349cadc352ce2ea96da1013a953757a815716f06390539365d8931fe02b44be6
PROD_CLONE_SHA=c7ce90d9980fdbc327427ded3e12559e9c5c474a7e9381f4c5e5f9ff47d2a1e6
LEGACY_CANDIDATE_SHA=5d50da493dcc03d173565ad2b7ac7d090150430828b5dd913d05ac553214b1e1
K16_ENV_SHA=8115d7602229ac9b0db08aa1d5cebed73ad694e666e27966df7e9e9a52166c34
MODEL_SNAPSHOT=/root/autodl-tmp/hf-cache/models--bosonai--higgs-audio-v3-tts-4b/snapshots/a7f70853f163c4cccbdd27ce9a80dd97961fc581

mkdir -p /root/autodl-tmp/prod-compare
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another SGLang arm switch is already running" >&2
  exit 2
fi
if [[ -e "$DIRTY_SENTINEL" ]]; then
  echo "refusing switch while benchmark dirty sentinel exists: $DIRTY_SENTINEL" >&2
  exit 2
fi
"$PYTHON" -c '
from pathlib import Path

expected = [
    b"/root/autodl-tmp/Fish-Audio/.venv/bin/python3",
    b"/root/autodl-tmp/Fish-Audio/.venv/bin/uvicorn",
    b"app:app",
    b"--host",
    b"127.0.0.1",
    b"--port",
    b"6007",
]
for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        cmdline = [item for item in cmdline_path.read_bytes().split(b"\0") if item]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if cmdline == expected:
        raise SystemExit("stop prod_clone_api before switching SGLang")
'
"$PYTHON" -c '
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:6006/health", timeout=10) as response:
    health = json.load(response)
jobs = health.get("tts_jobs") or {}
busy = (
    health.get("active_tts_jobs") != 0
    or health.get("outstanding_chunks") != 0
    or jobs.get("queued") != 0
    or jobs.get("running") != 0
    or any((health.get("lane_inflight") or {}).values())
    or any((health.get("lane_waiting") or {}).values())
)
if busy:
    raise SystemExit(f"candidate bridge is not idle: {health}")
'

if [[ "$arm" == "candidate" ]]; then
  selected="$RUNTIME_ROOT/higgs-sglang-prefill-gate.sh"
  expected_sha=$CANDIDATE_SHA
else
  selected="$RUNTIME_ROOT/higgs-sglang-prod-clone.sh"
  expected_sha=$PROD_CLONE_SHA
fi

actual_selected_sha=$(sha256sum "$selected" | awk '{print $1}')
if [[ "$actual_selected_sha" != "$expected_sha" ]]; then
  echo "selected launcher hash mismatch: $actual_selected_sha" >&2
  exit 2
fi

current_sha=$(sha256sum "$ACTIVE_LAUNCHER" | awk '{print $1}')
case "$current_sha" in
  "$CANDIDATE_SHA"|"$PROD_CLONE_SHA"|"$LEGACY_CANDIDATE_SHA") ;;
  *)
    echo "refusing to replace unknown active launcher: $current_sha" >&2
    exit 2
    ;;
esac

if [[ "$arm" == "candidate" ]]; then
  gate_source="$RUNTIME_ROOT/prefill-k16.env"
  actual_gate_sha=$(sha256sum "$gate_source" | awk '{print $1}')
  if [[ "$actual_gate_sha" != "$K16_ENV_SHA" ]]; then
    echo "K16 environment hash mismatch: $actual_gate_sha" >&2
    exit 2
  fi
fi

mkdir -p "$ROLLBACK_ROOT"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
rollback_launcher="$ROLLBACK_ROOT/higgs-sglang-$stamp-$current_sha.sh"
install -m 0755 "$ACTIVE_LAUNCHER" "$rollback_launcher"
rollback_gate_env="$ROLLBACK_ROOT/prefill-gate-$stamp.env"
gate_env_existed=0
if [[ -f "$ACTIVE_GATE_ENV" ]]; then
  install -m 0644 "$ACTIVE_GATE_ENV" "$rollback_gate_env"
  gate_env_existed=1
fi

wait_port_free() {
  local deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    if ! pgrep -f "sgl-omni serve" >/dev/null \
      && "$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 8000)); s.close()' 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "SGLang did not release process/port 8000" >&2
  return 1
}

wait_healthy() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "SGLang did not become healthy" >&2
  return 1
}

verify_active() {
  local expected_arm=$1
  local expected_launcher_sha expected_source expected_model expected_gate
  case "$expected_arm" in
    candidate)
      expected_launcher_sha=$CANDIDATE_SHA
      expected_source=/root/autodl-tmp/sglang-omni-prefill-gate-canary
      expected_model=$MODEL_SNAPSHOT
      expected_gate=present
      ;;
    prod-clone)
      expected_launcher_sha=$PROD_CLONE_SHA
      expected_source=/root/autodl-tmp/sglang-omni-prod-sim
      expected_model=$MODEL_SNAPSHOT
      expected_gate=absent
      ;;
    legacy-candidate)
      expected_launcher_sha=$LEGACY_CANDIDATE_SHA
      expected_source=/root/autodl-tmp/sglang-omni-prefill-gate-canary
      expected_model=bosonai/higgs-audio-v3-tts-4b
      expected_gate=present
      ;;
    *)
      echo "unknown verification arm: $expected_arm" >&2
      return 1
      ;;
  esac

  if [[ $(sha256sum "$ACTIVE_LAUNCHER" | awk '{print $1}') != "$expected_launcher_sha" ]]; then
    echo "active launcher changed after start" >&2
    return 1
  fi
  mapfile -t active_pids < <(pgrep -f "sgl-omni serve")
  if [[ ${#active_pids[@]} -ne 1 ]]; then
    echo "expected exactly one SGLang PID, got: ${active_pids[*]-}" >&2
    return 1
  fi
  local pid=${active_pids[0]}
  mapfile -d '' -t argv <"/proc/$pid/cmdline"
  declare -A options=()
  local index
  for ((index = 3; index < ${#argv[@]}; index += 2)); do
    options["${argv[index]}"]=${argv[index + 1]-}
  done
  if [[ ${options[--port]-} != 8000 \
    || ${options[--host]-} != 127.0.0.1 \
    || ${options[--model-path]-} != "$expected_model" ]]; then
    echo "active SGLang CLI does not match $expected_arm" >&2
    return 1
  fi
  if [[ $expected_arm != legacy-candidate \
    && ${options[--model-name]-} != bosonai/higgs-audio-v3-tts-4b ]]; then
    echo "active SGLang public model name changed" >&2
    return 1
  fi
  if [[ $expected_gate == present ]]; then
    if [[ ${options[--stages.2.factory_args.prefill_coalesce_requests]-} != 16 \
      || ${options[--stages.2.factory_args.prefill_coalesce_wait_ms]-} != 60 ]]; then
      echo "candidate prefill gate is not K16/T60" >&2
      return 1
    fi
  elif [[ -n ${options[--stages.2.factory_args.prefill_coalesce_requests]-} \
    || -n ${options[--stages.2.factory_args.prefill_coalesce_wait_ms]-} ]]; then
    echo "prod clone unexpectedly contains prefill gate CLI" >&2
    return 1
  fi
  if ! tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "PYTHONPATH=$expected_source" >/dev/null; then
    echo "active SGLang source does not match $expected_arm" >&2
    return 1
  fi
  if [[ $expected_arm != legacy-candidate ]]; then
    if ! tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "MODEL_PATH=$MODEL_SNAPSHOT" >/dev/null \
      || ! tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "HF_HUB_OFFLINE=1" >/dev/null \
      || ! tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "TRANSFORMERS_OFFLINE=1" >/dev/null; then
      echo "active SGLang is not pinned offline to the production snapshot" >&2
      return 1
    fi
  fi
  curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models \
    | "$PYTHON" -c 'import json, sys; data=json.load(sys.stdin); assert data["data"][0]["id"] == "bosonai/higgs-audio-v3-tts-4b"'
}

restore_previous() {
  local exit_code=$?
  trap - ERR
  echo "switch failed; restoring launcher $current_sha" >&2
  "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" stop higgs_sglang >/dev/null 2>&1 || true
  install -m 0755 "$rollback_launcher" "$ACTIVE_LAUNCHER.next"
  mv -f "$ACTIVE_LAUNCHER.next" "$ACTIVE_LAUNCHER"
  if (( gate_env_existed )); then
    install -m 0644 "$rollback_gate_env" "$ACTIVE_GATE_ENV.next"
    mv -f "$ACTIVE_GATE_ENV.next" "$ACTIVE_GATE_ENV"
  else
    rm -f "$ACTIVE_GATE_ENV"
  fi
  wait_port_free
  "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" start higgs_sglang
  wait_healthy
  if [[ "$current_sha" == "$PROD_CLONE_SHA" ]]; then
    verify_active prod-clone
  elif [[ "$current_sha" == "$CANDIDATE_SHA" ]]; then
    verify_active candidate
  else
    verify_active legacy-candidate
  fi
  exit "$exit_code"
}
trap restore_previous ERR

"$SUPERVISORCTL" -c "$SUPERVISOR_CONF" stop higgs_sglang >/dev/null 2>&1 || true
wait_port_free

if [[ "$arm" == "candidate" ]]; then
  install -m 0644 "$gate_source" "$ACTIVE_GATE_ENV.next"
  mv -f "$ACTIVE_GATE_ENV.next" "$ACTIVE_GATE_ENV"
fi
install -m 0755 "$selected" "$ACTIVE_LAUNCHER.next"
mv -f "$ACTIVE_LAUNCHER.next" "$ACTIVE_LAUNCHER"

"$SUPERVISORCTL" -c "$SUPERVISOR_CONF" start higgs_sglang
wait_healthy
verify_active "$arm"
trap - ERR

active_sha=$(sha256sum "$ACTIVE_LAUNCHER" | awk '{print $1}')
active_pid=$(pgrep -f "sgl-omni serve")
echo "active_arm=$arm launcher_sha=$active_sha pid=$active_pid"
