#!/usr/bin/env bash
set -euo pipefail

COMMAND_BOOT_ID=$(<"/proc/sys/kernel/random/boot_id")
COMMAND_START_EPOCH=$(
  /root/miniconda3/bin/python3 -c 'import time; print(time.monotonic_ns())'
)

# Test-host-only process controller. It never uses supervisorctl or broad
# process matching, so an arm switch can stop only the PID it started.

APP_ROOT=${APP_ROOT:-/root/autodl-tmp/Fish-Audio}
BASELINE_SOURCE_ROOT=/root/autodl-tmp/sglang-omni-k16
FSO_SOURCE_ROOT=/root/autodl-tmp/sglang-omni-fso-wt-a04c1b6
SYNCFREE_SOURCE_ROOT=/root/autodl-tmp/sglang-omni-syncfree-wt
UPSTREAM_SOURCE_ROOT=/root/autodl-tmp/sglang-omni-upstream-d957
G92_CONFIG_PATH="$APP_ROOT/canary/higgs-sweetspot/config/higgs-g92.yaml"
UPSTREAM_CONFIG_PATH="$APP_ROOT/canary/higgs-sweetspot/config/higgs-d957-triton.yaml"
FSO_REPO_ROOT=/root/autodl-tmp/fso-a04c1b6
FSO_PYTHON_ROOT="$FSO_REPO_ROOT/python"
FSO_NATIVE_EXTENSION="$FSO_PYTHON_ROOT/fish_scales_ops/_C.cpython-312-x86_64-linux-gnu.so"
FSO_LD_LIBRARY_PATH=/root/autodl-tmp/sglang-omni/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64
VENV_ROOT=${VENV_ROOT:-/root/autodl-tmp/sglang-omni}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:-/root/autodl-tmp/hf-cache/models--bosonai--higgs-audio-v3-tts-4b/snapshots/a7f70853f163c4cccbdd27ce9a80dd97961fc581}
STATE_DIR=${STATE_DIR:-/root/autodl-tmp/higgs-test-control}
LOG_DIR=${LOG_DIR:-/root/autodl-tmp/logs}

EXPECTED_SOURCE_DIFF=304eb276c6d3f19acbfa1bb32723f9c533b6d88be40a5f536cb83cf1ed9d097a
EXPECTED_SOURCE_HEAD=df62e91a00d383e6f73ab9604386ffac6c520529
EXPECTED_SOURCE_STATUS=abfbe6c8cdb655cfad3c9604dac169c3526bbd8b6b717f5cbd4337e3c12ac55c
EXPECTED_FSO_SOURCE_DIFF=8bb60e918e85087a929e9eaf105a31136f4fb240c5621778e1d640208086971e
EXPECTED_FSO_SOURCE_HEAD=df62e91a00d383e6f73ab9604386ffac6c520529
EXPECTED_FSO_SOURCE_STATUS=fae7f341b97d2c65b9702eade23bbf97295e5b4cfa3efbd6c7629e7e72146ed2
EXPECTED_SYNCFREE_SOURCE_DIFF=b1c134de8408ed001fa505263d58027fbd4951f31eeb5c7ece429a429b7f81ac
EXPECTED_SYNCFREE_SOURCE_HEAD=df62e91a00d383e6f73ab9604386ffac6c520529
EXPECTED_SYNCFREE_SOURCE_STATUS=f0deeb96764385323df863a31c6fe9ba588e0566dbbbc7e23f0519cedb957fd0
EXPECTED_UPSTREAM_SOURCE_DIFF=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
EXPECTED_UPSTREAM_SOURCE_HEAD=d957911f477c8dcfd6158c48b11c2e6e732b6af4
EXPECTED_UPSTREAM_SOURCE_STATUS=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
EXPECTED_G92_CONFIG_SHA256=7f792075960a8bc42b0bfe30f35f2646b28026bb844e24f043d0b381c8034ea1
EXPECTED_UPSTREAM_CONFIG_SHA256=6c0cefbc18245bd9e395f9099cbc96888cb6714872a2d404b071c6c2c5037bde
EXPECTED_FSO_REPO_HEAD=a04c1b63b1a7a670840fb3e97a82c0dbe2a35ded
EXPECTED_FSO_CUTLASS_HEAD=da5e086dab31d63815acafdac9a9c5893b1c69e2
EXPECTED_FSO_NATIVE_SHA256=b982aea7039302e0b6510f6c7b8745ce19ec17a21cfa55e2fc34e5aa267d1f4b
EXPECTED_FSO_RUNTIME_MANIFEST=0aa7fe5bc6d25ac3d6b407b13fb248690b86d6a1b64a67b2bef65ce5a3410783
EXPECTED_APP_SHA256=2728570f7e83011cf46f377cebbd4557d7e34d4d19333bd1d27b6497cc31f17f
EXPECTED_ENV_SHA256=994bfa61039c2767413b24b8493a0516320e376a1d06f3b7098b8be078d0b8c1
EXPECTED_LAUNCHER_SHA256=f7bb29bb4016601f20321e1b0e9a9b14bc722b9f2d1ab548f396deb32a480eb6
EXPECTED_ISOLATOR_SHA256=531f5e408cba37b709f192221764a626f9465bd7baa71ff8ca26144d74fe0aa7
EXPECTED_API_LAUNCH_SHA256=ec14f51a1f12862a71e1b8c28ee11e9317f20d2ca4fbb4c208fedbd5770bfd0a
EXPECTED_SGLANG_LAUNCH_SHA256=ceea44cbeb655300ede924bf8a844394474221c35c1cb02d0bd4579765850f21
EXPECTED_SIGNAL_LAUNCH_SHA256=7bceb17fce8551d22383e20fc9c99ebffb78f7ee599b9c595b7c69c51df4dd39
EXPECTED_MODEL_LINK=../../blobs/2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a
EXPECTED_MODEL_BYTES=9309834930
EXPECTED_MODEL_SHA256=${EXPECTED_MODEL_LINK##*/}
MODEL_VERIFY_STAMP="$STATE_DIR/model-content-verified"

mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 9>"$STATE_DIR/control.lock"
COMMAND=${1:-}
case "$COMMAND" in
  stop-api | stop-sglang | status)
    ;;
  *)
    flock -x 9
    ;;
esac

process_matches() {
  local pid=$1
  local expected=$2
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq -- "$expected"
}

process_env_matches() {
  local pid=$1
  local expected=$2
  [[ -r "/proc/$pid/environ" ]] || return 1
  tr '\0' '\n' <"/proc/$pid/environ" | grep -Fxq -- "$expected"
}

process_env_absent() {
  local pid=$1
  local name=$2
  [[ -r "/proc/$pid/environ" ]] || return 1
  ! tr '\0' '\n' <"/proc/$pid/environ" | grep -q "^${name}="
}

process_start_ticks() {
  local pid=$1
  [[ -r "/proc/$pid/stat" ]] || return 1
  [[ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ]] || return 1
  awk '{print $22}' "/proc/$pid/stat"
}

process_group() {
  local pid=$1
  ps -o pgid= -p "$pid" | tr -d ' '
}

terminate_group() {
  local pgid=$1
  local label=$2
  local token=$3
  [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || {
    echo "Refusing invalid $label process group: $pgid" >&2
    return 2
  }
  [[ "$token" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Refusing invalid $label launch token" >&2
    return 2
  }
  /root/miniconda3/bin/python3 \
    "$APP_ROOT/canary/higgs-sweetspot/runtime/signal_launch.py" \
    --token "$token" --pgid "$pgid" --label "$label"
}

PROVISIONAL_PID=
PROVISIONAL_START_TICKS=
PROVISIONAL_PIDFILE=
PROVISIONAL_PGID=
PROVISIONAL_TOKEN=
PROVISIONAL_IDENTITY_FILE=
PROVISIONAL_REMOVE_ARM_STATE=0

cleanup_provisional() {
  local pid=${PROVISIONAL_PID:-}
  local expected_ticks=${PROVISIONAL_START_TICKS:-}
  local clean=0
  local pgid=${PROVISIONAL_PGID:-}
  local token=${PROVISIONAL_TOKEN:-}
  if [[ "$pid" =~ ^[0-9]+$ && ! "$expected_ticks" =~ ^[0-9]+$ ]]; then
    # Never signal a bare numeric PID. Wait for the isolator's immutable
    # PID/start-ticks/PGID record, which is published before service exec.
    for _ in $(seq 1 100); do
      local published_pid published_ticks published_pgid published_token
      if [[ -f "${PROVISIONAL_IDENTITY_FILE:-}" ]] &&
        read -r published_pid published_ticks published_pgid published_token <"$PROVISIONAL_IDENTITY_FILE" &&
        [[ "$published_pid" == "$pid" && "$published_pgid" == "$published_pid" &&
          "$published_ticks" =~ ^[0-9]+$ && "$published_token" == "$token" ]]; then
        expected_ticks=$published_ticks
        pgid=$published_pgid
        break
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        # The identity may have been atomically published between the first
        # file check and observing leader exit. Adopt it before deciding that
        # no process group can survive.
        if [[ -f "${PROVISIONAL_IDENTITY_FILE:-}" ]] &&
          read -r published_pid published_ticks published_pgid published_token <"$PROVISIONAL_IDENTITY_FILE" &&
          [[ "$published_pid" == "$pid" && "$published_pgid" == "$published_pid" &&
            "$published_ticks" =~ ^[0-9]+$ && "$published_token" == "$token" ]]; then
          expected_ticks=$published_ticks
          pgid=$published_pgid
        else
          clean=1
        fi
        break
      fi
      sleep 0.1
    done
  fi
  if [[ "$clean" == 0 && "$pid" =~ ^[0-9]+$ &&
    "$expected_ticks" =~ ^[0-9]+$ &&
    "$pgid" == "$pid" && "$token" =~ ^[0-9a-f]{64}$ ]]; then
    local current_ticks
    current_ticks=$(process_start_ticks "$pid" 2>/dev/null || true)
    if [[ "$current_ticks" == "$expected_ticks" ]]; then
      terminate_group "$pgid" "provisional" "$token" && clean=1
    elif [[ -z "$current_ticks" ]]; then
      terminate_group "$pgid" "orphaned provisional" "$token" && clean=1
    fi
  fi
  if [[ "$clean" == 1 && -f "${PROVISIONAL_PIDFILE:-}" ]]; then
    local recorded_pid=
    read -r recorded_pid _ <"$PROVISIONAL_PIDFILE" || true
    [[ "$recorded_pid" != "$pid" ]] || rm -f "$PROVISIONAL_PIDFILE"
  fi
  if [[ "$clean" == 1 && -n "${PROVISIONAL_IDENTITY_FILE:-}" ]]; then
    rm -f "$PROVISIONAL_IDENTITY_FILE"
  fi
  if [[ "$clean" == 1 && "$PROVISIONAL_REMOVE_ARM_STATE" == 1 ]]; then
    rm -f "$STATE_DIR/arm" "$STATE_DIR/active-arm.env"
  fi
  if [[ "$clean" == 0 && "$pid" =~ ^[0-9]+$ ]]; then
    {
      printf 'pid=%s\n' "$pid"
      printf 'start_ticks=%s\n' "$expected_ticks"
      printf 'observed_pgid=%s\n' "$pgid"
      printf 'launch_token=%s\n' "$token"
      printf 'pidfile=%s\n' "${PROVISIONAL_PIDFILE:-}"
      printf 'reason=provisional cleanup could not prove the process group empty\n'
    } >"$STATE_DIR/provisional-quarantine-${pid}.env"
  fi
}

prepare_provisional_cleanup() {
  local pidfile=$1
  local identity_file=$2
  local launch_token=$3
  PROVISIONAL_PID=
  PROVISIONAL_START_TICKS=
  PROVISIONAL_PIDFILE=$pidfile
  PROVISIONAL_PGID=
  PROVISIONAL_TOKEN=$launch_token
  PROVISIONAL_IDENTITY_FILE=$identity_file
  PROVISIONAL_REMOVE_ARM_STATE=0
  trap cleanup_provisional EXIT
  # Ignore signals only across the tiny spawn -> $! registration window.
  trap '' INT TERM
}

activate_provisional_signals() {
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

capture_provisional_identity() {
  local pid=$1
  [[ "$PROVISIONAL_PID" == "$pid" ]] || {
    echo "Provisional PID registration mismatch" >&2
    return 2
  }
  for _ in $(seq 1 100); do
    local published_pid published_ticks published_pgid published_token
    if [[ -f "$PROVISIONAL_IDENTITY_FILE" ]] &&
      read -r published_pid published_ticks published_pgid published_token <"$PROVISIONAL_IDENTITY_FILE" &&
      [[ "$published_pid" == "$pid" && "$published_pgid" == "$pid" &&
        "$published_ticks" =~ ^[0-9]+$ &&
        "$published_token" == "$PROVISIONAL_TOKEN" ]]; then
      PROVISIONAL_START_TICKS=$published_ticks
      PROVISIONAL_PGID=$published_pgid
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  echo "Unable to capture isolated process identity for PID $pid" >&2
  return 2
}

disarm_provisional_cleanup() {
  trap '' INT TERM
  trap - EXIT
  PROVISIONAL_PID=
  PROVISIONAL_START_TICKS=
  PROVISIONAL_PIDFILE=
  PROVISIONAL_PGID=
  PROVISIONAL_TOKEN=
  [[ -z "${PROVISIONAL_IDENTITY_FILE:-}" ]] || rm -f "$PROVISIONAL_IDENTITY_FILE"
  PROVISIONAL_IDENTITY_FILE=
  PROVISIONAL_REMOVE_ARM_STATE=0
  trap - INT
  trap - TERM
}

require_clean_control_state() {
  local path
  local dirty=0
  for path in \
    "$STATE_DIR"/provisional-quarantine-*.env \
    "$STATE_DIR"/*.launch.*; do
    [[ -e "$path" ]] || continue
    echo "Refusing start while unresolved control state exists: $path" >&2
    dirty=1
  done
  [[ "$dirty" == 0 ]]
}

time_ns() {
  /root/miniconda3/bin/python3 -c 'import time; print(time.monotonic_ns())'
}

write_cancel_marker() {
  local component=$1
  local marker="$STATE_DIR/$component.cancel"
  local lock="$STATE_DIR/$component.cancel.lock"
  local tmp="$marker.tmp.$$"
  local cancel_epoch cancel_fd
  exec {cancel_fd}>"$lock"
  flock -x "$cancel_fd"
  cancel_epoch=$(time_ns)
  local existing_boot= existing_epoch=
  if [[ -f "$marker" ]]; then
    read -r existing_boot existing_epoch <"$marker" || true
    if [[ "$existing_boot" == "$COMMAND_BOOT_ID" &&
      "$existing_epoch" =~ ^[0-9]+$ &&
      "$existing_epoch" -gt "$cancel_epoch" ]]; then
      cancel_epoch=$existing_epoch
    fi
  fi
  printf '%s %s\n' "$COMMAND_BOOT_ID" "$cancel_epoch" >"$tmp"
  mv "$tmp" "$marker"
  flock -u "$cancel_fd"
  exec {cancel_fd}>&-
}

stop_published_launch() {
  local identity_file=$1
  local label=$2
  local pid start_ticks pgid token
  if ! read -r pid start_ticks pgid token <"$identity_file" ||
    [[ ! "$pid" =~ ^[0-9]+$ || ! "$start_ticks" =~ ^[0-9]+$ ||
      "$pgid" != "$pid" || ! "$token" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Refusing invalid $label launch identity: $identity_file" >&2
    return 2
  fi
  local current_ticks
  current_ticks=$(process_start_ticks "$pid" 2>/dev/null || true)
  if [[ -n "$current_ticks" ]]; then
    [[ "$current_ticks" == "$start_ticks" ]] || {
      echo "Refusing $label launch identity with reused PID $pid" >&2
      return 2
    }
    local current_pgid
    current_pgid=$(process_group "$pid" 2>/dev/null || true)
    if [[ -n "$current_pgid" && "$current_pgid" != "$pgid" ]]; then
      echo "Refusing $label launch identity with changed process group" >&2
      return 2
    fi
    process_env_matches "$pid" "HIGGS_TEST_LAUNCH_TOKEN=$token" || {
      echo "Refusing $label launch identity with changed launch token" >&2
      return 2
    }
  fi
  terminate_group "$pgid" "published $label launch" "$token" || return 2
  rm -f "$identity_file" "$STATE_DIR/provisional-quarantine-${pid}.env"
}

cancelled_since() {
  local component=$1
  local start_epoch=$2
  local marker="$STATE_DIR/$component.cancel"
  local cancel_boot= cancel_epoch=
  [[ -f "$marker" ]] || return 1
  read -r cancel_boot cancel_epoch <"$marker" || return 1
  [[ "$cancel_boot" == "$COMMAND_BOOT_ID" ]] || return 1
  [[ "$cancel_epoch" =~ ^[0-9]+$ ]] || return 1
  ((cancel_epoch > start_epoch))
}

stop_component() {
  local component=$1
  local pidfile=$2
  local expected=$3
  local label=$4
  write_cancel_marker "$component"
  for _ in $(seq 1 1200); do
    local launch_identity
    for launch_identity in "$STATE_DIR/$component.launch."*; do
      [[ -e "$launch_identity" ]] || continue
      stop_published_launch "$launch_identity" "$label"
    done
    if [[ -f "$pidfile" ]]; then
      stop_pidfile "$pidfile" "$expected" "$label"
      require_clean_control_state
      return
    fi
    if flock -n 9; then
      # The starter may have published state immediately before releasing the
      # lock, after this loop's first scan. Re-scan while holding the lock.
      for launch_identity in "$STATE_DIR/$component.launch."*; do
        [[ -e "$launch_identity" ]] || continue
        stop_published_launch "$launch_identity" "$label"
      done
      if [[ -f "$pidfile" ]]; then
        stop_pidfile "$pidfile" "$expected" "$label"
      fi
      require_clean_control_state
      return 0
    fi
    sleep 0.1
  done
  echo "Timed out waiting to stop an in-progress $label start" >&2
  return 2
}

port_open() {
  local port=$1
  /root/miniconda3/bin/python3 -c \
    'import socket,sys
port=int(sys.argv[1])
probe=socket.socket()
probe.settimeout(0.2)
if probe.connect_ex(("127.0.0.1", port)) == 0:
    probe.close()
    raise SystemExit(0)
probe.close()
s=socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
unavailable=False
try:
    s.bind(("127.0.0.1", port))
except OSError:
    unavailable=True
finally:
    s.close()
raise SystemExit(0 if unavailable else 1)' \
    "$port"
}

wait_for_vanilla_bind() {
  local port=$1
  # SGLang-Omni probes its requested port with a plain bind() and otherwise
  # silently moves to a random port. Wait out TIME_WAIT sockets using the same
  # semantics so a controlled arm can never become healthy on the wrong port.
  for _ in $(seq 1 180); do
    if /root/miniconda3/bin/python3 -c \
      'import socket,sys
s=socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()' \
      "$port"; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for loopback port $port to become vanilla-bindable" >&2
  return 2
}

write_pid_state() {
  local pidfile=$1
  local pid=$2
  local start_ticks=$3
  local pgid=$4
  local token=$5
  local tmp="$pidfile.tmp.$$"
  printf '%s %s %s %s\n' "$pid" "$start_ticks" "$pgid" "$token" >"$tmp"
  mv "$tmp" "$pidfile"
}

read_pid_state() {
  local pidfile=$1
  read -r STATE_PID STATE_START_TICKS STATE_PGID STATE_TOKEN <"$pidfile"
  [[ "$STATE_PID" =~ ^[0-9]+$ ]] || return 1
  [[ "$STATE_START_TICKS" =~ ^[0-9]+$ ]] || return 1
  [[ "$STATE_PGID" =~ ^[0-9]+$ ]] || return 1
  [[ "$STATE_TOKEN" =~ ^[0-9a-f]{64}$ ]] || return 1
}

pid_state_matches() {
  local pidfile=$1
  local expected=$2
  local STATE_PID STATE_START_TICKS STATE_PGID STATE_TOKEN
  read_pid_state "$pidfile" || return 1
  [[ "$(process_start_ticks "$STATE_PID" 2>/dev/null || true)" == "$STATE_START_TICKS" ]] &&
    [[ "$(process_group "$STATE_PID" 2>/dev/null || true)" == "$STATE_PGID" ]] &&
    [[ "$STATE_PGID" == "$STATE_PID" ]] &&
    process_env_matches "$STATE_PID" "HIGGS_TEST_LAUNCH_TOKEN=$STATE_TOKEN" &&
    process_matches "$STATE_PID" "$expected"
}

wait_for_identity() {
  local pid=$1
  local expected=$2
  for _ in $(seq 1 100); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if process_matches "$pid" "$expected"; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

stop_pidfile() {
  local pidfile=$1
  local expected=$2
  local label=$3
  [[ -f "$pidfile" ]] || return 0

  local STATE_PID STATE_START_TICKS STATE_PGID STATE_TOKEN
  if ! read_pid_state "$pidfile"; then
    echo "Refusing invalid $label PID file: $pidfile" >&2
    return 2
  fi
  local pid=$STATE_PID
  [[ "$STATE_PGID" == "$STATE_PID" ]] || {
    echo "Refusing $label with an invalid recorded process group" >&2
    return 2
  }
  local current_ticks
  current_ticks=$(process_start_ticks "$pid" 2>/dev/null || true)
  if [[ -z "$current_ticks" ]]; then
    terminate_group "$STATE_PGID" "orphaned $label" "$STATE_TOKEN" || return 2
    rm -f "$pidfile"
    return
  fi
  [[ "$current_ticks" == "$STATE_START_TICKS" ]] || {
    echo "Refusing to stop PID $pid: start time changed" >&2
    return 2
  }
  process_env_matches "$pid" "HIGGS_TEST_LAUNCH_TOKEN=$STATE_TOKEN" || {
    echo "Refusing to stop PID $pid: launch token changed" >&2
    return 2
  }
  if ! process_matches "$pid" "$expected"; then
    if [[ -z "$(process_start_ticks "$pid" 2>/dev/null || true)" ]]; then
      terminate_group "$STATE_PGID" "orphaned $label" "$STATE_TOKEN" || return 2
      rm -f "$pidfile"
      return
    fi
    echo "Refusing to stop PID $pid: it is not the recorded $label process" >&2
    return 2
  fi
  local pgid
  pgid=$(process_group "$pid" 2>/dev/null || true)
  if [[ -z "$pgid" && -z "$(process_start_ticks "$pid" 2>/dev/null || true)" ]]; then
    terminate_group "$STATE_PGID" "orphaned $label" "$STATE_TOKEN" || return 2
    rm -f "$pidfile"
    return
  fi
  if [[ "$pgid" != "$pid" || "$pgid" != "$STATE_PGID" ]]; then
    echo "Refusing to stop $label: process group identity changed" >&2
    return 2
  fi

  terminate_group "$pgid" "$label" "$STATE_TOKEN" || return 2
  rm -f "$pidfile"
}

verify_model_file() {
  local name=$1
  local expected_link=$2
  local expected_bytes=$3
  [[ "$(readlink "$MODEL_SNAPSHOT/$name")" == "$expected_link" ]] || {
    echo "Pinned model symlink mismatch: $name" >&2
    return 2
  }
  [[ "$(stat -Lc '%s' "$MODEL_SNAPSHOT/$name")" == "$expected_bytes" ]] || {
    echo "Pinned model size mismatch: $name" >&2
    return 2
  }
}

model_stat_fingerprint() {
  stat -Lc '%d:%i:%s:%Y:%Z' "$MODEL_SNAPSHOT/model.safetensors"
}

verify_model_content() {
  local before_fingerprint after_fingerprint actual_hash
  before_fingerprint=$(model_stat_fingerprint) || {
    echo "Unable to stat pinned model before hashing" >&2
    return 2
  }
  actual_hash=$(sha256sum "$MODEL_SNAPSHOT/model.safetensors" | awk '{print $1}')
  after_fingerprint=$(model_stat_fingerprint) || {
    echo "Unable to stat pinned model after hashing" >&2
    return 2
  }
  [[ "$before_fingerprint" == "$after_fingerprint" ]] || {
    echo "Pinned model changed while its content was being verified" >&2
    return 2
  }
  [[ "$actual_hash" == "$EXPECTED_MODEL_SHA256" ]] || {
    echo "Pinned model content hash mismatch" >&2
    return 2
  }
  local tmp="$MODEL_VERIFY_STAMP.tmp.$$"
  {
    printf 'sha256=%s\n' "$actual_hash"
    printf 'stat=%s\n' "$after_fingerprint"
  } >"$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$MODEL_VERIFY_STAMP"
}

require_model_verification_stamp() {
  [[ -f "$MODEL_VERIFY_STAMP" ]] || {
    echo "Model content is not verified; run '$0 verify' first" >&2
    return 2
  }
  local expected_stamp actual_stamp
  expected_stamp=$(<"$MODEL_VERIFY_STAMP")
  actual_stamp=$(
    printf 'sha256=%s\n' "$EXPECTED_MODEL_SHA256"
    printf 'stat=%s' "$(model_stat_fingerprint)"
  )
  [[ "$expected_stamp" == "$actual_stamp" ]] || {
    echo "Model verification stamp is stale; run '$0 verify' again" >&2
    return 2
  }
}

verify_source_tree() {
  local root=$1
  local expected_head=$2
  local expected_diff=$3
  local expected_status=$4
  local label=$5
  [[ "$(git -C "$root" rev-parse HEAD)" == "$expected_head" ]] || {
    echo "$label source HEAD mismatch" >&2
    return 2
  }
  local actual_diff
  actual_diff=$(git -C "$root" diff HEAD --binary | sha256sum | awk '{print $1}')
  [[ "$actual_diff" == "$expected_diff" ]] || {
    echo "$label source diff mismatch: $actual_diff" >&2
    return 2
  }
  local actual_status
  actual_status=$(
    git -C "$root" status --porcelain=v1 --untracked-files=all |
      sha256sum |
      awk '{print $1}'
  )
  [[ "$actual_status" == "$expected_status" ]] || {
    echo "$label source status mismatch: $actual_status" >&2
    return 2
  }
  git -C "$root" diff --check
}

fso_runtime_manifest() {
  (
    cd "$FSO_REPO_ROOT"
    find python/fish_scales_ops -type f \
      \( -name '*.py' -o -name '*.so' \) -print0 |
      sort -z |
      xargs -0 sha256sum |
      sha256sum |
      awk '{print $1}'
  )
}

verify_fso_runtime() {
  verify_source_tree \
    "$FSO_SOURCE_ROOT" \
    "$EXPECTED_FSO_SOURCE_HEAD" \
    "$EXPECTED_FSO_SOURCE_DIFF" \
    "$EXPECTED_FSO_SOURCE_STATUS" \
    "FSO candidate"
  [[ "$(git -C "$FSO_REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_FSO_REPO_HEAD" ]] || {
    echo "FSO repository HEAD mismatch" >&2
    return 2
  }
  [[ "$(git -C "$FSO_REPO_ROOT/3rdparty/cutlass" rev-parse HEAD)" == "$EXPECTED_FSO_CUTLASS_HEAD" ]] || {
    echo "FSO CUTLASS HEAD mismatch" >&2
    return 2
  }
  [[ -f "$FSO_NATIVE_EXTENSION" && ! -L "$FSO_NATIVE_EXTENSION" ]] || {
    echo "Pinned FSO native extension is missing or is a symlink" >&2
    return 2
  }
  local before_stat after_stat native_hash
  before_stat=$(stat -Lc '%d:%i:%s:%Y:%Z' "$FSO_NATIVE_EXTENSION")
  native_hash=$(sha256sum "$FSO_NATIVE_EXTENSION" | awk '{print $1}')
  after_stat=$(stat -Lc '%d:%i:%s:%Y:%Z' "$FSO_NATIVE_EXTENSION")
  [[ "$before_stat" == "$after_stat" ]] || {
    echo "FSO native extension changed while hashing" >&2
    return 2
  }
  [[ "$native_hash" == "$EXPECTED_FSO_NATIVE_SHA256" ]] || {
    echo "FSO native extension content mismatch" >&2
    return 2
  }
  local manifest_before manifest_after
  manifest_before=$(fso_runtime_manifest)
  manifest_after=$(fso_runtime_manifest)
  [[ "$manifest_before" == "$manifest_after" ]] || {
    echo "FSO Python runtime changed while hashing" >&2
    return 2
  }
  [[ "$manifest_after" == "$EXPECTED_FSO_RUNTIME_MANIFEST" ]] || {
    echo "FSO Python runtime manifest mismatch: $manifest_after" >&2
    return 2
  }
  local imported_extension
  imported_extension=$(
    env -i \
      PATH="$VENV_ROOT/.venv/bin:/usr/bin:/bin" \
      PYTHONPATH="$FSO_PYTHON_ROOT" \
      LD_LIBRARY_PATH="$FSO_LD_LIBRARY_PATH" \
      "$VENV_ROOT/.venv/bin/python" -c \
      'import os; import fish_scales_ops._C as native; print(os.path.realpath(native.__file__))'
  )
  [[ "$imported_extension" == "$(readlink -f "$FSO_NATIVE_EXTENSION")" ]] || {
    echo "Imported FSO native extension path mismatch: $imported_extension" >&2
    return 2
  }
}

verify_baseline() {
  local model_stamp_mode=${1:-require-model-stamp}
  [[ -x "$VENV_ROOT/.venv/bin/sgl-omni" ]] || {
    echo "Missing SGLang environment" >&2
    return 2
  }
  [[ -x "$APP_ROOT/.venv/bin/uvicorn" ]] || {
    echo "Missing API environment" >&2
    return 2
  }
  [[ -f "$APP_ROOT/.env" ]] || {
    echo "Missing test .env" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/.env" | awk '{print $1}')" == "$EXPECTED_ENV_SHA256" ]] || {
    echo "Test .env fingerprint mismatch" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/scripts/run_sglang.sh" | awk '{print $1}')" == "$EXPECTED_LAUNCHER_SHA256" ]] || {
    echo "SGLang launcher fingerprint mismatch" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/canary/higgs-sweetspot/runtime/exec_isolated.py" | awk '{print $1}')" == "$EXPECTED_ISOLATOR_SHA256" ]] || {
    echo "Isolated launcher fingerprint mismatch" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/canary/higgs-sweetspot/runtime/launch_api.sh" | awk '{print $1}')" == "$EXPECTED_API_LAUNCH_SHA256" ]] || {
    echo "API launch wrapper fingerprint mismatch" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/canary/higgs-sweetspot/runtime/launch_sglang.sh" | awk '{print $1}')" == "$EXPECTED_SGLANG_LAUNCH_SHA256" ]] || {
    echo "SGLang launch wrapper fingerprint mismatch" >&2
    return 2
  }
  [[ "$(sha256sum "$APP_ROOT/canary/higgs-sweetspot/runtime/signal_launch.py" | awk '{print $1}')" == "$EXPECTED_SIGNAL_LAUNCH_SHA256" ]] || {
    echo "Launch signal helper fingerprint mismatch" >&2
    return 2
  }
  /root/miniconda3/bin/python3 \
    "$APP_ROOT/canary/higgs-sweetspot/runtime/signal_launch.py" --probe
  verify_model_file model.safetensors "$EXPECTED_MODEL_LINK" "$EXPECTED_MODEL_BYTES"
  verify_model_file config.json ../../blobs/fb8848872136814c75ee10055d83ecc38f5f0169 2755
  verify_model_file model.safetensors.index.json ../../blobs/6f55f418d83cc485a7f7f116b919df88c3148138 90103
  verify_model_file tokenizer.json ../../blobs/eb883de2de5adc5113f1f02b54830a0ea7cd6ef191cde65c41aceb3737d4d1c1 11433924
  verify_model_file tokenizer_config.json ../../blobs/cfbe88c0bb630370fce5c497550b00a5b3608edc 1937
  verify_model_file chat_template.jinja ../../blobs/28028c056af412405debd878cdda0171e35fa5d1 2427
  if [[ "$model_stamp_mode" == require-model-stamp ]]; then
    require_model_verification_stamp
  elif [[ "$model_stamp_mode" != skip-model-stamp ]]; then
    echo "Invalid model verification mode: $model_stamp_mode" >&2
    return 2
  fi
  [[ "$(sha256sum "$APP_ROOT/app.py" | awk '{print $1}')" == "$EXPECTED_APP_SHA256" ]] || {
    echo "Bridge app.py fingerprint mismatch" >&2
    return 2
  }
  verify_source_tree \
    "$BASELINE_SOURCE_ROOT" \
    "$EXPECTED_SOURCE_HEAD" \
    "$EXPECTED_SOURCE_DIFF" \
    "$EXPECTED_SOURCE_STATUS" \
    "K16 baseline"
}

start_api() {
  local start_epoch
  start_epoch=$COMMAND_START_EPOCH
  require_clean_control_state
  local pidfile="$STATE_DIR/api.pid"
  local pcm_stats_audioop=${TEST_PCM_STATS_AUDIOOP:-0}
  local lane_admission_mode=${TEST_LANE_ADMISSION_MODE:-dual}
  local ffmpeg_timing=${TEST_FFMPEG_TIMING:-0}
  local max_concurrent_chunks=${TEST_MAX_CONCURRENT_CHUNKS:-96}
  [[ "$pcm_stats_audioop" =~ ^(0|peak|all)$ ]] || {
    echo "Invalid TEST_PCM_STATS_AUDIOOP" >&2
    return 2
  }
  [[ "$lane_admission_mode" =~ ^(dual|soft_reserved)$ ]] || {
    echo "Invalid TEST_LANE_ADMISSION_MODE" >&2
    return 2
  }
  [[ "$ffmpeg_timing" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_FFMPEG_TIMING" >&2
    return 2
  }
  [[ "$max_concurrent_chunks" =~ ^[0-9]+$ &&
    "$max_concurrent_chunks" -gt 0 &&
    "$max_concurrent_chunks" -le 512 ]] || {
    echo "Invalid TEST_MAX_CONCURRENT_CHUNKS" >&2
    return 2
  }
  port_open 6006 && {
    echo "Refusing start: loopback port 6006 is already in use" >&2
    return 2
  }
  if [[ -f "$pidfile" ]]; then
    echo "Refusing existing API PID file" >&2
    return 2
  fi
  verify_baseline
  if cancelled_since api "$start_epoch"; then
    echo "API start was cancelled before launch" >&2
    return 2
  fi

  local identity_file="$STATE_DIR/api.launch.$$"
  local launch_token
  launch_token=$(/root/miniconda3/bin/python3 -c 'import secrets; print(secrets.token_hex(32))')
  prepare_provisional_cleanup "$pidfile" "$identity_file" "$launch_token"
  (
    trap - INT TERM
    exec 9>&-
    export HIGGS_TEST_LAUNCH_TOKEN="$launch_token"
    export HIGGS_TEST_PCM_STATS_AUDIOOP="$pcm_stats_audioop"
    export HIGGS_TEST_LANE_ADMISSION_MODE="$lane_admission_mode"
    export HIGGS_TEST_FFMPEG_TIMING="$ffmpeg_timing"
    export HIGGS_TEST_MAX_CONCURRENT_CHUNKS="$max_concurrent_chunks"
    export APP_ROOT
    exec /root/miniconda3/bin/python3 \
      "$APP_ROOT/canary/higgs-sweetspot/runtime/exec_isolated.py" \
      "$identity_file" -- \
      "$APP_ROOT/canary/higgs-sweetspot/runtime/launch_api.sh"
  ) >"$LOG_DIR/higgs_test_api.log" 2>&1 &
  local pid=$!
  PROVISIONAL_PID=$pid
  activate_provisional_signals
  capture_provisional_identity "$pid"
  if ! wait_for_identity "$pid" "uvicorn app:app"; then
    echo "Test API exited before identity verification" >&2
    return 2
  fi
  local pgid start_ticks
  pgid=$(process_group "$pid")
  start_ticks=$(process_start_ticks "$pid")
  [[ "$pgid" == "$pid" ]] || {
    echo "Test API failed to enter an isolated process group" >&2
    return 2
  }
  PROVISIONAL_PGID=$pgid
  write_pid_state "$pidfile" "$pid" "$start_ticks" "$pgid" "$launch_token"
  for _ in $(seq 1 60); do
    if curl --fail --silent --show-error --max-time 2 \
      http://127.0.0.1:6006/openapi.json >/dev/null; then
      pid_state_matches "$pidfile" "uvicorn app:app" || {
        echo "Test API identity changed during startup" >&2
        return 2
      }
      process_env_matches "$pid" "HIGGS_PCM_STATS_AUDIOOP=$pcm_stats_audioop" || {
        echo "Test API environment does not contain the PCM stats mode" >&2
        return 2
      }
      process_env_matches "$pid" "HIGGS_LANE_ADMISSION_MODE=$lane_admission_mode" || {
        echo "Test API environment does not contain the lane admission mode" >&2
        return 2
      }
      process_env_matches "$pid" "HIGGS_FFMPEG_TIMING=$ffmpeg_timing" || {
        echo "Test API environment does not contain the FFmpeg timing mode" >&2
        return 2
      }
      process_env_matches "$pid" "MAX_CONCURRENT_CHUNKS=$max_concurrent_chunks" || {
        echo "Test API environment does not contain the chunk capacity" >&2
        return 2
      }
      {
        printf 'pcm_stats_audioop=%s\n' "$pcm_stats_audioop"
        printf 'lane_admission_mode=%s\n' "$lane_admission_mode"
        printf 'ffmpeg_timing=%s\n' "$ffmpeg_timing"
        printf 'max_concurrent_chunks=%s\n' "$max_concurrent_chunks"
      } >"$STATE_DIR/active-api.env"
      disarm_provisional_cleanup
      echo "Started healthy test API PID $pid"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if stop_pidfile "$pidfile" "uvicorn app:app" "API"; then
    disarm_provisional_cleanup
  fi
  echo "Test API failed its startup health gate" >&2
  return 2
}

start_sglang() {
  local start_epoch
  start_epoch=$COMMAND_START_EPOCH
  require_clean_control_state
  local pidfile="$STATE_DIR/sglang.pid"
  port_open 8000 && {
    echo "Refusing start: loopback port 8000 is already in use" >&2
    return 2
  }
  wait_for_vanilla_bind 8000
  [[ ! -f "$pidfile" ]] || {
    echo "SGLang PID file exists; stop the active arm first" >&2
    return 2
  }
  local fso_mxfp8=${TEST_FSO_MXFP8:-0}
  local fso_source_only=${TEST_FSO_SOURCE_ONLY:-0}
  local syncfree_source=${TEST_SYNCFREE_SOURCE:-0}
  local syncfree_launch=${TEST_SYNCFREE_LAUNCH:-0}
  local upstream_source=${TEST_UPSTREAM_SOURCE:-0}
  local config_variant=${TEST_CONFIG_VARIANT:-default}
  local config_path=
  local source_root runtime_variant runtime_pythonpath runtime_ld_library_path
  local source_head source_diff source_status verify_fso_candidate
  [[ "$upstream_source" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_UPSTREAM_SOURCE" >&2
    return 2
  }
  if [[ "$upstream_source" == 1 ]]; then
    if [[ "$syncfree_source:$fso_mxfp8:$fso_source_only" != 0:0:0 ]]; then
      echo "Upstream source cannot be combined with sync-free or FSO source modes" >&2
      return 2
    fi
    source_root=$UPSTREAM_SOURCE_ROOT
    runtime_variant=upstream-d957
    runtime_pythonpath=$UPSTREAM_SOURCE_ROOT
    runtime_ld_library_path=
    source_head=$EXPECTED_UPSTREAM_SOURCE_HEAD
    source_diff=$EXPECTED_UPSTREAM_SOURCE_DIFF
    source_status=$EXPECTED_UPSTREAM_SOURCE_STATUS
    verify_fso_candidate=0
  else case "$syncfree_source:$fso_mxfp8:$fso_source_only" in
    0:0:0)
      source_root=$BASELINE_SOURCE_ROOT
      runtime_variant=baseline
      runtime_pythonpath=$BASELINE_SOURCE_ROOT
      runtime_ld_library_path=
      source_head=$EXPECTED_SOURCE_HEAD
      source_diff=$EXPECTED_SOURCE_DIFF
      source_status=$EXPECTED_SOURCE_STATUS
      verify_fso_candidate=0
      ;;
    0:0:1)
      source_root=$FSO_SOURCE_ROOT
      runtime_variant=instrumented-bf16
      runtime_pythonpath=$FSO_SOURCE_ROOT
      runtime_ld_library_path=
      source_head=$EXPECTED_FSO_SOURCE_HEAD
      source_diff=$EXPECTED_FSO_SOURCE_DIFF
      source_status=$EXPECTED_FSO_SOURCE_STATUS
      verify_fso_candidate=1
      ;;
    0:1:0)
      source_root=$FSO_SOURCE_ROOT
      runtime_variant=fso-mxfp8-gate-up
      runtime_pythonpath="$FSO_SOURCE_ROOT:$FSO_PYTHON_ROOT"
      runtime_ld_library_path=$FSO_LD_LIBRARY_PATH
      source_head=$EXPECTED_FSO_SOURCE_HEAD
      source_diff=$EXPECTED_FSO_SOURCE_DIFF
      source_status=$EXPECTED_FSO_SOURCE_STATUS
      verify_fso_candidate=1
      ;;
    1:0:0)
      source_root=$SYNCFREE_SOURCE_ROOT
      runtime_variant=syncfree
      runtime_pythonpath=$SYNCFREE_SOURCE_ROOT
      runtime_ld_library_path=
      source_head=$EXPECTED_SYNCFREE_SOURCE_HEAD
      source_diff=$EXPECTED_SYNCFREE_SOURCE_DIFF
      source_status=$EXPECTED_SYNCFREE_SOURCE_STATUS
      verify_fso_candidate=0
      ;;
    *)
      echo "Invalid source mode; sync-free and FSO modes are mutually exclusive" >&2
      return 2
      ;;
  esac
  fi
  verify_baseline
  if [[ "$verify_fso_candidate" == 1 ]]; then
    verify_fso_runtime
  fi
  if [[ "$syncfree_source" == 1 ]]; then
    verify_source_tree \
      "$SYNCFREE_SOURCE_ROOT" \
      "$EXPECTED_SYNCFREE_SOURCE_HEAD" \
      "$EXPECTED_SYNCFREE_SOURCE_DIFF" \
      "$EXPECTED_SYNCFREE_SOURCE_STATUS" \
      "sync-free candidate"
  fi
  if [[ "$upstream_source" == 1 ]]; then
    verify_source_tree \
      "$UPSTREAM_SOURCE_ROOT" \
      "$EXPECTED_UPSTREAM_SOURCE_HEAD" \
      "$EXPECTED_UPSTREAM_SOURCE_DIFF" \
      "$EXPECTED_UPSTREAM_SOURCE_STATUS" \
      "upstream d957 candidate"
  fi
  if cancelled_since sglang "$start_epoch"; then
    echo "SGLang start was cancelled before launch" >&2
    return 2
  fi

  local arm=${TEST_ARM_LABEL:?Set TEST_ARM_LABEL}
  local prefill_k=${TEST_PREFILL_K:-16}
  local prefill_wait_ms=${TEST_PREFILL_WAIT_MS:-60}
  local quantization=${TEST_QUANTIZATION:-none}
  local attention_backend=${TEST_ATTENTION_BACKEND:-triton}
  local page_size=${TEST_PAGE_SIZE:-1}
  local log_level=${TEST_LOG_LEVEL:-info}
  local ras_win_len=${TEST_RAS_WIN_LEN:-7}
  local schedule_policy=${TEST_SCHEDULE_POLICY:-fcfs}
  local schedule_conservativeness=${TEST_SCHEDULE_CONSERVATIVENESS:-1.0}
  local torch_compile=${TEST_TORCH_COMPILE:-0}
  local max_running_requests=${TEST_MAX_RUNNING_REQUESTS:-96}
  local cuda_graph_max_bs=${TEST_CUDA_GRAPH_MAX_BS:-96}
  local extra_server_args=${TEST_EXTRA_SERVER_ARGS:-}
  local upstream_frontend_split=${TEST_UPSTREAM_FRONTEND_SPLIT:-1}
  local upstream_vocoder_split=${TEST_UPSTREAM_VOCODER_SPLIT:-1}
  local upstream_compile_decode=${TEST_UPSTREAM_COMPILE_DECODE:-1}
  local upstream_preprocess_concurrency=${TEST_UPSTREAM_PREPROCESS_CONCURRENCY:-16}
  local upstream_vocoder_batch=${TEST_UPSTREAM_VOCODER_BATCH:-16}
  [[ "$syncfree_launch" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_SYNCFREE_LAUNCH" >&2
    return 2
  }
  if [[ "$syncfree_launch" == 1 && "$syncfree_source" != 1 && "$upstream_source" != 1 ]]; then
    echo "Sync-free launch requires the patched sync-free source or upstream d957" >&2
    return 2
  fi
  case "$config_variant" in
    default)
      config_path=
      ;;
    g92)
      [[ -f "$G92_CONFIG_PATH" && ! -L "$G92_CONFIG_PATH" ]] || {
        echo "Pinned G92 config is missing or is a symlink" >&2
        return 2
      }
      [[ "$(sha256sum "$G92_CONFIG_PATH" | awk '{print $1}')" == "$EXPECTED_G92_CONFIG_SHA256" ]] || {
        echo "Pinned G92 config fingerprint mismatch" >&2
        return 2
      }
      config_path=$G92_CONFIG_PATH
      ;;
    upstream)
      [[ "$upstream_source" == 1 ]] || {
        echo "The upstream config requires TEST_UPSTREAM_SOURCE=1" >&2
        return 2
      }
      [[ -f "$UPSTREAM_CONFIG_PATH" && ! -L "$UPSTREAM_CONFIG_PATH" ]] || {
        echo "Pinned upstream config is missing or is a symlink" >&2
        return 2
      }
      [[ "$(sha256sum "$UPSTREAM_CONFIG_PATH" | awk '{print $1}')" == "$EXPECTED_UPSTREAM_CONFIG_SHA256" ]] || {
        echo "Pinned upstream config fingerprint mismatch" >&2
        return 2
      }
      config_path=$UPSTREAM_CONFIG_PATH
      ;;
    *)
      echo "Invalid TEST_CONFIG_VARIANT" >&2
      return 2
      ;;
  esac
  if [[ "$upstream_source" == 1 && "$config_variant" != upstream ]]; then
    echo "Upstream d957 source requires the pinned upstream config" >&2
    return 2
  fi
  [[ "$prefill_k" =~ ^[0-9]+$ ]] || {
    echo "Invalid TEST_PREFILL_K" >&2
    return 2
  }
  [[ "$prefill_wait_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "Invalid TEST_PREFILL_WAIT_MS" >&2
    return 2
  }
  [[ "$arm" =~ ^[a-zA-Z0-9._-]+$ ]] || {
    echo "Invalid TEST_ARM_LABEL" >&2
    return 2
  }
  [[ "$quantization" =~ ^(none|mxfp8|fp8)$ ]] || {
    echo "Invalid TEST_QUANTIZATION" >&2
    return 2
  }
  if [[ "$fso_mxfp8" == 1 && "$quantization" != none ]]; then
    echo "Custom FSO MXFP8 cannot be combined with built-in quantization" >&2
    return 2
  fi
  if [[ "$fso_source_only" == 1 && "$quantization" != none ]]; then
    echo "Instrumented BF16 source arm requires TEST_QUANTIZATION=none" >&2
    return 2
  fi
  [[ "$attention_backend" =~ ^(triton|flashinfer|triton-trtllm)$ ]] || {
    echo "Invalid TEST_ATTENTION_BACKEND" >&2
    return 2
  }
  [[ "$page_size" =~ ^(1|16|32|64)$ ]] || {
    echo "Invalid TEST_PAGE_SIZE" >&2
    return 2
  }
  if [[ "$attention_backend" == triton-trtllm && "$page_size" != 64 ]]; then
    echo "TRTLLM decode requires TEST_PAGE_SIZE=64" >&2
    return 2
  fi
  [[ "$log_level" =~ ^(debug|info|warning|error)$ ]] || {
    echo "Invalid TEST_LOG_LEVEL" >&2
    return 2
  }
  [[ "$ras_win_len" =~ ^(0|7)$ ]] || {
    echo "Invalid TEST_RAS_WIN_LEN" >&2
    return 2
  }
  [[ "$schedule_policy" =~ ^(fcfs|lpm)$ ]] || {
    echo "Invalid TEST_SCHEDULE_POLICY" >&2
    return 2
  }
  [[ "$schedule_conservativeness" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "Invalid TEST_SCHEDULE_CONSERVATIVENESS" >&2
    return 2
  }
  [[ "$torch_compile" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_TORCH_COMPILE" >&2
    return 2
  }
  [[ "$max_running_requests" =~ ^[0-9]+$ && "$max_running_requests" -gt 0 ]] || {
    echo "Invalid TEST_MAX_RUNNING_REQUESTS" >&2
    return 2
  }
  [[ "$cuda_graph_max_bs" =~ ^[0-9]+$ && "$cuda_graph_max_bs" -gt 0 ]] || {
    echo "Invalid TEST_CUDA_GRAPH_MAX_BS" >&2
    return 2
  }
  # Chỉ cho ký tự an toàn dạng "--stages.x.factory_args.y 128"; cấm mọi shell metachar.
  [[ "$extra_server_args" =~ ^[A-Za-z0-9._[:space:]-]*$ ]] || {
    echo "Invalid TEST_EXTRA_SERVER_ARGS" >&2
    return 2
  }
  [[ "$upstream_frontend_split" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_UPSTREAM_FRONTEND_SPLIT" >&2
    return 2
  }
  [[ "$upstream_vocoder_split" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_UPSTREAM_VOCODER_SPLIT" >&2
    return 2
  }
  [[ "$upstream_compile_decode" =~ ^(0|1)$ ]] || {
    echo "Invalid TEST_UPSTREAM_COMPILE_DECODE" >&2
    return 2
  }
  [[ "$upstream_preprocess_concurrency" =~ ^[0-9]+$ &&
    "$upstream_preprocess_concurrency" -gt 0 &&
    "$upstream_preprocess_concurrency" -le 256 ]] || {
    echo "Invalid TEST_UPSTREAM_PREPROCESS_CONCURRENCY" >&2
    return 2
  }
  [[ "$upstream_vocoder_batch" =~ ^[0-9]+$ &&
    "$upstream_vocoder_batch" -gt 0 &&
    "$upstream_vocoder_batch" -le 256 ]] || {
    echo "Invalid TEST_UPSTREAM_VOCODER_BATCH" >&2
    return 2
  }

  local server_args
  server_args="--model-name bosonai/higgs-audio-v3-tts-4b"
  server_args+=" --stages.2.factory_args.prefill_coalesce_requests $prefill_k"
  server_args+=" --stages.2.factory_args.prefill_coalesce_wait_ms $prefill_wait_ms"
  if [[ "$upstream_source" == 1 ]]; then
    if [[ "$attention_backend:$page_size:$schedule_policy:$schedule_conservativeness:$torch_compile:$quantization" != \
      "triton:1:fcfs:1.0:0:none" ]]; then
      echo "Upstream d957 arm currently requires triton/page1/fcfs1/torch-compile-off/BF16 defaults" >&2
      return 2
    fi
    server_args+=" --stages.2.factory_args.max_running_requests $max_running_requests"
    server_args+=" --stages.2.factory_args.cuda_graph_max_bs $cuda_graph_max_bs"
    server_args+=" --max-running-requests $max_running_requests"
    server_args+=" --cuda-graph-max-bs $cuda_graph_max_bs"
    server_args+=" --stages.audio_encoder.runtime.resources.total_gpu_memory_fraction 0.03"
    server_args+=" --stages.tts_engine.runtime.resources.total_gpu_memory_fraction 0.85"
    server_args+=" --stages.vocoder.runtime.resources.total_gpu_memory_fraction 0.10"
    server_args+=" --stages.preprocessing.factory_args.max_concurrency $upstream_preprocess_concurrency"
    server_args+=" --stages.vocoder.factory_args.vocoder_decode_batch_size $upstream_vocoder_batch"
    server_args+=" --talker-mem-fraction-static 0.85"
    if [[ "$upstream_frontend_split" == 0 ]]; then
      server_args+=" --stages.0.process pipeline"
      server_args+=" --stages.1.process pipeline"
    fi
    if [[ "$upstream_vocoder_split" == 0 ]]; then
      server_args+=" --stages.3.process pipeline"
    fi
    if [[ "$upstream_compile_decode" == 1 ]]; then
      server_args+=" --stages.3.factory_args.compile_decode true"
    else
      server_args+=" --stages.3.factory_args.compile_decode false"
    fi
  fi
  if [[ "$upstream_source" != 1 ]]; then
    if [[ "$attention_backend" == triton-trtllm ]]; then
      server_args+=" --stages.2.factory_args.server_args_overrides.attention_backend triton"
      server_args+=" --stages.2.factory_args.server_args_overrides.prefill_attention_backend triton"
      server_args+=" --stages.2.factory_args.server_args_overrides.decode_attention_backend trtllm_mha"
      server_args+=" --stages.2.factory_args.server_args_overrides.kv_cache_dtype bfloat16"
    else
      server_args+=" --stages.2.factory_args.server_args_overrides.attention_backend $attention_backend"
    fi
    server_args+=" --stages.2.factory_args.server_args_overrides.page_size $page_size"
    server_args+=" --stages.2.factory_args.server_args_overrides.schedule_policy $schedule_policy"
    server_args+=" --stages.2.factory_args.server_args_overrides.schedule_conservativeness $schedule_conservativeness"
    server_args+=" --stages.2.factory_args.server_args_overrides.max_running_requests $max_running_requests"
    server_args+=" --stages.2.factory_args.server_args_overrides.cuda_graph_max_bs $cuda_graph_max_bs"
    if [[ "$torch_compile" == 1 ]]; then
      server_args+=" --stages.2.factory_args.server_args_overrides.enable_torch_compile true"
      server_args+=" --stages.2.factory_args.server_args_overrides.torch_compile_max_bs 32"
    fi
    if [[ "$quantization" != none ]]; then
      server_args+=" --stages.2.factory_args.server_args_overrides.quantization $quantization"
      server_args+=" --stages.2.factory_args.server_args_overrides.kv_cache_dtype bfloat16"
      server_args+=" --stages.2.factory_args.server_args_overrides.dtype bfloat16"
    fi
  fi
  server_args+=" --log-level $log_level"
  if [[ -n "$extra_server_args" ]]; then
    server_args+=" $extra_server_args"
  fi

  local identity_file="$STATE_DIR/sglang.launch.$$"
  local launch_token
  launch_token=$(/root/miniconda3/bin/python3 -c 'import secrets; print(secrets.token_hex(32))')
  prepare_provisional_cleanup "$pidfile" "$identity_file" "$launch_token"
  (
    trap - INT TERM
    exec 9>&-
    export HIGGS_TEST_LAUNCH_TOKEN="$launch_token"
    export APP_ROOT VENV_ROOT MODEL_SNAPSHOT
    export HIGGS_TEST_SOURCE_ROOT="$source_root"
    export HIGGS_TEST_PYTHONPATH="$runtime_pythonpath"
    export HIGGS_TEST_LD_LIBRARY_PATH="$runtime_ld_library_path"
    export HIGGS_TEST_FSO_MXFP8="$fso_mxfp8"
    export HIGGS_TEST_SYNCFREE_LAUNCH="$syncfree_launch"
    export HIGGS_TEST_UPSTREAM_SOURCE="$upstream_source"
    export HIGGS_TEST_CONFIG_PATH="$config_path"
    export HIGGS_TEST_SERVER_ARGS="$server_args"
    export HIGGS_TEST_RAS_WIN_LEN="$ras_win_len"
    export HIGGS_TEST_QUANTIZATION="$quantization"
    exec /root/miniconda3/bin/python3 \
      "$APP_ROOT/canary/higgs-sweetspot/runtime/exec_isolated.py" \
      "$identity_file" -- \
      "$APP_ROOT/canary/higgs-sweetspot/runtime/launch_sglang.sh"
  ) >"$LOG_DIR/higgs_test_sglang_${arm}.log" 2>&1 &
  local pid=$!
  PROVISIONAL_PID=$pid
  activate_provisional_signals
  capture_provisional_identity "$pid"
  if ! wait_for_identity "$pid" "sgl-omni serve"; then
    echo "Test SGLang exited before identity verification" >&2
    return 2
  fi
  local pgid start_ticks
  pgid=$(process_group "$pid")
  start_ticks=$(process_start_ticks "$pid")
  [[ "$pgid" == "$pid" ]] || {
    echo "Test SGLang failed to enter an isolated process group" >&2
    return 2
  }
  PROVISIONAL_PGID=$pgid
  write_pid_state "$pidfile" "$pid" "$start_ticks" "$pgid" "$launch_token"
  PROVISIONAL_REMOVE_ARM_STATE=1
  printf '%s\n' "$arm" >"$STATE_DIR/arm"
  {
    printf 'arm=%s\n' "$arm"
    printf 'runtime_variant=%s\n' "$runtime_variant"
    printf 'source_root=%q\n' "$source_root"
    printf 'source_head=%s\n' "$source_head"
    printf 'source_diff_sha256=%s\n' "$source_diff"
    printf 'source_status_sha256=%s\n' "$source_status"
    printf 'higgs_fso_mxfp8=%s\n' "$fso_mxfp8"
    printf 'fso_source_only=%s\n' "$fso_source_only"
    printf 'syncfree_source=%s\n' "$syncfree_source"
    printf 'syncfree_launch=%s\n' "$syncfree_launch"
    printf 'upstream_source=%s\n' "$upstream_source"
    printf 'upstream_frontend_split=%s\n' "$upstream_frontend_split"
    printf 'upstream_vocoder_split=%s\n' "$upstream_vocoder_split"
    printf 'upstream_compile_decode=%s\n' "$upstream_compile_decode"
    printf 'upstream_preprocess_concurrency=%s\n' "$upstream_preprocess_concurrency"
    printf 'upstream_vocoder_batch=%s\n' "$upstream_vocoder_batch"
    printf 'config_variant=%s\n' "$config_variant"
    printf 'config_path=%q\n' "$config_path"
    if [[ "$config_variant" == g92 ]]; then
      printf 'config_sha256=%s\n' "$EXPECTED_G92_CONFIG_SHA256"
    elif [[ "$config_variant" == upstream ]]; then
      printf 'config_sha256=%s\n' "$EXPECTED_UPSTREAM_CONFIG_SHA256"
    else
      printf 'config_sha256=\n'
    fi
    printf 'pythonpath=%q\n' "$runtime_pythonpath"
    if [[ "$fso_mxfp8" == 1 ]]; then
      printf 'ld_library_path=%q\n' "$runtime_ld_library_path"
      printf 'fso_native_extension=%q\n' "$FSO_NATIVE_EXTENSION"
      printf 'fso_native_sha256=%s\n' "$EXPECTED_FSO_NATIVE_SHA256"
      printf 'fso_runtime_manifest=%s\n' "$EXPECTED_FSO_RUNTIME_MANIFEST"
    else
      printf 'ld_library_path=UNSET\n'
      printf 'fso_native_extension=\n'
      printf 'fso_native_sha256=\n'
      printf 'fso_runtime_manifest=\n'
    fi
    printf 'prefill_k=%s\n' "$prefill_k"
    printf 'prefill_wait_ms=%s\n' "$prefill_wait_ms"
    printf 'quantization=%s\n' "$quantization"
    printf 'attention_backend=%s\n' "$attention_backend"
    printf 'page_size=%s\n' "$page_size"
    printf 'log_level=%s\n' "$log_level"
    printf 'ras_win_len=%s\n' "$ras_win_len"
    printf 'schedule_policy=%s\n' "$schedule_policy"
    printf 'schedule_conservativeness=%s\n' "$schedule_conservativeness"
    printf 'torch_compile=%s\n' "$torch_compile"
    printf 'max_running_requests=%s\n' "$max_running_requests"
    printf 'cuda_graph_max_bs=%s\n' "$cuda_graph_max_bs"
    printf 'server_args=%q\n' "$server_args"
    if [[ "$quantization" == none ]]; then
      printf 'fp8_ignored_layers=\n'
    else
      printf 'fp8_ignored_layers=self_attn,lm_head\n'
    fi
  } >"$STATE_DIR/active-arm.env"
  for _ in $(seq 1 900); do
    if curl --fail --silent --show-error --max-time 2 \
      http://127.0.0.1:8000/health >/dev/null; then
      pid_state_matches "$pidfile" "sgl-omni serve" || {
        echo "Test SGLang identity changed during startup" >&2
        return 2
      }
      process_matches "$pid" "$MODEL_SNAPSHOT" || {
        echo "Test SGLang argv does not contain the pinned model path" >&2
        return 2
      }
      process_env_matches "$pid" "SOURCE_ROOT=$source_root" || {
        echo "Test SGLang environment does not contain the selected source root" >&2
        return 2
      }
      process_env_matches "$pid" "PYTHONPATH=$runtime_pythonpath" || {
        echo "Test SGLang environment does not contain the pinned PYTHONPATH" >&2
        return 2
      }
      process_env_matches "$pid" "HIGGS_FSO_MXFP8=$fso_mxfp8" || {
        echo "Test SGLang environment does not contain the FSO mode" >&2
        return 2
      }
      process_env_matches "$pid" "SGLANG_OMNI_SYNCFREE_LAUNCH=$syncfree_launch" || {
        echo "Test SGLang environment does not contain the sync-free launch mode" >&2
        return 2
      }
      if [[ -n "$config_path" ]]; then
        process_env_matches "$pid" "SGLANG_CONFIG=$config_path" || {
          echo "Test SGLang environment does not contain the selected config" >&2
          return 2
        }
        process_matches "$pid" "--config $config_path" || {
          echo "Test SGLang argv does not contain the selected config" >&2
          return 2
        }
      else
        process_env_absent "$pid" SGLANG_CONFIG || {
          echo "Default arm unexpectedly inherited SGLANG_CONFIG" >&2
          return 2
        }
      fi
      if [[ "$fso_mxfp8" == 1 ]]; then
        process_env_matches "$pid" "LD_LIBRARY_PATH=$runtime_ld_library_path" || {
          echo "Test SGLang environment does not contain the pinned FSO LD_LIBRARY_PATH" >&2
          return 2
        }
      else
        process_env_absent "$pid" LD_LIBRARY_PATH || {
          echo "Baseline SGLang unexpectedly inherited LD_LIBRARY_PATH" >&2
          return 2
        }
      fi
      process_matches "$pid" "--stages.2.factory_args.prefill_coalesce_requests $prefill_k" || {
        echo "Test SGLang argv does not contain the selected K value" >&2
        return 2
      }
      if [[ "$upstream_source" == 1 ]]; then
        process_matches "$pid" "--max-running-requests $max_running_requests" || {
          echo "Upstream SGLang argv does not contain max_running_requests" >&2
          return 2
        }
        process_matches "$pid" "--cuda-graph-max-bs $cuda_graph_max_bs" || {
          echo "Upstream SGLang argv does not contain cuda_graph_max_bs" >&2
          return 2
        }
        process_matches "$pid" "--talker-mem-fraction-static 0.85" || {
          echo "Upstream SGLang argv does not contain the pinned talker memory fraction" >&2
          return 2
        }
        process_matches "$pid" "--stages.preprocessing.factory_args.max_concurrency $upstream_preprocess_concurrency" || {
          echo "Upstream SGLang argv does not contain preprocessing concurrency" >&2
          return 2
        }
        process_matches "$pid" "--stages.vocoder.factory_args.vocoder_decode_batch_size $upstream_vocoder_batch" || {
          echo "Upstream SGLang argv does not contain vocoder batch size" >&2
          return 2
        }
      else
        process_matches "$pid" "--stages.2.factory_args.server_args_overrides.schedule_policy $schedule_policy" || {
          echo "Test SGLang argv does not contain the selected schedule policy" >&2
          return 2
        }
        process_matches "$pid" "--stages.2.factory_args.server_args_overrides.schedule_conservativeness $schedule_conservativeness" || {
          echo "Test SGLang argv does not contain the selected schedule conservativeness" >&2
          return 2
        }
        process_matches "$pid" "--stages.2.factory_args.server_args_overrides.max_running_requests $max_running_requests" || {
          echo "Test SGLang argv does not contain the selected max_running_requests" >&2
          return 2
        }
        process_matches "$pid" "--stages.2.factory_args.server_args_overrides.cuda_graph_max_bs $cuda_graph_max_bs" || {
          echo "Test SGLang argv does not contain the selected cuda_graph_max_bs" >&2
          return 2
        }
        if [[ "$attention_backend" == triton-trtllm ]]; then
          process_matches "$pid" "--stages.2.factory_args.server_args_overrides.decode_attention_backend trtllm_mha" || {
            echo "Test SGLang argv does not contain the TRTLLM decode backend" >&2
            return 2
          }
        fi
        if [[ "$torch_compile" == 1 ]]; then
          process_matches "$pid" "--stages.2.factory_args.server_args_overrides.enable_torch_compile true" || {
            echo "Test SGLang argv does not contain torch.compile enablement" >&2
            return 2
          }
        fi
      fi
      process_matches "$pid" "--host 127.0.0.1" || {
        echo "Test SGLang argv does not contain the pinned loopback bind" >&2
        return 2
      }
      if [[ -n "$extra_server_args" ]]; then
        process_matches "$pid" "$extra_server_args" || {
          echo "Test SGLang argv does not contain the extra server args" >&2
          return 2
        }
      fi
      process_env_matches "$pid" "HIGGS_RAS_WIN_LEN=$ras_win_len" || {
        echo "Test SGLang environment does not contain the selected RAS window" >&2
        return 2
      }
      if [[ "$quantization" != none ]]; then
        process_matches "$pid" "--stages.2.factory_args.server_args_overrides.quantization $quantization" || {
          echo "Test SGLang argv does not contain the selected quantization" >&2
          return 2
        }
        process_env_matches "$pid" "SGLANG_FP8_IGNORED_LAYERS=self_attn,lm_head" || {
          echo "Test SGLang environment does not contain the FP8 ignored-layer policy" >&2
          return 2
        }
      fi
      disarm_provisional_cleanup
      echo "Started healthy test SGLang arm $arm as PID $pid"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  local stopped=0
  stop_pidfile "$pidfile" "sgl-omni serve" "SGLang" && stopped=1
  if [[ "$stopped" == 1 ]]; then
    rm -f "$STATE_DIR/arm" "$STATE_DIR/active-arm.env"
    disarm_provisional_cleanup
  fi
  echo "Test SGLang arm $arm failed its startup health gate" >&2
  return 2
}

status() {
  require_clean_control_state
  local component pidfile expected
  for component in api sglang; do
    pidfile="$STATE_DIR/$component.pid"
    if [[ "$component" == api ]]; then
      expected="uvicorn app:app"
    else
      expected="sgl-omni serve"
    fi
    if [[ ! -f "$pidfile" ]]; then
      echo "$component: stopped"
      continue
    fi
    local STATE_PID STATE_START_TICKS STATE_PGID STATE_TOKEN
    if read_pid_state "$pidfile" &&
      [[ "$(process_start_ticks "$STATE_PID" 2>/dev/null || true)" == "$STATE_START_TICKS" ]] &&
      [[ "$(process_group "$STATE_PID" 2>/dev/null || true)" == "$STATE_PGID" ]] &&
      [[ "$STATE_PGID" == "$STATE_PID" ]] &&
      process_env_matches "$STATE_PID" "HIGGS_TEST_LAUNCH_TOKEN=$STATE_TOKEN" &&
      process_matches "$STATE_PID" "$expected"; then
      local health_url
      if [[ "$component" == api ]]; then
        health_url=http://127.0.0.1:6006/openapi.json
      else
        health_url=http://127.0.0.1:8000/health
      fi
      if curl --fail --silent --max-time 2 "$health_url" >/dev/null; then
        echo "$component: healthy PID $STATE_PID PGID $STATE_PGID"
      else
        echo "$component: running but unresponsive PID $STATE_PID PGID $STATE_PGID" >&2
        return 2
      fi
    else
      echo "$component: stale PID file" >&2
      return 2
    fi
  done
  [[ ! -f "$STATE_DIR/arm" ]] || printf 'arm: %s\n' "$(<"$STATE_DIR/arm")"
}

case "$COMMAND" in
  verify)
    verify_baseline skip-model-stamp
    verify_model_content
    echo "Verified pinned K16 source, bridge, environment, launcher, and model content"
    ;;
  verify-fso)
    verify_baseline
    verify_fso_runtime
    echo "Verified pinned FSO candidate source, Python runtime, native extension, and baseline assets"
    ;;
  start-api)
    start_api
    ;;
  stop-api)
    stop_component api "$STATE_DIR/api.pid" "uvicorn app:app" "API"
    rm -f "$STATE_DIR/active-api.env"
    ;;
  start-sglang)
    start_sglang
    ;;
  stop-sglang)
    stop_component sglang "$STATE_DIR/sglang.pid" "sgl-omni serve" "SGLang"
    rm -f "$STATE_DIR/arm" "$STATE_DIR/active-arm.env"
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {verify|verify-fso|start-api|stop-api|start-sglang|stop-sglang|status}" >&2
    exit 2
    ;;
esac
