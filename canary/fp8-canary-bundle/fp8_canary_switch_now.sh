#!/bin/bash
# fp8_canary_switch_now.sh — PROD box 22405, owner asked 29/09/2026 to switch immediately (run 3, no quiet wait). Switches the engine to
# FP8 (env block in Fish-Audio/.env + sglang-omni-k16 -> sglang-omni-fp8-canary) and restarts ONLY higgs_sglang.
# Auto-rollback if: not ready in 6 min, FP8 not confirmed in the engine, >= 3 job failures (counted from the moment the
# new engine is ready) or > 2 min not ready during the first 30 min. State in /root/autodl-tmp/fp8-canary.state, log in logs/fp8_canary.log.
. /root/autodl-tmp/fp8-canary-bundle/fp8_canary_lib.sh
echo "SWITCHING_NOW since $(date '+%F %T')" > $STATE
log "run 3: owner asked to switch NOW (no quiet wait)"
h=$(health); a=$(hfield "$h" active_tts_jobs); o=$(hfield "$h" outstanding_chunks); r=$(sg_running "$h")
f0=$(jobs_failed "$h"); T0=$(date '+%F %T')
log "switching to FP8 now (jobs=$a outstanding=$o running=$r failed_so_far=$f0)"
[ -f $ENVF.bak-fp8canary-20260928 ] || cp -p $ENVF $ENVF.bak-fp8canary-20260928
env_add; link_to $CANARY
log "env block added, $(readlink $LINK) ; restarting higgs_sglang"
restart_engine
if ! wait_ready 360; then rollback "sglang not ready 6 min after the FP8 restart"; exit 1; fi
pid=$(pgrep -f "sgl-omni serve" | head -1)
fp8=$(grep -h "HIGGS_FSO_MXFP8=all: patched 36" $ELOG $ELOG.1 2>/dev/null | tail -1 | cut -c1-19)
envok=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -c -E '^(HIGGS_FSO_MXFP8=all|HIGGS_DECODE_NOSYNC=1|HIGGS_SAMPLER_TAIL=1|FSO_TUNE_TABLE=9728)')
log "ready. engine pid=$pid, last FP8 patch line at '${fp8:-none}' (restart at $T0), FP8 env vars in engine: $envok/4, tune table lines: $(tail -c 20000000 $ELOG | grep -c 'FSO_TUNE_TABLE: 13 entries')"
if [ -z "$fp8" ] || [[ "$fp8" < "${T0:0:16}" ]] || [ "$envok" != "4" ]; then rollback "FP8 not confirmed in the new engine"; exit 1; fi
f1=$(jobs_failed "$(health)"); log "job failures during the restart window: $((f1 - f0)) ($f0 -> $f1)"; f0=$f1
echo "CANARY_VERIFYING since $(date '+%F %T')" > $STATE
notready=0
for i in $(seq 1 60); do   # 30 min guard
  sleep 30; h=$(health)
  if echo "$h" | grep -q '"sglang_ready":true'; then notready=0; else notready=$((notready+30)); fi
  f=$(jobs_failed "$h")
  [ $((i % 10)) = 0 ] && log "guard ${i}/60: failed $f0 -> $f, active_jobs $(hfield "$h" active_tts_jobs), outstanding $(hfield "$h" outstanding_chunks)"
  [ "$notready" -gt 120 ] && { rollback "engine not ready > 2 min during the guard"; exit 1; }
  [ -n "$f" ] && [ -n "$f0" ] && [ $((f - f0)) -ge 3 ] && { rollback "$((f - f0)) job failures in the first 30 min"; exit 1; }
done
log "CANARY ACTIVE: 30 min guard passed (failed $f0 -> $f)"
echo "CANARY_ACTIVE since $T0 (guard passed $(date '+%F %T'))" > $STATE
