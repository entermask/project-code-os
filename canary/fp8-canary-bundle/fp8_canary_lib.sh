# fp8_canary_lib.sh — sourced by the FP8 canary scripts on PROD box 22405 (owner-approved 28/09/2026).
A=/root/autodl-tmp
ENVF=$A/Fish-Audio/.env
CONF=$A/supervisor/supervisord.conf
SUPD=/usr/bin/supervisord
LINK=$A/sglang-omni-k16
ORIG=$A/sglang-omni-k16-e161d2f
CANARY=$A/sglang-omni-fp8-canary
STATE=$A/fp8-canary.state
LOG=$A/logs/fp8_canary.log
ELOG=$A/logs/higgs_sglang.log
TUNE="9728,2560,17,32,32,64,4,3;9728,2560,33,64,64,64,4,3;9728,2560,65,96,32,128,4,2;9728,2560,97,128,64,128,2,3;4096,2560,9,16,16,128,4,4;4096,2560,17,32,32,64,4,2;4096,2560,33,64,64,64,4,2;4096,2560,65,96,32,128,4,2;4096,2560,97,128,64,128,2,3;2560,6144,17,32,32,64,4,1;2560,6144,97,128,128,128,2,3;2560,19456,17,32,32,64,4,1;2560,19456,33,64,64,64,4,1"
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
health(){ curl -s -m 5 http://127.0.0.1:6006/health; }
hfield(){ echo "$1" | grep -o "\"$2\":[0-9a-z]*" | head -1 | cut -d: -f2; }
sg_running(){ local r; r=$(echo "$1" | grep -o '"request_states":{[^}]*}' | grep -o '"running":[0-9]*' | cut -d: -f2); echo "${r:-0}"; }
jobs_failed(){ echo "$1" | grep -o '"tts_jobs":{[^}]*}' | grep -o '"failed":[0-9]*' | cut -d: -f2; }
wait_ready(){ # $1 = seconds; returns 0 when sglang_ready:true
  local t=0; while [ $t -lt $1 ]; do health | grep -q '"sglang_ready":true' && return 0; sleep 5; t=$((t+5)); done; return 1; }
restart_engine(){ # stop, free orphaned GPU memory if any, start
  $SUPD ctl -c $CONF stop higgs_sglang >> $LOG 2>&1; sleep 5
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -gt 2000 ]; then
    log "GPU still holds ${used} MiB after stop -> killing orphaned sglang workers"
    pkill -9 -f '/root/autodl-tmp/sglang-omni/.venv/bin/python' 2>/dev/null; sleep 5
    log "GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  fi
  # port 8000 in TIME_WAIT -> sgl-omni silently binds a random port (see bin/higgs-keepalive.sh): wait until bindable
  for _ in $(seq 1 60); do
    /root/miniconda3/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()' 2>/dev/null && break
    sleep 2
  done
  $SUPD ctl -c $CONF start higgs_sglang >> $LOG 2>&1
}
env_add(){
  grep -q '^# >>> FP8 canary' $ENVF && return 0
  cat >> $ENVF <<EOB
# >>> FP8 canary 28/09/2026 (deploy-bundle-2026-09-28-v2, cau hinh "FP8 du"). Rollback: bash /root/autodl-tmp/fp8-canary-bundle/fp8_canary_rollback.sh
HIGGS_FSO_MXFP8=all
HIGGS_DECODE_NOSYNC=1
HIGGS_SAMPLER_TAIL=1
FSO_TUNE_TABLE="$TUNE"
# <<< FP8 canary
EOB
}
env_remove(){ sed -i '/^# >>> FP8 canary/,/^# <<< FP8 canary/d' $ENVF; }
link_to(){ rm -f $LINK.new; ln -s "$1" $LINK.new && mv -T $LINK.new $LINK; }
rollback(){ # $1 = reason
  log "ROLLBACK: $1"
  env_remove; link_to $ORIG
  restart_engine
  if wait_ready 420; then log "rollback: sglang_ready again (config restored)"; echo "ROLLED_BACK $(date '+%F %T') $1" > $STATE
  else log "rollback: sglang NOT ready after 7 min -> run bash /root/autodl-tmp/bin/higgs-restart.sh"; echo "ROLLBACK_NOT_READY $(date '+%F %T') $1" > $STATE; fi
}
