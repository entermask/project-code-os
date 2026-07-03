#!/bin/bash
# ============================================================================
# Manual FULL restart of Higgs TTS on a GPUHub bridge.
# Frees the GPU (kills stale / ORPHANED sglang workers that survive a normal
# restart and hold GPU memory -> CUDA OOM / crash-loop), then starts
# supervisord (-> higgs_sglang + higgs_api) + the keepalive watchdog, and
# waits for sglang to become ready.
#   Run:  bash /root/autodl-tmp/bin/higgs-restart.sh
# ============================================================================
CONF=/root/autodl-tmp/supervisor/supervisord.conf
SUPC=$(command -v supervisorctl || echo /usr/bin/supervisorctl)

echo "[1/4] stop programs + supervisord + watchdog"
"$SUPC" -c "$CONF" stop all 2>/dev/null
pkill -9 -f 'higgs-keepalive.sh' 2>/dev/null
pkill -9 -f "supervisord -c $CONF" 2>/dev/null
sleep 2

echo "[2/4] free GPU: kill ALL sglang-omni + uvicorn procs (incl orphaned workers)"
pkill -9 -f '/root/autodl-tmp/sglang-omni/.venv/bin/python3' 2>/dev/null
pkill -9 -f '/root/autodl-tmp/Fish-Audio/.venv/bin/uvicorn' 2>/dev/null
pkill -9 -f 'uvicorn app:app' 2>/dev/null
sleep 5
echo "    GPU now: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    still holding: /'
echo "    (GPU mem should be near 0 / empty here; if not, re-run this script)"

echo "[3/4] start supervisord + watchdog"
rm -f /root/autodl-tmp/supervisor/supervisor.sock /root/autodl-tmp/supervisor/supervisord.pid 2>/dev/null
bash /root/autodl-tmp/bin/higgs-up.sh 2>/dev/null || {
  SUPD=$(command -v supervisord || echo /usr/bin/supervisord)
  setsid -f "$SUPD" -c "$CONF" </dev/null >>/root/autodl-tmp/logs/keepalive.log 2>&1
  setsid -f bash /root/autodl-tmp/bin/higgs-keepalive.sh </dev/null >/dev/null 2>&1
}
sleep 3
"$SUPC" -c "$CONF" status 2>&1

echo "[4/4] waiting for sglang to load (model ~2-4 min)..."
for i in $(seq 1 30); do
  curl -s --max-time 5 localhost:6006/health 2>/dev/null | grep -q '"sglang_ready":true' && { echo "    READY after ~$((i*10))s"; break; }
  sleep 10
done
curl -s --max-time 5 localhost:6006/health 2>/dev/null | grep -o '"sglang_ready":[a-z]*' || echo "    not ready yet - check: $SUPC -c $CONF status ; tail /root/autodl-tmp/logs/higgs_sglang.log"
echo "== restart done =="
