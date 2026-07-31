#!/bin/bash
# Chuyen tu Go supervisord sang Python supervisor (chuan prod), roi verify + smoke.
set -u
APP=/root/autodl-tmp/Fish-Audio
CONF=/root/autodl-tmp/supervisor/supervisord.conf
STG=/root/autodl-tmp/stage_prodstd
step() { echo "== [$(date +%H:%M:%S)] $*"; }

step "1/5 shutdown Go supervisord (keo theo services)"
/usr/bin/supervisord ctl -c $CONF shutdown 2>/dev/null || true
sleep 3
pkill -f '/usr/bin/supervisord -c' 2>/dev/null || true
for i in $(seq 1 20); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = "0" ] && break; sleep 3; done
echo "gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
rm -f /root/autodl-tmp/supervisor/supervisor.sock /root/autodl-tmp/supervisor/supervisord.pid

step "2/5 vanilla-bind wait port 8000"
for i in $(seq 1 90); do
  /root/miniconda3/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()' 2>/dev/null && break
  sleep 2
done

step "3/5 start Python supervisord"
/usr/local/bin/supervisord -c $CONF
OK=0
for i in $(seq 1 60); do
  curl -s --max-time 3 localhost:6006/health 2>/dev/null | grep -q '"sglang_ready":true' && { OK=1; break; }
  ST=$(/usr/local/bin/supervisorctl -c $CONF status higgs_sglang 2>/dev/null | awk '{print $2}')
  echo "$ST" | grep -qE 'FATAL' && { echo "engine FATAL"; break; }
  sleep 5
done
[ "$OK" = "1" ] || { echo "FAIL: engine khong ready"; tail -5 /root/autodl-tmp/logs/higgs_sglang.log; exit 1; }
echo "engine ready"

step "4/5 verify + smoke"
H=$(curl -s --max-time 4 localhost:6006/health)
echo "$H" | grep -q '"max_concurrent_chunks":128' || { echo "FAIL cap"; exit 1; }
echo "$H" | grep -q '"lane_admission_mode":"soft_reserved"' || { echo "FAIL lane"; exit 1; }
CAP=$(grep -o 'Capture cuda graph bs \[[0-9, ]*\]' /root/autodl-tmp/logs/higgs_sglang.log | tail -1)
echo "capture: ${CAP:-khong thay}"
REFF=$(ls /root/autodl-tmp/tts-cache/ref-audio 2>/dev/null | head -1)
( cd /root/autodl-tmp/tts-cache/ref-audio && setsid nohup /root/miniconda3/bin/python3 -m http.server 18998 --bind 127.0.0.1 >/dev/null 2>&1 < /dev/null & )
sleep 1
SMOKE=$($APP/.venv/bin/python $STG/smoke_probe.py "http://127.0.0.1:18998/$REFF" 2>&1 | tail -1)
pkill -f 'http.server 18998' 2>/dev/null
echo "smoke: $SMOKE"
echo "$SMOKE" | grep -q '^SMOKE_OK' || { echo "FAIL smoke"; exit 1; }

step "5/5 status"
/usr/local/bin/supervisorctl -c $CONF status
echo "SWITCH_SUPERVISOR_OK"
