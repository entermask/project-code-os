#!/bin/bash
# Bundle C rolling deploy — one box. Backup -> apply -> clean restart -> verify -> smoke -> undrain.
# Auto-rollback on any engine/api failure. All output to stdout (caller tees to log).
set -u
APP=/root/autodl-tmp/Fish-Audio
K16=/root/autodl-tmp/sglang-omni-k16/sglang_omni/models/higgs_tts
STG=/root/autodl-tmp/stage_new
SUP="supervisorctl -c /root/autodl-tmp/supervisor/supervisord.conf"
SLOG=/root/autodl-tmp/logs/higgs_sglang.log
TS=preC20260731
EXPECT_APP=2728570f7e83011cf46f377cebbd4557d7e34d4d19333bd1d27b6497cc31f17f
EXPECT_BASE=d5b8ae1fe4b7a34baacd294779a8a9fa0711a58fe27bc25b725a67fa513a73b9
EXPECT_NEWSTAGES=10e01414c392c740f1565b23ccafdfce8323921d46a926d545818f00d66a52a1

step() { echo "== [$(date +%H:%M:%S)] $*"; }

# sgl-omni tu nhay sang port ngau nhien neu 8000 con TIME_WAIT — cho den khi
# bind vanilla (khong SO_REUSEADDR) thanh cong roi moi start engine.
wait_port8000() {
  for i in $(seq 1 90); do
    /root/miniconda3/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8000)); s.close()' 2>/dev/null && return 0
    sleep 2
  done
  echo "WARN: port 8000 van chua vanilla-bindable sau 180s"
  return 1
}

rollback() {
  step "ROLLBACK bat dau"
  cp "$APP/app.py.bak.$TS" "$APP/app.py" 2>/dev/null
  cp "$APP/.env.bak.$TS" "$APP/.env" 2>/dev/null
  cp "$K16/stages.py.bak.$TS" "$K16/stages.py" 2>/dev/null
  find /root/autodl-tmp/sglang-omni-k16 -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  $SUP stop higgs_sglang >/dev/null 2>&1
  for i in $(seq 1 20); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = "0" ] && break; sleep 3; done
  wait_port8000
  $SUP start higgs_sglang >/dev/null 2>&1
  $SUP restart higgs_api >/dev/null 2>&1
  for i in $(seq 1 60); do
    curl -s --max-time 3 localhost:6006/health 2>/dev/null | grep -q '"sglang_ready":true' && { echo "ROLLBACK: engine ready lai"; break; }
    sleep 5
  done
  rm -f "$APP/.draining"
  echo "ROLLED_BACK"
  exit 1
}

step "0/9 precheck"
[ "$(sha256sum $STG/app.py | cut -d' ' -f1)" = "$EXPECT_APP" ] || { echo "FAIL: staged app.py hash"; exit 1; }
[ "$(sha256sum $STG/stages_prod_c.py | cut -d' ' -f1)" = "$EXPECT_NEWSTAGES" ] || { echo "FAIL: staged stages hash"; exit 1; }
BASE=$(sha256sum $K16/stages.py | cut -d' ' -f1)
[ "$BASE" = "$EXPECT_BASE" ] || { echo "FAIL: stages.py base hash khac ($BASE) — box nay co drift, DUNG LAI"; exit 1; }
/root/autodl-tmp/sglang-omni/.venv/bin/python -c "import ast; ast.parse(open('$STG/stages_prod_c.py').read())" || { echo "FAIL: stages syntax"; exit 1; }
$APP/.venv/bin/python -c "import ast; ast.parse(open('$STG/app.py').read())" || { echo "FAIL: app syntax"; exit 1; }
$APP/.venv/bin/python -c "import audioop" || { echo "FAIL: audioop khong co trong venv bridge"; exit 1; }

step "1/9 drain (public 429, loopback van song)"
touch "$APP/.draining"
ACT=-1
for i in $(seq 1 90); do
  H=$(curl -s --max-time 3 localhost:6006/health 2>/dev/null)
  Q=$(echo "$H" | grep -o '"queued": *[0-9]*' | grep -oE '[0-9]+' | head -1)
  R=$(echo "$H" | grep -o '"running": *[0-9]*' | grep -oE '[0-9]+' | head -1)
  ACT=$(( ${Q:-0} + ${R:-0} ))
  [ "$ACT" = "0" ] && break
  sleep 5
done
echo "active jobs sau drain-wait: $ACT"
[ "$ACT" = "0" ] || { echo "FAIL: van con job sau 450s"; rm -f "$APP/.draining"; exit 1; }

step "2/9 backup"
cp -n "$APP/app.py" "$APP/app.py.bak.$TS"
cp -n "$APP/.env" "$APP/.env.bak.$TS"
cp -n "$K16/stages.py" "$K16/stages.py.bak.$TS"

step "3/9 apply files"
cp "$STG/app.py" "$APP/app.py"
cp "$STG/stages_prod_c.py" "$K16/stages.py"
find /root/autodl-tmp/sglang-omni-k16 -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

step "4/9 env edits"
setkv() { grep -q "^$1=" "$APP/.env" && sed -i "s|^$1=.*|$1=$2|" "$APP/.env" || echo "$1=$2" >> "$APP/.env"; }
setkv MAX_CONCURRENT_CHUNKS 128
setkv HIGGS_PCM_STATS_AUDIOOP all
setkv HIGGS_LANE_ADMISSION_MODE soft_reserved
setkv SGLANG_OMNI_HIGGS_REF_CODE_DISK_CACHE_DIR /root/autodl-tmp/higgs-ref-code-cache
sed -i 's|^SGLANG_EXTRA_ARGS=.*|SGLANG_EXTRA_ARGS="--stages.2.factory_args.server_args_overrides.attention_backend triton --stages.2.factory_args.server_args_overrides.max_running_requests 128 --stages.2.factory_args.server_args_overrides.cuda_graph_max_bs 128 --stages.0.factory_args.max_concurrency 128 --stages.1.factory_args.max_batch_size 128 --stages.3.factory_args.max_batch_size 128"|' "$APP/.env"
mkdir -p /root/autodl-tmp/higgs-ref-code-cache
grep -E '^(MAX_CONCURRENT_CHUNKS|HIGGS_PCM|HIGGS_LANE|SGLANG_OMNI_HIGGS|SGLANG_EXTRA_ARGS)' "$APP/.env"

step "5/9 restart engine (clean GPU release + vanilla-bind wait)"
$SUP stop higgs_sglang >/dev/null 2>&1
for i in $(seq 1 20); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = "0" ] && break; sleep 3; done
wait_port8000 || { echo "FAIL: port 8000 khong giai phong"; rollback; }
$SUP start higgs_sglang >/dev/null 2>&1
OK=0
for i in $(seq 1 60); do
  curl -s --max-time 3 localhost:6006/health 2>/dev/null | grep -q '"sglang_ready":true' && { OK=1; break; }
  ST=$($SUP status higgs_sglang | awk '{print $2}')
  echo "$ST" | grep -qE 'FATAL' && { echo "engine $ST"; break; }
  sleep 5
done
[ "$OK" = "1" ] || { echo "FAIL: engine khong ready"; tail -5 "$SLOG"; rollback; }
echo "engine ready"

step "6/9 restart api (drain_wrapper)"
$SUP restart higgs_api >/dev/null 2>&1
OK=0
for i in $(seq 1 30); do
  curl -s --max-time 3 localhost:6006/health 2>/dev/null | grep -q '"status":"ok"' && { OK=1; break; }
  sleep 2
done
[ "$OK" = "1" ] || { echo "FAIL: api khong len"; rollback; }

step "7/9 verify config"
H=$(curl -s --max-time 4 localhost:6006/health)
echo "$H" | grep -q '"max_concurrent_chunks":128' || { echo "FAIL verify: cap"; rollback; }
echo "$H" | grep -q '"lane_admission_mode":"soft_reserved"' || { echo "FAIL verify: lane mode"; rollback; }
echo "$H" | grep -q '"sglang_ready":true' || { echo "FAIL verify: sglang"; rollback; }
CAP=$(grep -o 'Capture cuda graph bs \[[0-9, ]*\]' "$SLOG" | tail -1)
echo "capture: ${CAP:-khong thay dong capture}"
echo "gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

step "8/9 loopback smoke (drain van bat)"
REFF=$(ls /root/autodl-tmp/tts-cache/ref-audio | head -1)
[ -n "$REFF" ] || { echo "FAIL: khong co ref de smoke"; rollback; }
( cd /root/autodl-tmp/tts-cache/ref-audio && setsid nohup /root/miniconda3/bin/python3 -m http.server 18998 --bind 127.0.0.1 >/dev/null 2>&1 < /dev/null & )
sleep 1
SMOKE=$($APP/.venv/bin/python /root/autodl-tmp/stage_new/smoke_probe.py "http://127.0.0.1:18998/$REFF" 2>&1 | tail -1)
pkill -f 'http.server 18998' 2>/dev/null
echo "smoke: $SMOKE"
echo "$SMOKE" | grep -q '^SMOKE_OK' || { echo "FAIL: smoke"; rollback; }

step "9/9 undrain — box vao lai rotation"
rm -f "$APP/.draining"
curl -s --max-time 3 localhost:6006/health | grep -o '"status":"[a-z]*"' | head -1
echo "BUNDLE_C_DEPLOY_OK"
