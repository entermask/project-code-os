#!/bin/bash
# Chuyen server test ve chuan PROD: supervisord + bin launchers + drain_wrapper
# + engine k16 voi stages bundle C. .env cua test duoc edit in-place theo cac
# key cau hinh chuan prod (KHONG dong token — token pool la viec cua operator).
set -u
APP=/root/autodl-tmp/Fish-Audio
K16=/root/autodl-tmp/sglang-omni-k16/sglang_omni/models/higgs_tts
STG=/root/autodl-tmp/stage_prodstd
CTL=$APP/canary/higgs-sweetspot/runtime/higgs-test-control.sh
SUP="supervisorctl -c /root/autodl-tmp/supervisor/supervisord.conf"
TS=pretostd20260731
EXPECT_BASE=d5b8ae1fe4b7a34baacd294779a8a9fa0711a58fe27bc25b725a67fa513a73b9
EXPECT_NEWSTAGES=10e01414c392c740f1565b23ccafdfce8323921d46a926d545818f00d66a52a1
MODEL_HF=/root/autodl-tmp/hf-cache/models--bosonai--higgs-audio-v3-tts-4b
step() { echo "== [$(date +%H:%M:%S)] $*"; }

step "0/8 precheck"
[ "$(sha256sum $STG/stages_prod_c.py | cut -d' ' -f1)" = "$EXPECT_NEWSTAGES" ] || { echo "FAIL staged stages hash"; exit 1; }
[ "$(sha256sum $K16/stages.py | cut -d' ' -f1)" = "$EXPECT_BASE" ] || { echo "FAIL k16 stages base hash"; exit 1; }
[ -d "$MODEL_HF/snapshots" ] || { echo "FAIL model hf-cache missing"; exit 1; }
[ -d /root/autodl-tmp/tts-cache ] || { echo "FAIL tts-cache dir missing"; exit 1; }
[ -f $STG/supervisord.conf ] && [ -f $STG/higgs-api.sh ] && [ -f $STG/higgs-sglang.sh ] || { echo "FAIL staged prodstd files"; exit 1; }
grep -q '^API_TOKEN=' "$APP/.env" || { echo "FAIL .env test thieu API_TOKEN"; exit 1; }

step "1/8 stop controller-managed services"
$CTL stop-api >/dev/null 2>&1
$CTL stop-sglang >/dev/null 2>&1
sleep 2
for f in sglang api; do
  P=/root/autodl-tmp/higgs-test-control/$f.pid
  if [ -f "$P" ]; then read -r pid _ < "$P"; kill -0 "$pid" 2>/dev/null || rm -f "$P"; fi
done
for i in $(seq 1 20); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = "0" ] && break; sleep 3; done
echo "gpu procs con lai: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"

step "2/8 backup"
cp -n "$APP/.env" "$APP/.env.bak.$TS"
cp -n "$K16/stages.py" "$K16/stages.py.bak.$TS"

step "3/8 apply engine stages bundle C vao k16 tree"
cp $STG/stages_prod_c.py "$K16/stages.py"
find /root/autodl-tmp/sglang-omni-k16 -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
/root/autodl-tmp/sglang-omni/.venv/bin/python -c "import ast; ast.parse(open('$K16/stages.py').read())" || { echo "FAIL stages syntax"; exit 1; }

step "4/8 install prod-standard layout + env edits in-place"
mkdir -p /root/autodl-tmp/bin /root/autodl-tmp/supervisor /root/autodl-tmp/logs /root/autodl-tmp/higgs-ref-code-cache
cp $STG/higgs-api.sh /root/autodl-tmp/bin/higgs-api.sh
cp $STG/higgs-sglang.sh /root/autodl-tmp/bin/higgs-sglang.sh
cp $STG/supervisord.conf /root/autodl-tmp/supervisor/supervisord.conf
chmod +x /root/autodl-tmp/bin/higgs-api.sh /root/autodl-tmp/bin/higgs-sglang.sh
setkv() { grep -q "^$1=" "$APP/.env" && sed -i "s|^$1=.*|$1=$2|" "$APP/.env" || echo "$1=$2" >> "$APP/.env"; }
setkv HOST 0.0.0.0
setkv PORT 6006
setkv MAX_CONCURRENT_CHUNKS 128
setkv HIGGS_PCM_STATS_AUDIOOP all
setkv HIGGS_LANE_ADMISSION_MODE soft_reserved
setkv SGLANG_OMNI_HIGGS_REF_CODE_DISK_CACHE_DIR /root/autodl-tmp/higgs-ref-code-cache
setkv TTS_CACHE_DIR /root/autodl-tmp/tts-cache
setkv SGLANG_ALLOWED_LOCAL_MEDIA_PATH /root/autodl-tmp/tts-cache
sed -i 's|^SGLANG_EXTRA_ARGS=.*|SGLANG_EXTRA_ARGS="--stages.2.factory_args.server_args_overrides.attention_backend triton --stages.2.factory_args.server_args_overrides.max_running_requests 128 --stages.2.factory_args.server_args_overrides.cuda_graph_max_bs 128 --stages.0.factory_args.max_concurrency 128 --stages.1.factory_args.max_batch_size 128 --stages.3.factory_args.max_batch_size 128"|' "$APP/.env"
grep -q '^SGLANG_EXTRA_ARGS=' "$APP/.env" || echo 'SGLANG_EXTRA_ARGS="--stages.2.factory_args.server_args_overrides.attention_backend triton --stages.2.factory_args.server_args_overrides.max_running_requests 128 --stages.2.factory_args.server_args_overrides.cuda_graph_max_bs 128 --stages.0.factory_args.max_concurrency 128 --stages.1.factory_args.max_batch_size 128 --stages.3.factory_args.max_batch_size 128"' >> "$APP/.env"
grep -E '^(HOST|PORT|MAX_CONCURRENT_CHUNKS|HIGGS_PCM|HIGGS_LANE|SGLANG_OMNI_HIGGS|SGLANG_EXTRA_ARGS)' "$APP/.env"

step "5/8 model symlink cho path chuan prod"
mkdir -p /root/.cache/huggingface/hub
ln -sfn "$MODEL_HF" /root/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b
SNAP=/root/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/snapshots/a7f70853f163c4cccbdd27ce9a80dd97961fc581
[ "$(readlink "$SNAP/model.safetensors")" = "../../blobs/2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a" ] || { echo "FAIL model symlink check"; exit 1; }
[ "$(stat -Lc '%s' "$SNAP/model.safetensors")" = "9309834930" ] || { echo "FAIL model size check"; exit 1; }

step "6/8 start supervisord"
if [ -S /root/autodl-tmp/supervisor/supervisor.sock ]; then
  $SUP reread >/dev/null 2>&1; $SUP update >/dev/null 2>&1; $SUP restart higgs_sglang higgs_api >/dev/null 2>&1
else
  /usr/bin/supervisord -c /root/autodl-tmp/supervisor/supervisord.conf
fi
OK=0
for i in $(seq 1 60); do
  curl -s --max-time 3 localhost:6006/health 2>/dev/null | grep -q '"sglang_ready":true' && { OK=1; break; }
  ST=$($SUP status higgs_sglang 2>/dev/null | awk '{print $2}')
  echo "$ST" | grep -qE 'FATAL' && { echo "engine FATAL"; break; }
  sleep 5
done
[ "$OK" = "1" ] || { echo "FAIL: engine khong ready"; tail -5 /root/autodl-tmp/logs/higgs_sglang.log; exit 1; }
echo "engine ready"

step "7/8 verify config + smoke"
H=$(curl -s --max-time 4 localhost:6006/health)
echo "$H" | grep -q '"max_concurrent_chunks":128' || { echo "FAIL cap"; exit 1; }
echo "$H" | grep -q '"lane_admission_mode":"soft_reserved"' || { echo "FAIL lane mode"; exit 1; }
CAP=$(grep -o 'Capture cuda graph bs \[[0-9, ]*\]' /root/autodl-tmp/logs/higgs_sglang.log | tail -1)
echo "capture: ${CAP:-khong thay}"
REFF=$(ls /root/autodl-tmp/tts-cache/ref-audio 2>/dev/null | head -1)
if [ -n "$REFF" ]; then
  ( cd /root/autodl-tmp/tts-cache/ref-audio && setsid nohup /root/miniconda3/bin/python3 -m http.server 18998 --bind 127.0.0.1 >/dev/null 2>&1 < /dev/null & )
  sleep 1
  SMOKE=$($APP/.venv/bin/python $STG/smoke_probe.py "http://127.0.0.1:18998/$REFF" 2>&1 | tail -1)
  pkill -f 'http.server 18998' 2>/dev/null
  echo "smoke: $SMOKE"
  echo "$SMOKE" | grep -q '^SMOKE_OK' || { echo "FAIL smoke"; exit 1; }
else
  echo "WARN: ref-audio rong, bo qua smoke"
fi

step "8/8 xong"
$SUP status
echo "TEST_TO_PRODSTD_OK"
