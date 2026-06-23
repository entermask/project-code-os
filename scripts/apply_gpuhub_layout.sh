#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
HF_HOME_VALUE="${HF_HOME_VALUE:-/root/.cache/huggingface}"
TTS_CACHE_DIR_VALUE="${TTS_CACHE_DIR_VALUE:-/root/autodl-tmp/tts-cache}"
SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-/root/sglang-omni}"
export HF_HOME_VALUE TTS_CACHE_DIR_VALUE SGLANG_RUNTIME_DIR

if [ ! -f "$ENV_FILE" ]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
fi

mkdir -p "$HF_HOME_VALUE"
mkdir -p "$SGLANG_RUNTIME_DIR"
mkdir -p "$TTS_CACHE_DIR_VALUE"

python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import os
import sys

env_path = Path(sys.argv[1])
hf_home = os.environ["HF_HOME_VALUE"]
tts_cache_dir = os.environ["TTS_CACHE_DIR_VALUE"]
updates = {
    "API_TOKEN": "change-me",
    "HOST": "0.0.0.0",
    "PORT": "6006",
    "TTS_BACKEND_NAME": "bosonai/higgs-audio-v3-tts-4b",
    "MODEL_PATH": "bosonai/higgs-audio-v3-tts-4b",
    "SGLANG_BASE_URL": "http://127.0.0.1:8000",
    "SGLANG_HOST": "127.0.0.1",
    "SGLANG_PORT": "8000",
    "SGLANG_ALLOWED_LOCAL_MEDIA_PATH": tts_cache_dir,
    "TTS_CACHE_DIR": tts_cache_dir,
    "HF_HOME": hf_home,
    "STREAMED_JOB_TTL_SECONDS": "60",
    "JOB_CLEANUP_INTERVAL_SECONDS": "30",
    "STREAM_CHUNK_SIZE_BYTES": "4194304",
}

lines = env_path.read_text(encoding="utf-8").splitlines()
seen = set()
out = []
for line in lines:
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    key = line.split("=", 1)[0].strip()
    if key in updates:
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY

cat <<EOF
GPUHub layout applied:
  system:
    HF_HOME=${HF_HOME_VALUE}
    SGLang runtime target=${SGLANG_RUNTIME_DIR}
  data:
    TTS_CACHE_DIR=${TTS_CACHE_DIR_VALUE}

Install SGLang-Omni into /root/sglang-omni or symlink
/root/autodl-tmp/sglang-omni -> /root/sglang-omni for compatibility with
existing supervisor scripts.
EOF
