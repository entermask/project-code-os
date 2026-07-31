#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=${APP_ROOT:?Missing APP_ROOT}
VENV_ROOT=${VENV_ROOT:?Missing VENV_ROOT}
MODEL_SNAPSHOT=${MODEL_SNAPSHOT:?Missing MODEL_SNAPSHOT}
HIGGS_TEST_SERVER_ARGS=${HIGGS_TEST_SERVER_ARGS:?Missing HIGGS_TEST_SERVER_ARGS}
HIGGS_TEST_RAS_WIN_LEN=${HIGGS_TEST_RAS_WIN_LEN:?Missing HIGGS_TEST_RAS_WIN_LEN}
HIGGS_TEST_QUANTIZATION=${HIGGS_TEST_QUANTIZATION:?Missing HIGGS_TEST_QUANTIZATION}
HIGGS_TEST_SYNCFREE_LAUNCH=${HIGGS_TEST_SYNCFREE_LAUNCH:?Missing HIGGS_TEST_SYNCFREE_LAUNCH}
HIGGS_TEST_UPSTREAM_SOURCE=${HIGGS_TEST_UPSTREAM_SOURCE:?Missing HIGGS_TEST_UPSTREAM_SOURCE}
trusted_source_root=${HIGGS_TEST_SOURCE_ROOT:?Missing HIGGS_TEST_SOURCE_ROOT}
trusted_pythonpath=${HIGGS_TEST_PYTHONPATH:?Missing HIGGS_TEST_PYTHONPATH}
trusted_fso_mxfp8=${HIGGS_TEST_FSO_MXFP8:?Missing HIGGS_TEST_FSO_MXFP8}
trusted_ld_library_path=${HIGGS_TEST_LD_LIBRARY_PATH-}
trusted_config_path=${HIGGS_TEST_CONFIG_PATH-}
launch_token=${HIGGS_TEST_LAUNCH_TOKEN:?Missing HIGGS_TEST_LAUNCH_TOKEN}

cd "$APP_ROOT"
set -a
. "$APP_ROOT/.env"
set +a
if [[ "$HIGGS_TEST_UPSTREAM_SOURCE" == 1 ]]; then
  unset SGLANG_MEM_FRACTION_STATIC
  export SGLANG_OMNI_STARTUP_TIMEOUT=1200
fi
export HIGGS_TEST_LAUNCH_TOKEN="$launch_token"
export SOURCE_ROOT="$trusted_source_root"
export PYTHONPATH="$trusted_pythonpath"
export HIGGS_FSO_MXFP8="$trusted_fso_mxfp8"
if [[ "$trusted_fso_mxfp8" == 1 ]]; then
  [[ -n "$trusted_ld_library_path" ]] || {
    echo "Missing pinned FSO LD_LIBRARY_PATH" >&2
    exit 2
  }
  export LD_LIBRARY_PATH="$trusted_ld_library_path"
else
  unset LD_LIBRARY_PATH
fi

export MODEL_PATH="$MODEL_SNAPSHOT"
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT=8000
export SGLANG_ALLOWED_LOCAL_MEDIA_PATH="${TTS_CACHE_DIR:?Missing TTS_CACHE_DIR}"
if [[ -n "$trusted_config_path" ]]; then
  export SGLANG_CONFIG="$trusted_config_path"
else
  unset SGLANG_CONFIG
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/root/autodl-tmp/hf-cache
export PYTHONUNBUFFERED=1
export CUDA_HOME="$VENV_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$VENV_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
cccl_include="$VENV_ROOT/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export CPATH="$cccl_include${CPATH:+:$CPATH}"
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-0}"
export HIGGS_RAS_WIN_LEN="$HIGGS_TEST_RAS_WIN_LEN"
export SGLANG_OMNI_SYNCFREE_LAUNCH="$HIGGS_TEST_SYNCFREE_LAUNCH"
export SGLANG_EXTRA_ARGS="$HIGGS_TEST_SERVER_ARGS"
if [[ "$HIGGS_TEST_QUANTIZATION" == none ]]; then
  unset SGLANG_FP8_IGNORED_LAYERS
else
  export SGLANG_FP8_IGNORED_LAYERS=self_attn,lm_head
fi
unset HIGGS_VOCODER_CANARY_POLICY
unset HIGGS_VOCODER_CANARY_DUMP_DIR
unset HIGGS_VOCODER_CANARY_FADE_OUT_MS
unset HIGGS_VOCODER_CANARY_FADE_CURVE
ulimit -n 65534

exec "$APP_ROOT/scripts/run_sglang.sh"
