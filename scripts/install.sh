#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/sglang-tts-api}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN not found. Install Python 3.12 or set PYTHON_BIN=/path/to/python3.12." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
python -m pip install -e ".[dev]"

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Installed SGLang TTS API dependencies in $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
