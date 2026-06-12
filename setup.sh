#!/usr/bin/env bash
# One-shot setup for BMO Orb on Arch / CachyOS.
# Idempotent: safe to re-run. Creates two venvs (backend + whisper), downloads
# the whisper-small model, and lays out the wakeword model directory.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# Python 3.12 — the audio stack (onnxruntime / ctranslate2) lacks 3.14 wheels.
PY="${PYTHON_BIN:-$(command -v python3.12 || true)}"
if [ -z "$PY" ] && command -v mise >/dev/null 2>&1; then
  mise install python@3.12.13 >/dev/null 2>&1 || true
  PY="$(mise which python 2>/dev/null || echo "$HOME/.local/share/mise/installs/python/3.12.13/bin/python")"
fi
[ -x "$PY" ] || { echo "Need Python 3.12 (set PYTHON_BIN=...)"; exit 1; }
echo "Using Python: $("$PY" --version)"

# System dependency.
if ! pacman -Qq portaudio >/dev/null 2>&1; then
  echo ">> Installing portaudio (needs sudo)"; sudo pacman -S --needed --noconfirm portaudio
fi

# Backend venv.
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Whisper venv (isolated, called via subprocess by the backend).
[ -d .venv-audio ] || "$PY" -m venv .venv-audio
.venv-audio/bin/python -m pip install --upgrade pip
.venv-audio/bin/python -m pip install faster-whisper huggingface_hub

# Whisper-small model (CTranslate2 format).
if [ ! -f models/whisper-small/model.bin ]; then
  .venv-audio/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Systran/faster-whisper-small', local_dir='models/whisper-small')
PY
fi

# Wakeword: shared feature models ship with openwakeword; copy them in.
OWW="$(.venv/bin/python -c 'import os,openwakeword;print(os.path.join(os.path.dirname(openwakeword.__file__),"resources","models"))')"
mkdir -p models/openWakeWordModel/resources
cp -n "$OWW/melspectrogram.onnx"  models/openWakeWordModel/resources/
cp -n "$OWW/embedding_model.onnx" models/openWakeWordModel/resources/
# Fallback test wakeword so the app runs before the bespoke "hey bmo" is trained.
cp -n "$OWW/hey_jarvis_v0.1.onnx" models/openWakeWordModel/hey_jarvis_v0.1.onnx

# Voice-assistant tools → ~/.local/bin (the agent invokes them by absolute path).
mkdir -p "$HOME/.local/bin"
for t in "$APP_DIR"/tools/bmo-*; do
  ln -sf "$t" "$HOME/.local/bin/$(basename "$t")"
done
echo "Tools linked into ~/.local/bin: $(ls "$APP_DIR"/tools | tr '\n' ' ')"

echo
echo "Setup complete. Still required on your side:"
echo "  * Drop the trained wakeword at models/openWakeWordModel/hey_bee_moh.onnx"
echo "    (otherwise it falls back to hey_jarvis as the trigger word)."
echo "  * Configure OpenClaw (the LLM backend) — see README.md."
echo
echo "Run:  ./launch_kiosk.sh"
