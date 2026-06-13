#!/usr/bin/env bash
set -e

# Repo root = this script's directory. Keeps the launcher portable across machines.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# mise instala `openclaw` como shim; un bash no-login no lo trae en PATH.
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"

# Optional startup delay — useful when launched from Hyprland autostart so the
# desktop / OpenClaw gateway can come up first. Manual runs: leave at 0.
sleep "${BMO_STARTUP_DELAY:-0}"

PORT="${BMO_PORT:-8765}"
CHROMIUM_BIN="${CHROMIUM_BIN:-/usr/bin/chromium}"

# ── dependencies: OpenClaw gateway + Ollama ────────────────────────────────────
# BMO talks to OpenClaw (systemd user service), which talks to local Ollama.
if ! systemctl --user is-active --quiet openclaw-gateway.service; then
  echo "[BMO] starting openclaw-gateway.service…"
  systemctl --user start openclaw-gateway.service || true
fi

# ── Conversación nueva en cada arranque ─────────────────────────────────────────
# BMO usa una sesión DEDICADA (aislada del 'main' de Telegram) y la RESETEA en cada
# lanzamiento: cada vez que inicias BMO empiezas una charla en blanco, sin arrastrar
# historia vieja (ni un estado corrupto como el bucle de auto-reenvío). El retry
# también sirve de espera a que el gateway termine de levantar.
export OPENCLAW_SESSION_KEY="${OPENCLAW_SESSION_KEY:-agent:main:voice}"
echo "[BMO] esperando gateway y reseteando sesión $OPENCLAW_SESSION_KEY…"
for _ in $(seq 1 30); do
  if openclaw gateway call sessions.reset --params "{\"key\":\"$OPENCLAW_SESSION_KEY\"}" >/dev/null 2>&1; then
    echo "[BMO] sesión reseteada — conversación nueva"
    break
  fi
  sleep 0.5
done

# Ollama runs manually on this machine (not the system service — models in ~/.ollama).
if ! curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; then
  # OLLAMA_CONTEXT_LENGTH lives as a fish universal var; pull it in for bash.
  if [ -z "${OLLAMA_CONTEXT_LENGTH:-}" ] && command -v fish >/dev/null; then
    OLLAMA_CONTEXT_LENGTH="$(fish -c 'echo -n $OLLAMA_CONTEXT_LENGTH' 2>/dev/null || true)"
  fi
  [ -n "${OLLAMA_CONTEXT_LENGTH:-}" ] && export OLLAMA_CONTEXT_LENGTH
  # Keep the model loaded (no 5-min unload → no 7s cold reload) and fit the
  # full 16k context 100% on the RTX 5050 via flash attention + q8_0 KV cache.
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
  export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
  export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
  echo "[BMO] starting ollama serve (ctx=${OLLAMA_CONTEXT_LENGTH:-default}, keep_alive=$OLLAMA_KEEP_ALIVE)…"
  nohup ollama serve >/tmp/bmo-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

# Activate the backend venv (created by setup.sh).
source "$APP_DIR/.venv/bin/activate"

# Restart backend. SIGKILL after a grace period — the sounddevice thread keeps
# uvicorn from exiting on SIGTERM, and stale backends hold the USB mic hostage.
pkill -f 'uvicorn backend:app' || true
sleep 1.5
pkill -9 -f 'uvicorn backend:app' || true
sleep 0.5
# Turnos con visión de pantalla (bmo-screen + gemma4) pueden tardar ~40-50s;
# el default de 45s del backend quedaba justo.
export OPENCLAW_TIMEOUT_MS="${OPENCLAW_TIMEOUT_MS:-90000}"
nohup python -m uvicorn backend:app --host 127.0.0.1 --port "$PORT" \
  >/tmp/bmo-orb-backend.log 2>&1 &

# Wait until the server answers before opening the kiosk window.
backend_up=0
for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then backend_up=1; break; fi
  sleep 0.5
done
if [ "$backend_up" -ne 1 ]; then
  echo "[BMO] $(date '+%H:%M:%S') ERROR: backend no responde en :$PORT tras 20s — revisa /tmp/bmo-orb-backend.log"
fi

# Relaunch semantics: kill any previous kiosk window using this profile so the
# new one starts clean (a half-dead instance would block the SingletonLock).
pkill -f 'user-data-dir=/tmp/bmo-orb-profile' 2>/dev/null && sleep 1 || true

# Chromium occasionally SIGABRTs right at startup (wayland/NVIDIA); without a
# retry the launcher used to die silently and no window appeared. Vulkan is
# explicitly incompatible with --ozone-platform=wayland, so disable it.
launch_kiosk() {
  "$CHROMIUM_BIN" \
    --noerrdialogs \
    --disable-infobars \
    --autoplay-policy=no-user-gesture-required \
    --password-store=basic \
    --disable-features=Vulkan \
    --user-data-dir=/tmp/bmo-orb-profile \
    --app="http://127.0.0.1:$PORT" \
    --window-size=720,720 \
    --window-position=600,140
}

for attempt in 1 2 3; do
  echo "[BMO] $(date '+%H:%M:%S') abriendo kiosk (intento $attempt/3)…"
  start_ts=$(date +%s)
  code=0; launch_kiosk || code=$?
  ran=$(( $(date +%s) - start_ts ))
  if [ "$ran" -ge 10 ]; then
    # Lived long enough to be a real session; exiting now means the user closed it.
    echo "[BMO] $(date '+%H:%M:%S') kiosk cerrado (code=$code tras ${ran}s)"
    exit 0
  fi
  echo "[BMO] $(date '+%H:%M:%S') kiosk murió en ${ran}s (code=$code) — reintentando"
  sleep 1
done
echo "[BMO] $(date '+%H:%M:%S') ERROR: el kiosk no sobrevivió 3 intentos — revisa este log"
exit 1
