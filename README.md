# BMO Orb

Voice-assistant "orb" for a small kiosk display. Pipeline:

```
Alt+\ (push-to-talk: Hyprland bind o tecla en el kiosk → POST /trigger) →
mic (sounddevice) record + VAD → faster-whisper (transcribe) →
OpenClaw chat.send (LLM via Ollama) → Piper TTS (voz) →
WebSocket → orb UI (blob amorfo de partículas blancas, Chromium kiosk)
```

No hay wakeword: se activa con **Alt+\\** (toggle — segunda pulsación cancela).
Single-shot: una consulta → una respuesta → vuelta a idle (sin turnos).

## Herramientas de voz (`tools/`)

Scripts deterministas que cubren las debilidades del LLM local (8B): el agente
solo tiene que invocarlos vía `exec` y leer el resultado. `setup.sh` los enlaza
en `~/.local/bin/`; el agente los llama por ruta absoluta.

| Herramienta | Función |
|-------------|---------|
| `bmo-spotify` | reproducir canción por nombre (resuelve el track sin API key), pause/resume/next/prev/status |
| `bmo-screen` | screenshot temporal (se sobreescribe y borra) + análisis con modelo de visión local (gemma4 vía Ollama) |
| `bmo-system-inventory` | resumen del sistema: OS, CPU, RAM, GPU, disco, uptime; modos `disco` y `procesos` |
| `bmo-weather` | clima actual o `pronostico` 3 días (wttr.in, sin API key) |
| `bmo-time` | hora/fecha en español, local o de otra ciudad (`bmo-time tokio`) |
| `bmo-reminder` | recordatorios/temporizadores con systemd-run; al cumplirse: notificación + BMO lo dice por voz |
| `bmo-calc` | aritmética, porcentajes y conversión de monedas (open.er-api.com, sin API key) |
| `bmo-net` | IP local/pública/gateway, dispositivos en la LAN (ARP), puertos locales escuchando |
| `bmo-serial` | puertos serie USB conectados (Arduino/ESP32) con fabricante y modelo |

Las instrucciones de cuándo usar cada una viven en el `AGENTS.md` del workspace
de OpenClaw (fuera de este repo).

## Uso diario

```bash
bmo                 # ~/.local/bin/bmo → launch_kiosk.sh
```

`launch_kiosk.sh` arranca (si no están corriendo) `openclaw-gateway.service`
(systemd user) y `ollama serve` (manual, hereda `OLLAMA_CONTEXT_LENGTH` de fish),
luego el backend FastAPI y la ventana kiosk de Chromium.

## Setup (Arch / CachyOS)

```bash
./setup.sh          # builds both venvs, downloads whisper-small, lays out models
./launch_kiosk.sh   # starts deps + backend + opens the kiosk window
```

`setup.sh` is idempotent. It uses **Python 3.12** (the audio stack has no 3.14
wheels yet) via `mise`, and installs `portaudio` with pacman.

## What you must provide

1. **OpenClaw** — the LLM backend. The orb calls `openclaw gateway call
   chat.send` against the session key `agent:main:main` and reads replies from
   the OpenClaw session file. OpenClaw must be configured, with its model
   provider pointed at local Ollama (modelo primario: `llama3.1:8b`).
2. **Hyprland bind global** (ya configurado en `~/.config/hypr/bindings.lua`):
   ```lua
   o.bind("ALT + BACKSLASH", "BMO escuchar", "curl -s -X POST http://127.0.0.1:8765/trigger")
   ```

## Configuration (environment variables)

All paths default to repo-local locations; override as needed:

| Var | Default | Purpose |
|-----|---------|---------|
| `BMO_PORT` | `8765` | backend / UI port |
| `WHISPER_MODEL_PATH` | `./models/whisper-small` | faster-whisper model |
| `WHISPER_VENV_PYTHON` | `./.venv-audio/bin/python` | isolated whisper interpreter |
| `OPENCLAW_SESSION_KEY` | `agent:main:main` | OpenClaw session |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home (sessions index) |
| `PIPER_VOICE_ONNX` | `./models/piper/es_MX-claude-high.onnx` | voz TTS |
| `TTS_ENABLED` | `1` | poner `0` para silenciar |
| `SPEECH_START_RMS` / `SILENCE_RMS` | `0.04` / `0.02` | umbrales VAD del mic |
| `CHROMIUM_BIN` | `/usr/bin/chromium` | kiosk browser |
| `BMO_STARTUP_DELAY` | `0` | sleep before launch (autostart) |

## HTTP API

| Endpoint | Uso |
|----------|-----|
| `POST /trigger` | push-to-talk toggle (idle ⇄ escuchando) |
| `POST /bot_message` | inyectar un mensaje de bot (lo lee por TTS el front no lo muestra) |
| `WS /ws` | estados (`idle`/`recording`/`thinking`) + nivel de audio para el orbe |

## Manual run (no kiosk window)

```bash
.venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 8765
# then open http://127.0.0.1:8765
```
