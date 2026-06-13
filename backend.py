#!/usr/bin/env python3
"""
BMO Orb — voice assistant backend.

Audio pipeline
──────────────
sounddevice (1280-sample / 80 ms blocks at 16 kHz)
  └─▶ raw_q  ──▶  pipeline_thread
                    ├─ IDLE       : waits for a push-to-talk trigger (POST /trigger,
                    │               fired by Alt+backslash via Hyprland bind or kiosk page)
                    ├─ RECORDING  : accumulates COMMAND_SECONDS of audio after trigger
                    └─ offloads Whisper + bot query to worker thread
                         └─▶ broadcast_q  ──▶  broadcaster (async)  ──▶  WebSocket clients

OpenClaw integration
────────────────────
Uses `openclaw gateway call chat.send` against a real sessionKey.
This avoids assuming the gateway root is a plain REST chat endpoint.

External systems can also push bot messages directly via:
  POST /bot_message  {"text": "..."}
"""
import asyncio
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
try:
    from piper import PiperVoice
except Exception:  # piper optional — app still runs (silent) without it
    PiperVoice = None

# ── paths ──────────────────────────────────────────────────────────────────────
# All external-asset locations are env-overridable so the app is portable across
# machines. Defaults preserve the original OpenClaw workspace layout.
APP_DIR = Path(__file__).resolve().parent
OPENCLAW_HOME = Path(os.getenv('OPENCLAW_HOME', str(Path.home() / '.openclaw')))
# Audio assets default to repo-local locations (set up by setup.sh) so the app is
# self-contained; override with env vars to point elsewhere.
WHISPER_VENV_PYTHON = os.getenv('WHISPER_VENV_PYTHON', str(APP_DIR / '.venv-audio/bin/python'))
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', str(APP_DIR / 'models/whisper-small'))

# ── OpenClaw integration ───────────────────────────────────────────────────────
OPENCLAW_SESSION_KEY = os.getenv('OPENCLAW_SESSION_KEY', 'agent:main:main')
OPENCLAW_TIMEOUT_MS = int(os.getenv('OPENCLAW_TIMEOUT_MS', '45000'))
OPENCLAW_SESSIONS_INDEX = Path(os.getenv(
    'OPENCLAW_SESSIONS_INDEX', str(OPENCLAW_HOME / 'agents/main/sessions/sessions.json')))

# Robustez: textos que BMO NO debe decir en voz alta. Son ecos del envoltorio de
# mensajería interna de OpenClaw que un modelo pequeño a veces parrotea (fue la
# causa del bucle "Sender (untrusted metadata)/contenido inter-sesión"). Si una
# respuesta es solo este ruido, se descarta.
_NOISE_MARKERS = (
    'contenido inter-sesión',
    'contenido inter-sesion',
    '[inter-session message]',
    'inter-session data',
    'untrusted metadata',
    'sourcetool=sessions',
    'sourcesession=',
    'el sender (no confiable)',
    'el sender no confiable',
    'sender (untrusted metadata)',
)
# Señales de sesión "clavada" (overflow / compactación fallida). No se dicen: en su
# lugar el backend resetea la sesión y reintenta una vez.
_WEDGED_MARKERS = (
    'auto-compaction could not recover',
    'context overflow',
    'already compacted',
    'prompt too large for the model',
    'start a fresh session',
)


def _is_noise_reply(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _NOISE_MARKERS)


def _is_wedged_reply(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _WEDGED_MARKERS)

# ── audio / model config ───────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280                                   # 80 ms blocks at 16 kHz
CHUNK_SECONDS = CHUNK_SAMPLES / TARGET_SAMPLE_RATE     # 0.08 s per chunk

# VAD: after the push-to-talk trigger, energy endpointing.
MIN_COMMAND_SECONDS = 1.0
MAX_COMMAND_SECONDS = 12.0
SILENCE_HANG_SECONDS = 0.9
# RMS thresholds (normalized 0-1). Env-tunable per mic — the AM8 reads lower than
# the original defaults, so these are set conservatively low and can be overridden.
SPEECH_START_RMS_THRESHOLD = float(os.getenv('SPEECH_START_RMS', '0.04'))
SILENCE_RMS_THRESHOLD = float(os.getenv('SILENCE_RMS', '0.02'))
ASSISTANT_IDLE_WINDOW_SECONDS = 2.0

# ── text-to-speech (Piper) ───────────────────────────────────────────────────────
TTS_ENABLED = os.getenv('TTS_ENABLED', '1').lower() not in ('0', 'false', 'no')
PIPER_VOICE_ONNX = os.getenv('PIPER_VOICE_ONNX', str(APP_DIR / 'models/piper/es_MX-claude-high.onnx'))

# ── queues & shared state ──────────────────────────────────────────────────────
raw_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
broadcast_q: "queue.Queue[dict]" = queue.Queue(maxsize=50)
# Worker thread signals pipeline which state to resume ('recording' or 'idle')
next_state_q: "queue.Queue[str]" = queue.Queue(maxsize=4)
# Push-to-talk: POST /trigger drops a token here; pipeline_thread toggles state.
trigger_q: "queue.Queue[str]" = queue.Queue(maxsize=4)
CLIENTS: set["WebSocket"] = set()

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')


# ── text-to-speech (Piper) ───────────────────────────────────────────────────────
_AUDIO_PLAYER = next((p for p in ('pw-play', 'paplay', 'aplay') if shutil.which(p)), None)


def _load_tts():
    if not TTS_ENABLED or PiperVoice is None:
        print('[TTS] disabled', flush=True)
        return None
    if not Path(PIPER_VOICE_ONNX).exists():
        print(f'[TTS] voice not found: {PIPER_VOICE_ONNX} — TTS off', flush=True)
        return None
    try:
        voice = PiperVoice.load(PIPER_VOICE_ONNX)
        print(f'[TTS] Piper voice: {Path(PIPER_VOICE_ONNX).name} (player={_AUDIO_PLAYER})', flush=True)
        return voice
    except Exception as exc:
        print(f'[TTS] load failed: {type(exc).__name__}: {exc}', flush=True)
        return None


tts_voice = _load_tts()


def _speak(text: str) -> None:
    """Synthesize text with Piper and play it through the default output sink."""
    if not tts_voice or not _AUDIO_PLAYER or not text.strip():
        return
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            have_rate = False
            for chunk in tts_voice.synthesize(text):
                if not have_rate:
                    wf.setframerate(chunk.sample_rate)
                    have_rate = True
                wf.writeframes(chunk.audio_int16_bytes)
        if have_rate:
            subprocess.run([_AUDIO_PLAYER, wav_path], capture_output=True, timeout=120)
    except Exception as exc:
        print(f'[TTS] speak failed: {type(exc).__name__}: {exc}', flush=True)
    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


# ── device selection ───────────────────────────────────────────────────────────
def _find_usb_input(retries: int = 10, delay: float = 3.0) -> int | None:
    """Return sounddevice index of first USB microphone, or None for default.

    The mic can be briefly absent right after a backend restart (the old process
    is still releasing it), so re-scan a few times before giving up.
    """
    for attempt in range(1, retries + 1):
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0 and 'USB' in d['name'].upper():
                print(f'[AUDIO] Found USB mic: {d["name"]} (idx={i})', flush=True)
                return i
        print(f'[AUDIO] USB mic not found (attempt {attempt}/{retries}) — rescanning in {delay}s', flush=True)
        time.sleep(delay)
        sd._terminate()
        sd._initialize()
    print('[AUDIO] No USB mic found, using default device', flush=True)
    return None


# ── audio helpers ──────────────────────────────────────────────────────────────
def _write_wav(path: str, pcm: np.ndarray) -> None:
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm.astype(np.int16).tobytes())


WHISPER_TIMEOUT_SECONDS = 90

# Long-lived worker script: loads the model ONCE, then transcribes wav paths
# read from stdin (one per line), answering one JSON string per line.
_WHISPER_WORKER_SRC = (
    'import json, sys\n'
    'from faster_whisper import WhisperModel\n'
    f'm = WhisperModel({WHISPER_MODEL_PATH!r}, device="cpu", compute_type="int8")\n'
    'print("READY", flush=True)\n'
    'for line in sys.stdin:\n'
    '    p = line.strip()\n'
    '    if not p:\n'
    '        continue\n'
    '    try:\n'
    '        segs, info = m.transcribe(p, language="es", vad_filter=True)\n'
    '        text = " ".join(s.text.strip() for s in segs).strip()\n'
    '    except Exception:\n'
    '        text = ""\n'
    '    print(json.dumps(text), flush=True)\n'
)


class _WhisperServer:
    """Persistent faster-whisper subprocess — avoids ~3s model reload per query."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _readline(self, timeout: float) -> str | None:
        import select
        assert self._proc and self._proc.stdout
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            return None
        return self._proc.stdout.readline().strip()

    def _ensure(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True
        print('[WHISPER] starting persistent worker (loading model)…', flush=True)
        t0 = time.monotonic()
        self._proc = subprocess.Popen(
            [WHISPER_VENV_PYTHON, '-c', _WHISPER_WORKER_SRC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        line = self._readline(WHISPER_TIMEOUT_SECONDS)
        if line != 'READY':
            print('[WHISPER] worker failed to start', flush=True)
            self._kill()
            return False
        print(f'[WHISPER] worker ready in {time.monotonic() - t0:.1f}s', flush=True)
        return True

    def _kill(self) -> None:
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def warm_up(self) -> None:
        with self._lock:
            self._ensure()

    def transcribe(self, wav_path: str) -> str:
        with self._lock:
            if not self._ensure():
                return ''
            try:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(wav_path + '\n')
                self._proc.stdin.flush()
                line = self._readline(WHISPER_TIMEOUT_SECONDS)
                if line is None:
                    print(f'[WHISPER] timed out after {WHISPER_TIMEOUT_SECONDS}s — restarting worker', flush=True)
                    self._kill()
                    return ''
                return json.loads(line)
            except Exception as exc:
                print(f'[WHISPER] worker error: {type(exc).__name__}: {exc}', flush=True)
                self._kill()
                return ''


_whisper_server = _WhisperServer()


def _transcribe(pcm: np.ndarray) -> str:
    """Run faster-whisper on PCM audio via the persistent worker."""
    audio_seconds = len(pcm) / TARGET_SAMPLE_RATE
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
    try:
        _write_wav(wav_path, pcm)
        print(f'[WHISPER] transcribing {audio_seconds:.2f}s audio…', flush=True)
        t0 = time.monotonic()
        text = _whisper_server.transcribe(wav_path)
        print(f'[WHISPER] done in {time.monotonic() - t0:.1f}s', flush=True)
        return text
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _extract_text(node) -> str:
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, list):
        parts = [_extract_text(item) for item in node]
        return ' '.join(p for p in parts if p).strip()
    if isinstance(node, dict):
        if node.get('type') == 'text' and isinstance(node.get('text'), str):
            return node['text'].strip()
        if 'content' in node:
            return _extract_text(node['content'])
    return ''


def _run_gateway_call(method: str, params: dict, timeout_ms: int) -> subprocess.CompletedProcess:
    # `bash -lc` runs a login shell so PATH already includes the mise-managed
    # `openclaw` on this machine — no nvm needed. Override OPENCLAW_ENV_SETUP to
    # prepend extra setup if a different node manager is required (e.g. 'nvm use 22 && ').
    setup = os.getenv('OPENCLAW_ENV_SETUP', '')
    cmd = (
        f'{setup}'
        f'openclaw gateway call {method} --json --timeout {timeout_ms} '
        f'--params {json.dumps(json.dumps(params))}'
    )
    return subprocess.run(
        ['bash', '-lc', cmd],
        capture_output=True,
        text=True,
        timeout=max(20, int(timeout_ms / 1000) + 5),
    )


def _session_file_for_key(session_key: str) -> Path | None:
    try:
        data = json.loads(OPENCLAW_SESSIONS_INDEX.read_text())
        session_file = data.get(session_key, {}).get('sessionFile')
        return Path(session_file) if session_file else None
    except Exception as exc:
        print(f'[OPENCLAW] session file lookup failed: {exc}', flush=True)
        return None


def _read_new_assistant_replies(session_file: Path | None, start_offset: int) -> tuple[list[str], int]:
    replies: list[str] = []
    if not session_file or not session_file.exists():
        return replies, start_offset
    try:
        with session_file.open('r', encoding='utf-8') as fh:
            fh.seek(start_offset)
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get('type') != 'message':
                    continue
                msg = entry.get('message', {})
                if msg.get('role') != 'assistant':
                    continue
                content = _extract_text(msg.get('content', ''))
                cleaned = content.replace('[[reply_to_current]]', '').strip() if content else ''
                # Nunca pronunciar los ecos del envoltorio de mensajería interna.
                if cleaned and not _is_noise_reply(cleaned):
                    replies.append(cleaned)
            new_offset = fh.tell()
        return replies, new_offset
    except Exception as exc:
        print(f'[OPENCLAW] session file read failed: {exc}', flush=True)
        return [], start_offset


def _reset_session() -> bool:
    """Reset BMO's OpenClaw session to recover from a wedged/overflowed state."""
    try:
        res = _run_gateway_call('sessions.reset', {'key': OPENCLAW_SESSION_KEY}, 15000)
        ok = res.returncode == 0
        print(f'[OPENCLAW] session reset ({"ok" if ok else "failed"})', flush=True)
        return ok
    except Exception as exc:
        print(f'[OPENCLAW] session reset error: {exc}', flush=True)
        return False


def _query_openclaw(text: str, _attempt: int = 1) -> list[str]:
    if not OPENCLAW_SESSION_KEY:
        return []

    idempotency_key = f'bmo-orb-{uuid.uuid4()}'

    try:
        session_file = _session_file_for_key(OPENCLAW_SESSION_KEY)
        start_offset = session_file.stat().st_size if session_file and session_file.exists() else 0

        result = _run_gateway_call(
            'chat.send',
            {
                'sessionKey': OPENCLAW_SESSION_KEY,
                'message': text,
                'idempotencyKey': idempotency_key,
            },
            OPENCLAW_TIMEOUT_MS,
        )
        if result.returncode != 0:
            print(f'[OPENCLAW] chat.send failed: {result.stderr.strip() or result.stdout.strip()}', flush=True)
            return []

        stdout = result.stdout.strip()
        if stdout:
            print(f'[OPENCLAW] chat.send -> {stdout}', flush=True)

        # Sesión nueva (key dedicada recién reseteada): el archivo aún no existía
        # cuando calculamos session_file. Re-resolverlo tras el envío.
        if session_file is None:
            for _ in range(6):
                session_file = _session_file_for_key(OPENCLAW_SESSION_KEY)
                if session_file and session_file.exists():
                    break
                time.sleep(0.5)

        deadline = time.monotonic() + max(15, OPENCLAW_TIMEOUT_MS / 1000)
        replies: list[str] = []
        last_reply_at: float | None = None
        while time.monotonic() < deadline:
            new_replies, start_offset = _read_new_assistant_replies(session_file, start_offset)
            if new_replies:
                replies.extend(new_replies)
                last_reply_at = time.monotonic()
            if replies and last_reply_at and (time.monotonic() - last_reply_at) >= ASSISTANT_IDLE_WINDOW_SECONDS:
                break
            time.sleep(0.5)

        # ¿Sesión clavada (overflow / compactación fallida)? Resetear y reintentar
        # UNA vez con una conversación limpia, en vez de pronunciar el error.
        if any(_is_wedged_reply(r) for r in replies):
            print('[OPENCLAW] sesión clavada detectada — reseteando y reintentando', flush=True)
            if _attempt == 1 and _reset_session():
                time.sleep(0.5)
                return _query_openclaw(text, _attempt=2)
            return ['Tuve que reiniciar la conversación. ¿Me lo repites?']

        # Quitar cualquier sentinela de error que se haya colado junto a texto válido.
        replies = [r for r in replies if not _is_wedged_reply(r)]
        if replies:
            return replies
        print('[OPENCLAW] no assistant reply found before timeout', flush=True)
        return []
    except Exception as exc:
        print(f'[OPENCLAW] {exc}', flush=True)
        return []


# ── broadcast helper ───────────────────────────────────────────────────────────
def _push(payload: dict) -> None:
    try:
        broadcast_q.put_nowait(payload)
    except queue.Full:
        print('[FRONT] broadcast queue full, dropping oldest payload', flush=True)
        try:
            broadcast_q.get_nowait()
        except queue.Empty:
            pass
        broadcast_q.put_nowait(payload)


# ── worker: transcribe → bot query → broadcast ─────────────────────────────────
def _respond_worker(audio: np.ndarray, device_name: str) -> None:
    """Single-shot: transcribe → answer → speak → back to idle (no turn loop)."""
    try:
        print('[WORKER] started', flush=True)
        text = _transcribe(audio)
        print(f'[WHISPER] transcription -> {text!r}', flush=True)

        if not text:
            print('[WORKER] empty transcription — back to idle', flush=True)
            next_state_q.put('idle')
            return

        print('[OPENCLAW] sending transcript to chat backend', flush=True)
        _push({'type': 'user_message', 'text': text})

        replies = _query_openclaw(text)
        print(f'[OPENCLAW] replies -> {replies!r}', flush=True)
        if replies:
            _push({'type': 'state', 'state': 'speaking'})
            for reply in replies:
                _push({'type': 'bot_message', 'text': reply})
                _speak(reply)
        else:
            print('[OPENCLAW] empty reply or endpoint unavailable', flush=True)

        next_state_q.put('idle')
    except Exception as exc:
        print(f'[ERROR][WORKER] {type(exc).__name__}: {exc}', flush=True)
        _push({'type': 'bot_message', 'text': f'Error: {type(exc).__name__}'})
        next_state_q.put('idle')


# ── main audio/pipeline thread ─────────────────────────────────────────────────
def pipeline_thread() -> None:
    try:
        device_idx = _find_usb_input()
        device_info = sd.query_devices(device_idx) if device_idx is not None else sd.query_devices(sd.default.device[0])
        device_name = device_info['name']
        input_sample_rate = int(device_info.get('default_samplerate') or 44100)
        input_blocksize = int(round(CHUNK_SAMPLES * input_sample_rate / TARGET_SAMPLE_RATE))

        resample_buf = np.array([], dtype=np.int16)

        def _sd_callback(indata, frames, time_info, status):
            nonlocal resample_buf
            if status:
                print(f'[SD] {status}', flush=True)
            mono = indata[:, 0].copy()
            resample_buf = np.concatenate([resample_buf, mono])
            resampled = resample_poly(resample_buf.astype(np.float32), TARGET_SAMPLE_RATE, input_sample_rate)
            ready_samples = (len(resampled) // CHUNK_SAMPLES) * CHUNK_SAMPLES
            if ready_samples == 0:
                return
            ready = np.clip(resampled[:ready_samples], -32768, 32767).astype(np.int16)
            consumed_input = int(round(ready_samples * input_sample_rate / TARGET_SAMPLE_RATE))
            resample_buf = resample_buf[consumed_input:]
            for start in range(0, len(ready), CHUNK_SAMPLES):
                try:
                    raw_q.put_nowait(ready[start:start + CHUNK_SAMPLES])
                except queue.Full:
                    pass

        with sd.InputStream(
            device=device_idx,
            samplerate=input_sample_rate,
            channels=1,
            dtype='int16',
            blocksize=input_blocksize,
            callback=_sd_callback,
        ):
            _push({'type': 'state', 'state': 'idle', 'device': device_name})
            print(f'[AUDIO] Stream open on "{device_name}" at {input_sample_rate} Hz, resampling to {TARGET_SAMPLE_RATE} Hz', flush=True)
            print('[TRIGGER] push-to-talk ready — POST /trigger (Alt+\\) toggles listening', flush=True)

            state = 'idle'
            command_buf: list[np.ndarray] = []
            speech_started = False
            silence_chunks = 0

            while True:
                try:
                    chunk = raw_q.get(timeout=1.0)
                except queue.Empty:
                    chunk = None   # no audio — still service the trigger below

                # Push-to-talk toggle from POST /trigger. Processed even when the
                # audio stream is silent so the hotkey never queues up unanswered.
                triggered = False
                try:
                    trigger_q.get_nowait()
                    triggered = True
                except queue.Empty:
                    pass

                if chunk is None:
                    if triggered and state == 'idle':
                        print('[TRIGGER] hotkey (sin audio del mic todavía) — modo conversación', flush=True)
                        state = 'recording'
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        _push({'type': 'state', 'state': 'recording'})
                    elif triggered and state == 'recording':
                        print('[TRIGGER] hotkey — cancelando, vuelta a idle', flush=True)
                        state = 'idle'
                        _push({'type': 'state', 'state': 'idle', 'device': device_name})
                    continue

                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))

                if state == 'thinking':
                    # Trigger is ignored while a response is in flight.
                    try:
                        next_state_q.get_nowait()
                    except queue.Empty:
                        _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})
                        continue
                    print('[WORKER] done — back to idle', flush=True)
                    state = 'idle'
                    _push({'type': 'state', 'state': 'idle', 'device': device_name})
                    continue

                if state == 'idle':
                    _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})
                    if triggered:
                        print('[TRIGGER] hotkey — entrando a modo conversación', flush=True)
                        state = 'recording'
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        _push({'type': 'state', 'state': 'recording'})

                elif state == 'recording':
                    if triggered:
                        # Second press cancels the conversation and returns to idle.
                        print('[TRIGGER] hotkey — cancelando, vuelta a idle', flush=True)
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        state = 'idle'
                        _push({'type': 'state', 'state': 'idle', 'device': device_name})
                        continue
                    _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})

                    if not speech_started:
                        if rms >= SPEECH_START_RMS_THRESHOLD:
                            speech_started = True
                            command_buf = [chunk]
                            silence_chunks = 0
                            print(f'[REC] speech started rms={rms:.4f}', flush=True)
                        elif rms > 0.008:
                            # Calibration: sound present but below start threshold.
                            print(f'[REC] waiting — rms={rms:.4f} < start {SPEECH_START_RMS_THRESHOLD}', flush=True)
                        continue

                    command_buf.append(chunk)
                    silence_chunks = silence_chunks + 1 if rms < SILENCE_RMS_THRESHOLD else 0

                    elapsed = len(command_buf) * CHUNK_SECONDS
                    silence_elapsed = silence_chunks * CHUNK_SECONDS
                    hit_max = elapsed >= MAX_COMMAND_SECONDS
                    hit_silence = elapsed >= MIN_COMMAND_SECONDS and silence_elapsed >= SILENCE_HANG_SECONDS

                    if len(command_buf) % 12 == 1:
                        print(f'[REC] {elapsed:.1f}s rms={rms:.4f} silence={silence_elapsed:.2f}s', flush=True)

                    if hit_max or hit_silence:
                        reason = 'max' if hit_max else 'silence'
                        print(f'[VAD] stop ({reason}) elapsed={elapsed:.2f}s silence={silence_elapsed:.2f}s', flush=True)
                        audio = np.concatenate(command_buf).astype(np.int16)
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        _push({'type': 'state', 'state': 'thinking'})
                        state = 'thinking'
                        threading.Thread(
                            target=_respond_worker,
                            args=(audio, device_name),
                            daemon=True,
                        ).start()
    except Exception as exc:
        print(f'[ERROR][PIPELINE] {type(exc).__name__}: {exc}', flush=True)
        _push({'type': 'bot_message', 'text': f'Pipeline error: {type(exc).__name__}: {exc}'})


# ── FastAPI routes ─────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return FileResponse(APP_DIR / 'index.html')

@app.get('/styles.css')
def styles():
    return FileResponse(APP_DIR / 'styles.css')

@app.get('/app.js')
def app_js():
    return FileResponse(APP_DIR / 'app.js')

@app.get('/favicon.ico')
def favicon():
    return FileResponse(APP_DIR / 'index.html')


@app.post('/bot_message')
async def bot_message_push(body: dict):
    """OpenClaw or any external service can POST here to push a bubble.

    Also speaks the text aloud (used by bmo-reminder and other tools so BMO
    announces reminders Alexa-style).
    """
    text = (body.get('text') or '').strip()
    if text:
        _push({'type': 'bot_message', 'text': text})
        _push({'type': 'state', 'state': 'speaking'})

        def _say() -> None:
            _speak(text)
            _push({'type': 'state', 'state': 'idle'})

        threading.Thread(target=_say, daemon=True).start()
    return JSONResponse({'ok': True})


@app.post('/trigger')
async def trigger():
    """Push-to-talk toggle: idle → recording, recording → idle (cancel).

    Fired by the global Alt+\\ Hyprland bind (curl) or a keypress in the kiosk page.
    """
    try:
        trigger_q.put_nowait('toggle')
    except queue.Full:
        pass
    return JSONResponse({'ok': True})


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        await ws.send_text(json.dumps({'type': 'state', 'state': 'idle'}))
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


async def broadcaster():
    while True:
        await asyncio.sleep(0.02)   # 50 Hz flush rate
        try:
            data = broadcast_q.get_nowait()
        except queue.Empty:
            continue
        payload = json.dumps(data)
        stale = []
        for ws in CLIENTS:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            CLIENTS.discard(ws)


@app.on_event('startup')
async def startup():
    threading.Thread(target=pipeline_thread, daemon=True).start()
    # Pre-load the whisper model so the first voice query doesn't pay for it.
    threading.Thread(target=_whisper_server.warm_up, daemon=True).start()
    asyncio.create_task(broadcaster())
