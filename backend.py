#!/usr/bin/env python3
"""
BMO Orb — voice assistant backend.

Audio pipeline
──────────────
sounddevice (1280-sample / 80 ms blocks at 16 kHz)
  └─▶ raw_q  ──▶  pipeline_thread
                    ├─ LISTENING  : feeds 80ms chunks to OpenWakeWord (correct window size)
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
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openwakeword.model import Model

# ── paths ──────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
BMO_MODEL_DIR = Path('/home/eduarp/.openclaw/workspace/repos/BMO/openWakeWordModel')
WHISPER_VENV_PYTHON = '/home/eduarp/.openclaw/workspace/.venv-audio/bin/python'
WHISPER_MODEL_PATH = '/home/eduarp/.openclaw/workspace/models/whisper-small'

# ── OpenClaw integration ───────────────────────────────────────────────────────
OPENCLAW_SESSION_KEY = os.getenv('OPENCLAW_SESSION_KEY', 'agent:main:main')
OPENCLAW_TIMEOUT_MS = int(os.getenv('OPENCLAW_TIMEOUT_MS', '45000'))
OPENCLAW_SESSIONS_INDEX = Path('/home/eduarp/.openclaw/agents/main/sessions/sessions.json')

# ── audio / model config ───────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
# CRITICAL: OpenWakeWord was trained on 80ms (1280-sample) windows at 16kHz.
# Passing larger chunks causes the model to evaluate only the LAST window and
# miss detections earlier in the block. Always feed exactly 1280 samples.
WAKEWORD_CHUNK = 1280
CHUNK_SECONDS = WAKEWORD_CHUNK / TARGET_SAMPLE_RATE   # 0.08 s per chunk
WAKEWORD_THRESHOLD = 0.5
TRIGGER_COOLDOWN = 3.0   # min seconds between back-to-back triggers

# VAD: after wakeword triggers, match BMO repo energy endpointing more closely.
MIN_COMMAND_SECONDS = 1.0
MAX_COMMAND_SECONDS = 12.0
SILENCE_HANG_SECONDS = 2.0
SPEECH_START_RMS_THRESHOLD = 0.16
SILENCE_RMS_THRESHOLD = 0.12
ASSISTANT_IDLE_WINDOW_SECONDS = 5.0

WAKEWORD_MODEL_ONNX = BMO_MODEL_DIR / 'hey_bee_moh.onnx'
WAKEWORD_FEATURE_DIR = BMO_MODEL_DIR / 'resources'

# Phrases that end conversation mode (normalized: lowercase, no diacritics).
EXIT_PHRASES = (
    'adios',
    'hasta luego',
    'chao',
    'chau',
    'dejemos hasta aqui',
    'corta',
)

# ── queues & shared state ──────────────────────────────────────────────────────
raw_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
broadcast_q: "queue.Queue[dict]" = queue.Queue(maxsize=50)
# Worker thread signals pipeline which state to resume ('recording' or 'listening')
next_state_q: "queue.Queue[str]" = queue.Queue(maxsize=4)
CLIENTS: set["WebSocket"] = set()

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')


# ── wakeword model ─────────────────────────────────────────────────────────────
def _load_wakeword_model() -> tuple["Model", str]:
    if not WAKEWORD_MODEL_ONNX.exists():
        raise FileNotFoundError(f'Missing model: {WAKEWORD_MODEL_ONNX}')
    melspec = WAKEWORD_FEATURE_DIR / 'melspectrogram.onnx'
    embed = WAKEWORD_FEATURE_DIR / 'embedding_model.onnx'
    if not melspec.exists() or not embed.exists():
        raise FileNotFoundError('Missing feature models in BMO resources dir')
    model = Model(
        wakeword_model_paths=[str(WAKEWORD_MODEL_ONNX)],
        melspec_onnx_model_path=str(melspec),
        embedding_onnx_model_path=str(embed),
    )
    name = next(iter(model.models))
    print(f'[MODEL] Loaded wakeword: {name}', flush=True)
    return model, name


wakeword_model, wakeword_name = _load_wakeword_model()


# ── device selection ───────────────────────────────────────────────────────────
def _find_usb_input() -> int | None:
    """Return sounddevice index of first USB microphone, or None for default."""
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and 'USB' in d['name'].upper():
            print(f'[AUDIO] Found USB mic: {d["name"]} (idx={i})', flush=True)
            return i
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


def _transcribe(pcm: np.ndarray) -> str:
    """Run faster-whisper on PCM audio. No language forced — auto-detects."""
    audio_seconds = len(pcm) / TARGET_SAMPLE_RATE
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
    try:
        _write_wav(wav_path, pcm)
        cmd = [
            WHISPER_VENV_PYTHON, '-c',
            (
                'from faster_whisper import WhisperModel; '
                f'm = WhisperModel("{WHISPER_MODEL_PATH}", device="cpu", compute_type="int8"); '
                f'segs, info = m.transcribe("{wav_path}"); '
                'text = " ".join(s.text.strip() for s in segs).strip(); '
                'import sys; print(f"LANG={{info.language}} PROB={{info.language_probability:.2f}}", file=sys.stderr); '
                'print(text)'
            ),
        ]
        print(f'[WHISPER] transcribing {audio_seconds:.2f}s audio (timeout {WHISPER_TIMEOUT_SECONDS}s)…', flush=True)
        t0 = time.monotonic()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=WHISPER_TIMEOUT_SECONDS)
        dt = time.monotonic() - t0
        if r.returncode != 0:
            print(f'[WHISPER] err {r.returncode} ({dt:.1f}s): {r.stderr.strip()[:300]}', flush=True)
            return ''
        if r.stderr.strip():
            print(f'[WHISPER] {r.stderr.strip()} ({dt:.1f}s)', flush=True)
        else:
            print(f'[WHISPER] done in {dt:.1f}s', flush=True)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f'[WHISPER] timed out after {WHISPER_TIMEOUT_SECONDS}s', flush=True)
        return ''
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
    cmd = (
        'source ~/.nvm/nvm.sh && nvm use 22 >/dev/null && '
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
                if cleaned:
                    replies.append(cleaned)
            new_offset = fh.tell()
        return replies, new_offset
    except Exception as exc:
        print(f'[OPENCLAW] session file read failed: {exc}', flush=True)
        return [], start_offset


def _query_openclaw(text: str) -> list[str]:
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

        deadline = time.monotonic() + max(15, OPENCLAW_TIMEOUT_MS / 1000)
        replies: list[str] = []
        last_reply_at: float | None = None
        while time.monotonic() < deadline:
            new_replies, start_offset = _read_new_assistant_replies(session_file, start_offset)
            if new_replies:
                replies.extend(new_replies)
                last_reply_at = time.monotonic()
            if replies and last_reply_at and (time.monotonic() - last_reply_at) >= ASSISTANT_IDLE_WINDOW_SECONDS:
                return replies
            time.sleep(0.5)

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


# ── exit phrase detection ──────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    stripped = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return stripped.lower().strip().rstrip('.,!?¡¿')


def _is_exit_phrase(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    return any(phrase in norm for phrase in EXIT_PHRASES)


# ── worker: transcribe → bot query → broadcast ─────────────────────────────────
def _respond_worker(audio: np.ndarray, device_name: str) -> None:
    try:
        print('[CONVERSATION] worker started', flush=True)
        text = _transcribe(audio)
        print(f'[WHISPER] transcription -> {text!r}', flush=True)

        if not text:
            print('[CONVERSATION] empty transcription, staying in conversation mode', flush=True)
            next_state_q.put('recording')
            return

        print('[OPENCLAW] sending transcript to chat backend', flush=True)
        _push({'type': 'user_message', 'text': text})

        if _is_exit_phrase(text):
            print('[CONVERSATION] exit phrase detected — leaving conversation mode', flush=True)
            _push({'type': 'bot_message', 'text': 'Hasta luego.'})
            _push({'type': 'state', 'state': 'listening', 'device': device_name})
            next_state_q.put('listening')
            return

        replies = _query_openclaw(text)
        print(f'[OPENCLAW] replies -> {replies!r}', flush=True)
        if replies:
            for reply in replies:
                _push({'type': 'bot_message', 'text': reply})
        else:
            print('[OPENCLAW] empty reply or endpoint unavailable', flush=True)

        print('[CONVERSATION] returning to recording mode for next turn', flush=True)
        next_state_q.put('recording')
    except Exception as exc:
        print(f'[ERROR][WORKER] {type(exc).__name__}: {exc}', flush=True)
        _push({'type': 'bot_message', 'text': f'Error: {type(exc).__name__}'})
        next_state_q.put('recording')


# ── main audio/pipeline thread ─────────────────────────────────────────────────
def pipeline_thread() -> None:
    try:
        device_idx = _find_usb_input()
        device_info = sd.query_devices(device_idx) if device_idx is not None else sd.query_devices(sd.default.device[0])
        device_name = device_info['name']
        input_sample_rate = int(device_info.get('default_samplerate') or 44100)
        input_blocksize = int(round(WAKEWORD_CHUNK * input_sample_rate / TARGET_SAMPLE_RATE))

        last_trigger = 0.0
        resample_buf = np.array([], dtype=np.int16)

        def _sd_callback(indata, frames, time_info, status):
            nonlocal resample_buf
            if status:
                print(f'[SD] {status}', flush=True)
            mono = indata[:, 0].copy()
            resample_buf = np.concatenate([resample_buf, mono])
            resampled = resample_poly(resample_buf.astype(np.float32), TARGET_SAMPLE_RATE, input_sample_rate)
            ready_samples = (len(resampled) // WAKEWORD_CHUNK) * WAKEWORD_CHUNK
            if ready_samples == 0:
                return
            ready = np.clip(resampled[:ready_samples], -32768, 32767).astype(np.int16)
            consumed_input = int(round(ready_samples * input_sample_rate / TARGET_SAMPLE_RATE))
            resample_buf = resample_buf[consumed_input:]
            for start in range(0, len(ready), WAKEWORD_CHUNK):
                try:
                    raw_q.put_nowait(ready[start:start + WAKEWORD_CHUNK])
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
            _push({'type': 'state', 'state': 'listening', 'device': device_name})
            print(f'[AUDIO] Stream open on "{device_name}" at {input_sample_rate} Hz, resampling to {TARGET_SAMPLE_RATE} Hz', flush=True)

            state = 'listening'
            command_buf: list[np.ndarray] = []
            speech_started = False
            silence_chunks = 0

            while True:
                try:
                    chunk = raw_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))

                if state == 'thinking':
                    try:
                        nxt = next_state_q.get_nowait()
                    except queue.Empty:
                        _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})
                        continue
                    print(f'[CONVERSATION] next state from worker -> {nxt}', flush=True)
                    if nxt == 'recording':
                        print('[CONVERSATION] continuing — next turn', flush=True)
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        state = 'recording'
                        _push({'type': 'state', 'state': 'recording'})
                    else:
                        state = 'listening'
                        _push({'type': 'state', 'state': 'listening', 'device': device_name})
                    continue

                if state == 'listening':
                    pred = wakeword_model.predict(chunk)
                    score = float(pred.get(wakeword_name, next(iter(pred.values()), 0.0)))
                    _push({'type': 'audio_level', 'level': round(rms, 4), 'score': round(score, 4)})
                    print(f'[WW] score={score:.4f} rms={rms:.4f}', flush=True)

                    now = time.monotonic()
                    if score >= WAKEWORD_THRESHOLD and (now - last_trigger) >= TRIGGER_COOLDOWN:
                        last_trigger = now
                        print(f'[WAKEWORD] Triggered! score={score:.4f} — entrando a modo conversación', flush=True)
                        state = 'recording'
                        command_buf = []
                        speech_started = False
                        silence_chunks = 0
                        _push({'type': 'state', 'state': 'recording'})

                elif state == 'recording':
                    _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})

                    if not speech_started:
                        if rms >= SPEECH_START_RMS_THRESHOLD:
                            speech_started = True
                            command_buf = [chunk]
                            silence_chunks = 0
                            print(f'[REC] speech started rms={rms:.4f}', flush=True)
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
    """OpenClaw or any external service can POST here to push a bubble."""
    text = (body.get('text') or '').strip()
    if text:
        _push({'type': 'bot_message', 'text': text})
    return JSONResponse({'ok': True})


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        await ws.send_text(json.dumps({'type': 'state', 'state': 'listening'}))
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
    asyncio.create_task(broadcaster())
