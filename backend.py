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
Set OPENCLAW_URL to POST {"message": "..."} and expect {"reply": "..."}.
Leave empty to skip bot querying (useful while debugging audio).

External systems can also push bot messages directly via:
  POST /bot_message  {"text": "..."}
"""
import asyncio
import json
import queue
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.request
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
# POST {"message": "..."} → expects {"reply": "..."}
# Leave as '' to skip bot query and only show transcription.
OPENCLAW_URL = 'http://127.0.0.1:18789'

# ── audio / model config ───────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
# CRITICAL: OpenWakeWord was trained on 80ms (1280-sample) windows at 16kHz.
# Passing larger chunks causes the model to evaluate only the LAST window and
# miss detections earlier in the block. Always feed exactly 1280 samples.
WAKEWORD_CHUNK = 1280
CHUNK_SECONDS = WAKEWORD_CHUNK / TARGET_SAMPLE_RATE   # 0.08 s per chunk
WAKEWORD_THRESHOLD = 0.5
TRIGGER_COOLDOWN = 3.0   # min seconds between back-to-back triggers

# VAD: after wakeword triggers, record until user stops talking.
MIN_COMMAND_SECONDS = 1.5   # ignore silence before this floor
MAX_COMMAND_SECONDS = 10.0  # hard cap
SILENCE_HANG_SECONDS = 0.8  # stop after this much continuous silence
SILENCE_RMS_THRESHOLD = 0.01

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


def _query_openclaw(text: str) -> str:
    if not OPENCLAW_URL:
        return ''
    try:
        data = json.dumps({'message': text}).encode()
        req = urllib.request.Request(
            OPENCLAW_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get('reply', '')
    except Exception as exc:
        print(f'[OPENCLAW] {exc}', flush=True)
        return ''


# ── broadcast helper ───────────────────────────────────────────────────────────
def _push(payload: dict) -> None:
    try:
        broadcast_q.put_nowait(payload)
    except queue.Full:
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
    text = _transcribe(audio)
    print(f'[WHISPER] heard: {text!r}', flush=True)

    if not text:
        # Empty transcription — stay in conversation so user can retry.
        next_state_q.put('recording')
        return

    _push({'type': 'user_message', 'text': text})

    if _is_exit_phrase(text):
        print('[CONVERSATION] exit phrase detected — leaving conversation mode', flush=True)
        _push({'type': 'bot_message', 'text': 'Hasta luego.'})
        _push({'type': 'state', 'state': 'listening', 'device': device_name})
        next_state_q.put('listening')
        return

    reply = _query_openclaw(text)
    if reply:
        print(f'[BOT] reply: {reply!r}', flush=True)
        _push({'type': 'bot_message', 'text': reply})

    # Stay in conversation — pipeline will record next turn without wakeword.
    next_state_q.put('recording')


# ── main audio/pipeline thread ─────────────────────────────────────────────────
def pipeline_thread() -> None:
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
        silence_chunks = 0

        while True:
            try:
                chunk = raw_q.get(timeout=1.0)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))

            # Worker signals back to pipeline via next_state_q when thinking ends.
            if state == 'thinking':
                try:
                    nxt = next_state_q.get_nowait()
                except queue.Empty:
                    # Still processing: keep UI alive and discard audio.
                    _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})
                    continue
                if nxt == 'recording':
                    print('[CONVERSATION] continuing — next turn', flush=True)
                    command_buf = []
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
                    print(f'[WAKEWORD] Triggered! score={score:.4f} — entering conversation', flush=True)
                    state = 'recording'
                    command_buf = []
                    silence_chunks = 0
                    _push({'type': 'state', 'state': 'recording'})

            elif state == 'recording':
                command_buf.append(chunk)
                _push({'type': 'audio_level', 'level': round(rms, 4), 'score': 0.0})

                silence_chunks = silence_chunks + 1 if rms < SILENCE_RMS_THRESHOLD else 0

                elapsed = len(command_buf) * CHUNK_SECONDS
                silence_elapsed = silence_chunks * CHUNK_SECONDS
                hit_max = elapsed >= MAX_COMMAND_SECONDS
                hit_silence = elapsed >= MIN_COMMAND_SECONDS and silence_elapsed >= SILENCE_HANG_SECONDS

                if len(command_buf) % 12 == 1:   # ~1 log per second
                    print(f'[REC] {elapsed:.1f}s rms={rms:.4f} silence={silence_elapsed:.2f}s', flush=True)

                if hit_max or hit_silence:
                    reason = 'max' if hit_max else 'silence'
                    print(f'[VAD] stop ({reason}) elapsed={elapsed:.2f}s silence={silence_elapsed:.2f}s', flush=True)
                    audio = np.concatenate(command_buf).astype(np.int16)
                    _push({'type': 'state', 'state': 'thinking'})
                    state = 'thinking'
                    threading.Thread(
                        target=_respond_worker,
                        args=(audio, device_name),
                        daemon=True,
                    ).start()


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
