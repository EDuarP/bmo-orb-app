#!/usr/bin/env python3
import asyncio
import json
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openwakeword.model import Model

APP_DIR = Path(__file__).resolve().parent
BMO_MODEL_DIR = Path('/home/eduarp/.openclaw/workspace/repos/BMO/openWakeWordModel')
WHISPER_VENV_PYTHON = '/home/eduarp/.openclaw/workspace/.venv-audio/bin/python'
WHISPER_MODEL_PATH = '/home/eduarp/.openclaw/workspace/models/whisper-small'
MIC_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=20)
CLIENTS: set[WebSocket] = set()

WAKEWORD_THRESHOLD = 0.4
CHUNK_SECONDS = 3
CHUNK_SAMPLES = 16000 * CHUNK_SECONDS
WAKEWORD_MODEL_ONNX = BMO_MODEL_DIR / 'hey_bee_moh.onnx'
WAKEWORD_FEATURE_MODELS_DIR = BMO_MODEL_DIR / 'resources'

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')


def load_wakeword_model():
    model_path = WAKEWORD_MODEL_ONNX
    if not model_path.exists():
        raise FileNotFoundError(f'Missing wakeword model: {model_path}')

    melspec_model_path = WAKEWORD_FEATURE_MODELS_DIR / 'melspectrogram.onnx'
    embedding_model_path = WAKEWORD_FEATURE_MODELS_DIR / 'embedding_model.onnx'

    if not melspec_model_path.exists() or not embedding_model_path.exists():
        raise FileNotFoundError('Missing openWakeWord feature models in BMO resources')

    model = Model(
        wakeword_model_paths=[str(model_path)],
        melspec_onnx_model_path=str(melspec_model_path),
        embedding_onnx_model_path=str(embedding_model_path),
    )
    model_name = next(iter(model.models.keys()))
    return model, model_name


wakeword_model, wakeword_name = load_wakeword_model()


def pick_arecord_device() -> str:
    result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, check=False)
    usb_line = None
    for line in result.stdout.splitlines():
        if 'USB' in line or 'Usb Audio Device' in line:
            usb_line = line
            break
    if not usb_line:
        return 'default'
    import re
    match = re.search(r'card (\d+):.*device (\d+):', usb_line)
    if not match:
        return 'default'
    return f'hw:{match.group(1)},{match.group(2)}'


def write_wav(path: str, pcm16: np.ndarray):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16.astype(np.int16).tobytes())


def transcribe_with_whisper(wav_path: str) -> str:
    cmd = [
        WHISPER_VENV_PYTHON,
        '-c',
        (
            'from faster_whisper import WhisperModel; '
            f'm=WhisperModel("{WHISPER_MODEL_PATH}", device="cpu", compute_type="int8"); '
            f'seg,_=m.transcribe("{wav_path}", language="en"); '
            'print(" ".join(s.text.strip() for s in seg).strip())'
        )
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f'[WHISPER] error code={result.returncode} stderr={result.stderr.strip()}', flush=True)
        return ''
    return result.stdout.strip()


def record_block_arecord(device: str, seconds: int) -> np.ndarray:
    cmd = [
        'arecord', '-D', device, '-q', '-d', str(seconds),
        '-r', '16000', '-f', 'S16_LE', '-c', '1', '-t', 'raw'
    ]
    started_at = time.time()
    print('GRABANDO', flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    elapsed = time.time() - started_at
    print(f'FIN DE GRABADO {elapsed:.3f}s', flush=True)
    if proc.returncode != 0:
        print(f'[AUDIO] arecord returned code {proc.returncode}', flush=True)
    return np.frombuffer(proc.stdout, dtype=np.int16)


def audio_loop():
    device = pick_arecord_device()
    print(f'[AUDIO] Using input device: {device}', flush=True)
    while True:
        pcm16 = record_block_arecord(device, CHUNK_SECONDS)
        if pcm16.size == 0:
            print('[AUDIO] No samples captured', flush=True)
            continue
        if pcm16.size != CHUNK_SAMPLES:
            print(f'[AUDIO] Expected {CHUNK_SAMPLES} samples, got {pcm16.size}', flush=True)
            if pcm16.size < CHUNK_SAMPLES:
                continue
            pcm16 = pcm16[:CHUNK_SAMPLES]

        float_audio = pcm16.astype(np.float32) / 32768.0
        level = float(np.sqrt(np.mean(np.square(float_audio))))
        print(f'[AUDIO] level={level:.4f} max={float(np.max(np.abs(float_audio))):.4f}', flush=True)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        try:
            write_wav(wav_path, pcm16)
            transcript = transcribe_with_whisper(wav_path)
        finally:
            Path(wav_path).unlink(missing_ok=True)

        print(f'[WHISPER] transcript={transcript or "--"}', flush=True)
        prediction = wakeword_model.predict(pcm16)
        score = float(prediction.get(wakeword_name, next(iter(prediction.values()), 0.0)))
        print(f"[WAKEWORD] model={wakeword_name} score={score:.4f} threshold={WAKEWORD_THRESHOLD:.2f} detected={score >= WAKEWORD_THRESHOLD}", flush=True)

        payload = {
            'level': round(level, 4),
            'score': round(score, 4),
            'detected': score >= WAKEWORD_THRESHOLD,
            'device': device,
            'heard': transcript or '--',
        }
        try:
            MIC_QUEUE.put_nowait(payload)
        except queue.Full:
            try:
                MIC_QUEUE.get_nowait()
            except queue.Empty:
                pass
            MIC_QUEUE.put_nowait(payload)


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


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        await ws.send_text(json.dumps({'type': 'status', 'mic': 'live', 'wake': 'armed'}))
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


async def broadcaster():
    detected_hold = 0
    while True:
        await asyncio.sleep(0.05)
        try:
            data = MIC_QUEUE.get_nowait()
        except queue.Empty:
            continue
        if data['detected']:
            detected_hold = 20
        elif detected_hold > 0:
            detected_hold -= 1
        wake_state = 'detected' if detected_hold > 0 else 'listening'
        payload = json.dumps({
            'type': 'audio_level',
            'level': data['level'],
            'heard': data['heard'],
            'device': data['device'],
            'wake': wake_state,
            'score': data['score'],
        })
        stale = []
        for ws in CLIENTS:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            CLIENTS.discard(ws)


@app.on_event('startup')
async def startup_event():
    threading.Thread(target=audio_loop, daemon=True).start()
    asyncio.create_task(broadcaster())
