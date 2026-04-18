#!/usr/bin/env python3
import asyncio
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openwakeword.model import Model

APP_DIR = Path(__file__).resolve().parent
BMO_MODEL_DIR = Path('/home/eduarp/.openclaw/workspace/repos/BMO/openWakeWordModel')
MIC_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=20)
CLIENTS: set[WebSocket] = set()

WAKEWORD_THRESHOLD = 0.4
CHUNK_SAMPLES = 32000  # 2 seconds at 16 kHz, matching BMO chunk_sec=2
WAKEWORD_MODEL_TFLITE = BMO_MODEL_DIR / 'hey_bee_moh.tflite'
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


def audio_loop():
    device = pick_arecord_device()
    print(f'[AUDIO] Using input device: {device}', flush=True)
    cmd = ['arecord', '-D', device, '-q', '-r', '16000', '-f', 'S16_LE', '-c', '1', '-t', 'raw']
    print(f"[AUDIO] Starting command: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buffer = np.array([], dtype=np.int16)
    frame_counter = 0
    last_log = 0.0
    try:
        while True:
            chunk = proc.stdout.read(1280 * 2)
            if not chunk:
                print('[AUDIO] Empty chunk received from arecord', flush=True)
                continue
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                print('[AUDIO] Zero-length PCM frame decoded', flush=True)
                continue
            frame_counter += 1
            buffer = np.concatenate([buffer, pcm16])
            if buffer.size < CHUNK_SAMPLES:
                if time.time() - last_log > 2:
                    print(f'[AUDIO] buffering... samples={buffer.size}/{CHUNK_SAMPLES}', flush=True)
                    last_log = time.time()
                continue

            audio_i16 = buffer[:CHUNK_SAMPLES]
            buffer = buffer[-1280:]  # keep small overlap for continuity
            float_audio = audio_i16.astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(np.square(float_audio))))
            print(f'[AUDIO] frame={frame_counter} level={level:.4f} max={float(np.max(np.abs(float_audio))):.4f}', flush=True)
            prediction = wakeword_model.predict(audio_i16)
            score = float(prediction.get(wakeword_name, next(iter(prediction.values()), 0.0)))
            print(f"[WAKEWORD] model={wakeword_name} score={score:.4f} threshold={WAKEWORD_THRESHOLD:.2f} detected={score >= WAKEWORD_THRESHOLD}", flush=True)

            payload = {
                'level': round(level, 4),
                'score': round(score, 4),
                'detected': score >= WAKEWORD_THRESHOLD,
                'device': device,
            }
            try:
                MIC_QUEUE.put_nowait(payload)
            except queue.Full:
                try:
                    MIC_QUEUE.get_nowait()
                except queue.Empty:
                    pass
                MIC_QUEUE.put_nowait(payload)
    finally:
        proc.terminate()


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
        await ws.send_text(json.dumps({'type': 'status', 'mic': 'live', 'wake': 'hey bemo armed'}))
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
            'heard': 'hey bemo' if detected_hold > 0 else '--',
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
