#!/usr/bin/env python3
import asyncio
import json
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openwakeword.model import Model
import openwakeword

APP_DIR = Path(__file__).resolve().parent
MIC_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=20)
CLIENTS: set[WebSocket] = set()

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')

openwakeword.utils.download_models()
model = Model(wakeword_models=[], inference_framework='onnx')
TRIGGER_THRESHOLD = 0.5
TARGET_MODEL = 'hey_jarvis'


def audio_callback(indata, frames, time, status):
    if status:
        return
    audio = indata[:, 0].copy()
    level = float(np.sqrt(np.mean(np.square(audio))))
    pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    scores = model.predict(pcm16)
    score = 0.0
    for key, value in scores.items():
      if TARGET_MODEL in key.replace(' ', '_').lower() or 'jarvis' in key.lower():
        score = float(value)
        break
    payload = {
        'level': round(level, 4),
        'score': round(score, 4),
        'detected': score >= TRIGGER_THRESHOLD,
    }
    try:
        MIC_QUEUE.put_nowait(payload)
    except queue.Full:
        try:
            MIC_QUEUE.get_nowait()
        except queue.Empty:
            pass
        MIC_QUEUE.put_nowait(payload)


def start_audio_stream():
    devices = sd.query_devices()
    input_device = None
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0 and 'USB' in device['name']:
            input_device = idx
            break
    if input_device is None:
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_device = idx
                break
    stream = sd.InputStream(device=input_device, channels=1, samplerate=16000, callback=audio_callback, blocksize=1280, dtype='float32')
    stream.start()
    return stream, devices[input_device]['name'] if input_device is not None else 'Unknown'


@app.get('/')
def root():
    return FileResponse(APP_DIR / 'index.html')


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        await ws.send_text(json.dumps({'type': 'status', 'mic': 'live', 'wake': 'hey jarvis armed'}))
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


async def broadcaster(device_name: str):
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
            'heard': 'hey jarvis' if detected_hold > 0 else '--',
            'device': device_name,
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


def run_audio_loop():
    stream, device_name = start_audio_stream()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(broadcaster(device_name))
    try:
        loop.run_forever()
    finally:
        stream.stop()
        stream.close()


threading.Thread(target=run_audio_loop, daemon=True).start()
