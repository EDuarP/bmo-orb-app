#!/usr/bin/env python3
import asyncio
import json
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import openwakeword
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openwakeword.model import Model

APP_DIR = Path(__file__).resolve().parent
MIC_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=20)
CLIENTS: set[WebSocket] = set()

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')

model = Model(wakeword_model_paths=[p for p in openwakeword.get_pretrained_model_paths() if 'hey_jarvis' in p])
TRIGGER_THRESHOLD = 0.5
TARGET_MODEL = 'hey_jarvis'


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
    cmd = [
        'arecord',
        '-D', device,
        '-q',
        '-r', '16000',
        '-f', 'S16_LE',
        '-c', '1',
        '-t', 'raw'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            chunk = proc.stdout.read(1280 * 2)
            if not chunk:
                continue
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            if pcm16.size == 0:
                continue
            float_audio = pcm16.astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(np.square(float_audio))))
            scores = model.predict(pcm16)
            score = 0.0
            for key, value in scores.items():
                key_norm = key.replace(' ', '_').lower()
                if TARGET_MODEL in key_norm or 'jarvis' in key_norm:
                    score = float(value)
                    break
            payload = {
                'level': round(level, 4),
                'score': round(score, 4),
                'detected': score >= TRIGGER_THRESHOLD,
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
    return FileResponse(APP_DIR / 'favicon.ico') if (APP_DIR / 'favicon.ico').exists() else FileResponse(APP_DIR / 'index.html')


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


async def broadcaster():
    detected_hold = 0
    last_device = 'unknown'
    while True:
        await asyncio.sleep(0.05)
        try:
            data = MIC_QUEUE.get_nowait()
        except queue.Empty:
            continue
        last_device = data['device']
        if data['detected']:
            detected_hold = 20
        elif detected_hold > 0:
            detected_hold -= 1
        wake_state = 'detected' if detected_hold > 0 else 'listening'
        payload = json.dumps({
            'type': 'audio_level',
            'level': data['level'],
            'heard': 'hey jarvis' if detected_hold > 0 else '--',
            'device': last_device,
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
