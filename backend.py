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

APP_DIR = Path(__file__).resolve().parent
MIC_QUEUE: "queue.Queue[float]" = queue.Queue(maxsize=10)
CLIENTS: set[WebSocket] = set()

app = FastAPI()
app.mount('/static', StaticFiles(directory=str(APP_DIR)), name='static')


def audio_callback(indata, frames, time, status):
    if status:
        return
    level = float(np.sqrt(np.mean(np.square(indata))))
    try:
        MIC_QUEUE.put_nowait(level)
    except queue.Full:
        try:
            MIC_QUEUE.get_nowait()
        except queue.Empty:
            pass
        MIC_QUEUE.put_nowait(level)


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
    stream = sd.InputStream(device=input_device, channels=1, samplerate=16000, callback=audio_callback, blocksize=2048)
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
        await ws.send_text(json.dumps({'type': 'status', 'mic': 'live', 'wake': 'backend listening'}))
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


async def broadcaster(device_name: str):
    while True:
        await asyncio.sleep(0.05)
        try:
            level = MIC_QUEUE.get_nowait()
        except queue.Empty:
            continue
        payload = json.dumps({
            'type': 'audio_level',
            'level': round(level, 4),
            'heard': '--',
            'device': device_name,
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
