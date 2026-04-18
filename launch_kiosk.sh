#!/usr/bin/env bash
set -e
sleep 45
cd /home/eduarp/.openclaw/workspace/repos/bmo-orb-app
pkill -f 'uvicorn backend:app' || true
nohup /usr/bin/python3 -m uvicorn backend:app --host 127.0.0.1 --port 8765 >/tmp/bmo-orb-backend.log 2>&1 &
sleep 4
/usr/bin/chromium \
  --noerrdialogs \
  --disable-infobars \
  --autoplay-policy=no-user-gesture-required \
  --password-store=basic \
  --user-data-dir=/tmp/bmo-orb-profile \
  --app=http://127.0.0.1:8765 \
  --window-size=720,720 \
  --window-position=600,140
