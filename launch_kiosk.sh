#!/usr/bin/env bash
set -e
sleep 14
cd /home/eduarp/.openclaw/workspace/repos/bmo-orb-app
/usr/bin/python3 -m http.server 8765 --bind 127.0.0.1 >/tmp/bmo-orb-server.log 2>&1 &
sleep 2
/usr/bin/chromium \
  --start-fullscreen \
  --noerrdialogs \
  --disable-infobars \
  --autoplay-policy=no-user-gesture-required \
  --use-fake-ui-for-media-stream \
  --app=http://127.0.0.1:8765
