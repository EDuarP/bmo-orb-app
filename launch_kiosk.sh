#!/usr/bin/env bash
set -e
sleep 22
cd /home/eduarp/.openclaw/workspace/repos/bmo-orb-app
pkill -f 'http.server 8765' || true
/usr/bin/python3 -m http.server 8765 --bind 127.0.0.1 >/tmp/bmo-orb-server.log 2>&1 &
sleep 2
/usr/bin/chromium \
  --start-fullscreen \
  --noerrdialogs \
  --disable-infobars \
  --autoplay-policy=no-user-gesture-required \
  --app=http://127.0.0.1:8765
