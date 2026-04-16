#!/usr/bin/env bash
set -e
sleep 8
/usr/bin/chromium \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --autoplay-policy=no-user-gesture-required \
  --use-fake-ui-for-media-stream \
  --start-fullscreen \
  file:///home/eduarp/.openclaw/workspace/repos/bmo-orb-app/index.html
