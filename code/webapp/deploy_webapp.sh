#!/usr/bin/env bash
# Deploy/update the OmniFaceRig-repro web preview service as a systemd unit.
# Usage: bash deploy_webapp.sh   (run on the A100 server)
# Installs: /etc/systemd/system/webapp.service -> code/webapp/webapp.service
# Copies:   ./webapp/app.py -> ~/work/webapp/app.py
set -euo pipefail

APP_SRC="$(cd "$(dirname "$0")" && pwd)/app.py"
UNIT_SRC="$(cd "$(dirname "$0")" && pwd)/webapp.service"
DEST_APP="$HOME/work/webapp/app.py"
UNIT_DEST="/etc/systemd/system/webapp.service"

echo "==> 1/5 copy webapp/app.py -> $DEST_APP"
cp -f "$APP_SRC" "$DEST_APP"

echo "==> 2/5 install unit $UNIT_DEST"
if [ -w /etc/systemd/system ]; then
  cp -f "$UNIT_SRC" "$UNIT_DEST"
else
  echo 'M5@cn' | sudo -S cp -f "$UNIT_SRC" "$UNIT_DEST"
fi

echo "==> 3/5 stop legacy nohup webapp on :8000 (if any)"
# Kill only the old flask process (the one whose cmdline is the app.py entry).
OLD_PIDS=$(pgrep -f 'python -u webapp/app.py' || true)
if [ -n "$OLD_PIDS" ]; then
  kill $OLD_PIDS 2>/dev/null || true
  sleep 1
fi

echo "==> 4/5 daemon-reload + enable + restart"
echo 'M5@cn' | sudo -S systemctl daemon-reload
echo 'M5@cn' | sudo -S systemctl enable webapp
echo 'M5@cn' | sudo -S systemctl restart webapp

echo "==> 5/5 verify"
sleep 3
echo 'M5@cn' | sudo -S systemctl status webapp --no-pager -l | head -15 || true
curl -s -o /dev/null -w 'GET / -> %{http_code}\n' http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/health; echo
