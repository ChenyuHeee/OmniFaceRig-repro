#!/usr/bin/env bash
# Deploy/update the OmniFaceRig-repro web preview service as a systemd unit.
# Usage: SUDO_PASS='<sudo password>' bash deploy_webapp.sh   (run on the A100 server)
#   - or run as a user with passwordless sudo (then SUDO_PASS is unused)
# Installs: /etc/systemd/system/webapp.service -> code/webapp/webapp.service
# Copies:   ./webapp/app.py -> ~/work/webapp/app.py
set -euo pipefail

APP_SRC="$(cd "$(dirname "$0")" && pwd)/app.py"
UNIT_SRC="$(cd "$(dirname "$0")" && pwd)/webapp.service"
DEST_APP="$HOME/work/webapp/app.py"
UNIT_DEST="/etc/systemd/system/webapp.service"

# sudo helper: passwordless sudo if available, otherwise SUDO_PASS env var
maybe_sudo() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [ -n "${SUDO_PASS:-}" ]; then
    echo "$SUDO_PASS" | sudo -S "$@"
  else
    echo "need sudo: set SUDO_PASS env or configure passwordless sudo" >&2
    exit 1
  fi
}

echo "==> 1/5 copy webapp/app.py -> $DEST_APP"
cp -f "$APP_SRC" "$DEST_APP"

echo "==> 2/5 install unit $UNIT_DEST"
if [ -w /etc/systemd/system ]; then
  cp -f "$UNIT_SRC" "$UNIT_DEST"
else
  maybe_sudo cp -f "$UNIT_SRC" "$UNIT_DEST"
fi

echo "==> 3/5 stop legacy nohup webapp on :8000 (if any)"
# Kill only the old flask process (the one whose cmdline is the app.py entry).
OLD_PIDS=$(pgrep -f 'python -u webapp/app.py' || true)
if [ -n "$OLD_PIDS" ]; then
  kill $OLD_PIDS 2>/dev/null || true
  sleep 1
fi

echo "==> 4/5 daemon-reload + enable + restart"
maybe_sudo systemctl daemon-reload
maybe_sudo systemctl enable webapp
maybe_sudo systemctl restart webapp

echo "==> 5/5 verify"
sleep 3
maybe_sudo systemctl status webapp --no-pager -l | head -15 || true
curl -s -o /dev/null -w 'GET / -> %{http_code}\n' http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/health; echo
