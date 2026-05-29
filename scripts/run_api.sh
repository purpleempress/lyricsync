#!/usr/bin/env bash
# Launch the LyricSync API wired to the stored auth file, on the LAN.
set -euo pipefail
cd "$(dirname "$0")/.."

AUTH="$PWD/credentials.json"
pkill -f "uvicorn api.main:app" 2>/dev/null || true
sleep 1

export LIBRESPOT_CREDENTIALS="$AUTH"
PYTHONPATH=. nohup .venv/bin/uvicorn api.main:app \
  --host 0.0.0.0 --port 8000 > /tmp/lyricsync_api.log 2>&1 &
disown
sleep 4
curl -s http://127.0.0.1:8000/health
echo
