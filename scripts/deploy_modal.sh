#!/usr/bin/env bash
# Deploy the GPU compute stage (Demucs + faster-whisper) to Modal.
# Needs `modal` installed (pip install -e ".[modal]") and auth: either
# MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in .env (sourced below) or ~/.modal.toml
# (run `modal token new` once to get the values). Re-run after changing
# modal_app.py or the lyricsync package; the model-download layer is cached.
set -euo pipefail
cd "$(dirname "$0")/.."

# Same .env the API uses, so the CLI authenticates with the same token.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

VENV_MODAL=".venv/bin/modal"
MODAL="$([ -x "$VENV_MODAL" ] && echo "$VENV_MODAL" || echo modal)"

exec "$MODAL" deploy modal_app.py
