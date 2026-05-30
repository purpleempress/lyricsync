#!/usr/bin/env bash
# Build the Lucid Lyrics fork and apply it to Spotify.
# Bundles the PATH for bun + node (nvm) + spicetify so it works in a fresh shell.
set -euo pipefail
export PATH="$HOME/.bun/bin:$HOME/.spicetify:$PATH"
# pick up nvm's node if present
for d in "$HOME"/.nvm/versions/node/*/bin; do [ -d "$d" ] && export PATH="$d:$PATH"; done
cd "$(dirname "$0")/../lucid-lyrics"
exec node_modules/.bin/spicetify-creator build --apply "$@"
