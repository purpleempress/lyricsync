#!/usr/bin/env python3
"""One-time Spotify login for LyricSync's librespot audio source.

Opens a Spotify OAuth URL; log in with your (Premium) account in the browser.
On success, writes reusable credentials so the API can stream tracks by id.

    .venv/bin/python scripts/librespot_login.py
    # then run the API with:
    LIBRESPOT_CREDENTIALS=$PWD/credentials.json \
      .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

Destination defaults to ./credentials.json (override with $LIBRESPOT_CREDENTIALS).
This is your personal account; keep credentials.json private (it's gitignored).
"""

from __future__ import annotations

import os
import webbrowser

from librespot.core import Session

DST = os.environ.get("LIBRESPOT_CREDENTIALS", "credentials.json")


def _on_url(url: str) -> None:
    print("\n  Open this URL in your browser and log in to Spotify:\n")
    print(f"    {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> int:
    conf = (Session.Configuration.Builder()
            .set_store_credentials(True)
            .set_stored_credential_file(DST)
            .build())
    session = Session.Builder(conf).oauth(_on_url).create()
    print(f"\n  ✓ Logged in as: {session.username()}")
    print(f"  ✓ Credentials saved to: {os.path.abspath(DST)}")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
