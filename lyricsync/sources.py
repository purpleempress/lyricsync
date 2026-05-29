"""Source full-length audio for a track so it can be force-aligned.

Lucid Lyrics runs inside Spotify and can only hand us a track id + metadata, not
an audio file -- so the API fetches the audio itself.

:class:`LibrespotSource` pulls the *exact* Spotify master by track id via
librespot-python: best alignment, no wrong-version drift, and it covers every
Spotify track. Needs a stored Spotify credentials file (personal Premium
account); configure with ``LIBRESPOT_CREDENTIALS=/path/to/credentials.json``.

A YouTube/yt-dlp fallback was deliberately dropped: a fuzzy duration match can
pick a different upload (live/remaster/sped-up) and silently desync the whole
song, which is worse than returning "couldn't sync" and keeping the unsynced
lyrics. The :class:`AudioSource` interface remains so a fallback can be slotted
back in if librespot ever proves insufficient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrackRef:
    spotify_id: str                 # bare id or full "spotify:track:..." uri
    title: str = ""
    artist: str = ""
    duration_ms: int = 0            # Spotify-reported duration, for yt match

    @property
    def uri(self) -> str:
        if self.spotify_id.startswith("spotify:track:"):
            return self.spotify_id
        return f"spotify:track:{self.spotify_id}"


class SourceError(RuntimeError):
    pass


class AudioSource:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def fetch(self, track: TrackRef, dst_dir: str) -> str:
        """Download audio for ``track`` into ``dst_dir``; return the file path."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class LibrespotSource(AudioSource):
    name = "librespot"

    def __init__(self, credentials: str | None = None):
        self.credentials = credentials or os.environ.get("LIBRESPOT_CREDENTIALS")

    def available(self) -> bool:
        if not self.credentials or not Path(self.credentials).is_file():
            return False
        try:
            import librespot  # noqa: F401
        except ImportError:
            return False
        return True

    # apresolve returns several access points (ports 4070/443/80); librespot
    # sometimes picks one that's firewalled (4070 commonly is) -> ConnectionRefused.
    # Retrying re-resolves and usually lands on a reachable AP.
    CONNECT_ATTEMPTS = 4

    def _create_session(self):
        import time

        from librespot.core import Session

        last = None
        for attempt in range(self.CONNECT_ATTEMPTS):
            try:
                return Session.Builder().stored_file(self.credentials).create()
            except Exception as exc:  # connection refused / AP resolve issues
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise SourceError(f"librespot could not connect after "
                          f"{self.CONNECT_ATTEMPTS} attempts: {last}")

    def fetch(self, track: TrackRef, dst_dir: str) -> str:
        from librespot.audio.decoders import AudioQuality, VorbisOnlyAudioQuality
        from librespot.metadata import TrackId

        session = self._create_session()
        try:
            track_id = TrackId.from_uri(track.uri)
            stream = session.content_feeder().load(
                track_id,
                VorbisOnlyAudioQuality(AudioQuality.VERY_HIGH),
                False, None,
            )
            dst = str(Path(dst_dir) / "librespot.ogg")
            with open(dst, "wb") as f:
                while True:
                    chunk = stream.input_stream.stream().read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        finally:
            session.close()
        if not Path(dst).stat().st_size:
            raise SourceError("librespot returned empty audio")
        return dst


# --------------------------------------------------------------------------- #
def source_audio(track: TrackRef, dst_dir: str, *,
                 prefer: tuple[AudioSource, ...] | None = None) -> tuple[str, str]:
    """Fetch audio for ``track`` into ``dst_dir``.

    Returns ``(audio_path, source_name)``. Tries each source in order, skipping
    unavailable ones, and raises :class:`SourceError` if all fail.
    """
    sources = prefer or (LibrespotSource(),)
    errors = []
    for src in sources:
        if not src.available():
            errors.append(f"{src.name}: unavailable")
            continue
        try:
            return src.fetch(track, dst_dir), src.name
        except Exception as exc:  # try the next source
            errors.append(f"{src.name}: {type(exc).__name__}: {exc}")
    raise SourceError("all audio sources failed -> " + " | ".join(errors))
