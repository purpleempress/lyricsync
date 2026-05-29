"""FastAPI service for LyricSync.

Alignment is CPU-bound and slow (tens of seconds, more with Demucs + sourcing),
so jobs run in a background thread pool and clients poll:

    POST /sync     JSON: {spotifyId, title, artist, duration(ms),
                          lyrics:[str] | ttml:str, wordLevel?, demucs?, model?}
                   -> {"job_id": "..."}            (Lucid Lyrics integration)
    POST /align    multipart: audio=<file>, lyrics_ttml=<file> | lyrics_text=<str>
                   -> {"job_id": "..."}            (local file testing)
    GET  /jobs/{id} -> {"status": ..., "result": <Lucid JSON>, "source": ...}
    GET  /health

``/sync`` sources the audio itself from the Spotify track id (librespot), since
Lucid runs inside Spotify and can't supply an audio file.

Config (env):
  LIBRESPOT_CREDENTIALS  path to stored Spotify credentials.json (required for /sync)
  LYRICSYNC_CORS         comma-separated allowed origins (default "*")
  LYRICSYNC_SYNC=1       make POSTs block and return the result inline (testing)
Run on the LAN with:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lyricsync import ttml_in
from lyricsync.align import _stage, _tlog, align
from lyricsync.lucid_json import to_lucid
from lyricsync.sources import TrackRef, source_audio
from lyricsync.ttml_out import to_ttml

app = FastAPI(title="LyricSync", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("LYRICSYNC_CORS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Single worker: alignment is CPU-bound (and itself multi-threaded), so running
# jobs one-at-a-time avoids thrashing. Complements the client's single-flight.
_pool = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
# Dedup: cache-key -> job_id of the in-flight (queued/running) job for that key.
_inflight: dict[str, str] = {}

_CACHE_DIR = Path(os.environ.get(
    "LYRICSYNC_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "sync"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _set(job_id: str, **kw) -> None:
    _jobs[job_id] = kw


def _sync_key(req: "SyncRequest") -> str:
    safe = "".join(c if c.isalnum() else "_" for c in req.spotifyId)[:64]
    return f"{safe}-{'word' if req.wordLevel else 'line'}-{req.model}-{int(req.demucs)}"


def _cache_load(key: str) -> dict | None:
    path = _CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def _cache_store(key: str, result: dict) -> None:
    try:
        (_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(result, ensure_ascii=False), "utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# /sync -- source audio by Spotify id, align, return Lucid JSON
# --------------------------------------------------------------------------- #
class SyncRequest(BaseModel):
    spotifyId: str
    title: str = ""
    artist: str = ""
    duration: int = 0                 # ms
    lyrics: list[str] | None = None   # plain unsynced lines (Lucid Static)
    ttml: str | None = None           # alternative raw TTML
    wordLevel: bool = False
    demucs: bool = True
    model: str = "small"
    language: str | None = None


def _run_sync(job_id: str, req: SyncRequest, key: str) -> None:
    _set(job_id, status="running")
    t_job = time.perf_counter()
    try:
        if req.lyrics:
            parsed = ttml_in.from_lines(req.lyrics, lang=req.language or "en")
        elif req.ttml:
            parsed = ttml_in.parse(req.ttml)
        else:
            raise ValueError("provide `lyrics` (list of lines) or `ttml`")

        with tempfile.TemporaryDirectory() as tmp:
            track = TrackRef(req.spotifyId, req.title, req.artist, req.duration)
            with _stage("source_audio"):   # librespot pull (network + decrypt)
                audio_path, src = source_audio(track, tmp)
            aligned = align(audio_path, parsed, demucs=req.demucs,
                            model_name=req.model, language=req.language)
        with _stage("to_lucid"):
            result = to_lucid(aligned, id=req.spotifyId, word_level=req.wordLevel)
        _cache_store(key, result)
        _tlog(f"{'JOB TOTAL':<13} {time.perf_counter() - t_job:7.2f}s "
              f"(source={src})")
        _set(job_id, status="done", result=result, source=src)
    except Exception as exc:
        _set(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if _inflight.get(key) == job_id:
            _inflight.pop(key, None)


@app.post("/sync")
async def sync_endpoint(req: SyncRequest):
    key = _sync_key(req)

    # 1) Server-side cache: skip sourcing + alignment entirely on a hit.
    cached = _cache_load(key)
    if cached is not None:
        job_id = uuid.uuid4().hex
        _set(job_id, status="done", result=cached, source="cache")
        return {"job_id": job_id}

    # 2) Dedup/debounce: an identical request already in flight -> share its job.
    existing = _inflight.get(key)
    if existing and _jobs.get(existing, {}).get("status") in ("queued", "running"):
        return {"job_id": existing}

    job_id = uuid.uuid4().hex
    if os.environ.get("LYRICSYNC_SYNC") == "1":
        _run_sync(job_id, req, key)
        return _jobs.pop(job_id)
    _inflight[key] = job_id
    _set(job_id, status="queued")
    _pool.submit(_run_sync, job_id, req, key)
    return {"job_id": job_id}


# --------------------------------------------------------------------------- #
# /align -- local file testing (audio uploaded directly)
# --------------------------------------------------------------------------- #
def _run_align(job_id: str, audio_bytes: bytes, suffix: str, lyrics: str,
               demucs: bool, model: str, language: str | None,
               word_level: bool) -> None:
    _set(job_id, status="running")
    try:
        parsed = ttml_in.parse(lyrics)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name
        try:
            aligned = align(audio_path, parsed, demucs=demucs,
                            model_name=model, language=language)
        finally:
            os.unlink(audio_path)
        _set(job_id, status="done", result={
            "ttml": to_ttml(aligned, word_level=word_level),
            "lucidJson": to_lucid(aligned, id="", word_level=word_level),
        })
    except Exception as exc:
        _set(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/align")
async def align_endpoint(
    audio: UploadFile = File(...),
    lyrics_ttml: UploadFile | None = File(None),
    lyrics_text: str | None = Form(None),
    demucs: bool = True,
    model: str = "small",
    language: str | None = None,
    word_level: bool = False,
):
    if lyrics_ttml is None and not lyrics_text:
        raise HTTPException(400, "provide lyrics_ttml file or lyrics_text")
    lyrics = (await lyrics_ttml.read()).decode("utf-8") if lyrics_ttml else lyrics_text
    audio_bytes = await audio.read()
    suffix = Path(audio.filename or "audio.opus").suffix or ".opus"
    job_id = uuid.uuid4().hex
    args = (job_id, audio_bytes, suffix, lyrics, demucs, model, language, word_level)
    if os.environ.get("LYRICSYNC_SYNC") == "1":
        _run_align(*args)
        return _jobs.pop(job_id)
    _set(job_id, status="queued")
    _pool.submit(_run_align, *args)
    return {"job_id": job_id}


# --------------------------------------------------------------------------- #
@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/health")
async def health():
    return {"status": "ok"}
