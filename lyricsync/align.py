"""Forced alignment of known lyrics to song audio.

Pipeline: ffmpeg decode -> (optional) Demucs vocal isolation -> stable-ts
``align()`` of the *known* lyric text -> per-word timestamps mapped back onto
the original line/token structure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from .model import AlignedLyrics, Line, Word
from .ttml_in import ParsedLyrics


# --------------------------------------------------------------------------- #
# Timing instrumentation (debug)
# --------------------------------------------------------------------------- #
# Stage timings print to stderr (so they show in `docker compose logs`) when
# LYRICSYNC_TIMING is set. Unset -> zero overhead, no output. Remove with
# `git apply -R` once the slow stage is identified.
_TIMING = os.environ.get("LYRICSYNC_TIMING", "1") not in ("", "0", "false")


def _tlog(msg: str) -> None:
    if _TIMING:
        print(f"[timing] {msg}", file=sys.stderr, flush=True)


@contextmanager
def _stage(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _tlog(f"{name:<13} {time.perf_counter() - t0:7.2f}s")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str = "cpu",
                compute_type: str | None = None):
    """Load a faster-whisper (CTranslate2) model, cached for the process.

    On CPU, ``compute_type="int8"`` is markedly faster than the
    openai-whisper/torch backend, and for *forced alignment* (the words are
    already known) int8 quantization costs basically no accuracy. On GPU
    (``device="cuda"``, the Modal path) ``float16`` is both faster and well
    within VRAM. The cache means a long-lived service / warm Modal container
    loads weights once rather than per job; keyed on (model, device,
    compute_type) so the CPU and GPU variants don't collide.
    """
    import stable_whisper

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return stable_whisper.load_faster_whisper(
        model_name, device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def _load_demucs(name: str = "htdemucs"):
    """Load a Demucs separation model, cached for the process.

    Same rationale as ``_load_model``: ``get_model`` re-reads the weights from
    disk every call, so loading it per job wastes seconds on every demucs-on
    request. Cache it once. ``.cpu().eval()`` is idempotent, so it's safe to
    apply at load time. Single worker -> no cache race.
    """
    from demucs.pretrained import get_model

    model = get_model(name)
    model.cpu().eval()
    return model


@lru_cache(maxsize=1)
def _load_silero_vad():
    """Pre-load stable-ts's Silero VAD so ``align(vad=True)`` doesn't stall.

    stable-ts loads Silero lazily via ``torch.hub`` on the first ``vad=True``
    alignment, which both downloads the snakers4/silero-vad repo (a cold-start
    cost) and re-fetches it per process. Calling this warms stable-ts's own
    process-level ``cached_model_instances`` cache so later alignments reuse it
    -- same rationale as ``_load_model`` / ``_load_demucs``. At image-build time
    it also populates the ``torch.hub`` cache into the baked layer, so a cold
    GPU worker never downloads it. Import is lazy and the path is internal to
    stable-ts, so failure here must not be fatal: VAD would simply lazy-load.
    """
    try:
        from stable_whisper.stabilization.silero_vad import load_silero_vad_model

        return load_silero_vad_model()
    except Exception as exc:  # pragma: no cover - warm-only optimization
        _tlog(f"silero_vad    warm skipped ({type(exc).__name__})")
        return None


# --------------------------------------------------------------------------- #
# Audio prep
# --------------------------------------------------------------------------- #
def probe_duration(audio_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", audio_path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def _decode_wav(audio_path: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
         "-vn", dst],
        capture_output=True, check=True,
    )


def _encode_align_audio(audio_path: str) -> bytes:
    """16 kHz mono Opus of the track, for shipping to a remote worker.

    The pipeline decodes to 16 kHz mono before Demucs regardless (see
    ``_decode_wav`` / ``_isolate_vocals``), so this is lossless versus what the
    worker would do anyway -- but it shrinks a multi-MB master to ~1-2 MB, well
    under RunPod's 20 MB request-body cap that a base64'd full track blows past.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "align.opus")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
             "-vn", "-c:a", "libopus", "-b:a", "48k", out],
            capture_output=True, check=True,
        )
        return Path(out).read_bytes()


def _isolate_vocals(wav_path: str, workdir: str, device: str = "cpu",
                    *, shifts: int = 1, overlap: float = 0.25) -> str:
    """Isolate the vocal stem with Demucs; return the written WAV path.

    Driven through Demucs's Python API and saved via ``soundfile`` rather than
    the CLI: torchaudio 2.x delegates saving to ``torchcodec`` (often absent),
    so the ``demucs`` CLI's save step fails. In-process also avoids a second
    model load. ``apply_model`` moves the (cpu-loaded) weights onto ``device``
    itself, so ``device="cuda"`` on the Modal path needs nothing else here.

    ``shifts`` / ``overlap`` are demucs speed/quality knobs: ``overlap`` is the
    fraction each chunk shares with its neighbour (lower = less compute),
    ``shifts`` averages N randomly-offset passes (0 = single pass).
    """
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    model = _load_demucs("htdemucs")

    wav = AudioFile(wav_path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, progress=False,
                              shifts=shifts, overlap=overlap)[0]
    sources = sources * ref.std() + ref.mean()
    vocals = sources[model.sources.index("vocals")]

    out = str(Path(workdir) / "vocals.wav")
    sf.write(out, vocals.T.numpy(), model.samplerate)
    return out


# --------------------------------------------------------------------------- #
# Compute stage: decode -> (demucs) -> forced align.  Runs either locally on
# CPU or, when offloaded, inside the Modal GPU container (see ``modal_app.py``).
# Returns plain JSON-serializable data so it can cross the Modal boundary.
# --------------------------------------------------------------------------- #
def _align_opts(fast_mode: bool | None,
                nonspeech_skip: float | None,
                vad: bool | None) -> dict:
    """Resolve stable-ts ``align()`` speed knobs, env overriding the defaults.

    Explicit args win; otherwise ``LYRICSYNC_FAST_MODE`` (truthy),
    ``LYRICSYNC_NONSPEECH_SKIP`` (seconds, or "none" to disable skipping) and
    ``LYRICSYNC_VAD`` (truthy, or "0"/"false" to disable) apply. Defaults match
    stable-ts except ``vad``: ``fast_mode=False``, ``nonspeech_skip=5.0``, and
    ``vad=True`` -- on by default since we align music, where Silero VAD on the
    (Demucs-isolated) vocal stem keeps word timings off instrumental rests.
    """
    if fast_mode is None:
        fast_mode = os.environ.get("LYRICSYNC_FAST_MODE", "") not in ("", "0", "false")
    if nonspeech_skip is None:
        ns = os.environ.get("LYRICSYNC_NONSPEECH_SKIP")
        nonspeech_skip = (5.0 if ns is None
                          else None if ns.strip().lower() in ("none", "")
                          else float(ns))
    if vad is None:
        vad = os.environ.get("LYRICSYNC_VAD", "1") not in ("0", "false")
    return {"fast_mode": fast_mode, "nonspeech_skip": nonspeech_skip, "vad": vad}


def _demucs_opts(shifts: int | None, overlap: float | None) -> dict:
    """Resolve Demucs speed knobs, env overriding the defaults.

    Explicit args win; otherwise ``LYRICSYNC_DEMUCS_SHIFTS`` /
    ``LYRICSYNC_DEMUCS_OVERLAP`` apply. Defaults match demucs: ``shifts=1``,
    ``overlap=0.25``.
    """
    if shifts is None:
        s = os.environ.get("LYRICSYNC_DEMUCS_SHIFTS")
        shifts = 1 if s is None else int(s)
    if overlap is None:
        o = os.environ.get("LYRICSYNC_DEMUCS_OVERLAP")
        overlap = 0.25 if o is None else float(o)
    return {"shifts": shifts, "overlap": overlap}


def _compute_alignment(
    audio_path: str,
    alignment_text: str,
    *,
    demucs: bool = True,
    model_name: str = "small",
    language: str = "en",
    device: str = "cpu",
    fast_mode: bool | None = None,
    nonspeech_skip: float | None = None,
    vad: bool | None = None,
    demucs_shifts: int | None = None,
    demucs_overlap: float | None = None,
) -> tuple[float, list[tuple[str, float, float]]]:
    """The GPU-heavy half of :func:`align`, isolated so it can run remotely.

    Returns ``(duration_seconds, [(word, begin, end), ...])`` -- the raw
    stable-ts word timings, before they're mapped back onto the lyric
    structure (that mapping is cheap and stays on the caller). The ``fast_mode``
    / ``nonspeech_skip`` / ``vad`` (stable-ts) and ``demucs_shifts`` /
    ``demucs_overlap`` (Demucs) knobs fall back to env when left None (see
    ``_align_opts`` / ``_demucs_opts``).
    """
    align_opts = _align_opts(fast_mode, nonspeech_skip, vad)
    dmx_opts = _demucs_opts(demucs_shifts, demucs_overlap)
    with _stage("probe"):
        dur = probe_duration(audio_path)

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio.wav")
        with _stage("decode"):
            _decode_wav(audio_path, wav)
        if demucs:
            with _stage("demucs_load"):  # ~0s warm; htdemucs weights on first job
                _load_demucs("htdemucs")
            with _stage("demucs"):
                align_audio = _isolate_vocals(wav, tmp, device=device, **dmx_opts)
        else:
            align_audio = wav
            _tlog("demucs        skipped")

        with _stage("model_load"):   # ~0s on a warm cache; large on first job
            model = _load_model(model_name, device=device)
        with _stage("align"):
            result = model.align(align_audio, alignment_text, language=language,
                                 **align_opts)

    if hasattr(result, "all_words"):
        words = result.all_words()
    else:
        words = [w for seg in result.segments for w in seg.words]
    return dur, [(w.word, float(w.start), float(w.end)) for w in words]


# Where the GPU-heavy compute stage runs. LYRICSYNC_BACKEND picks one of
# "local" (CPU, default), "modal", or "runpod".
def _backend() -> str:
    return os.environ.get("LYRICSYNC_BACKEND", "local").strip().lower() or "local"


def _align_on_modal(
    audio_path: str,
    alignment_text: str,
    *,
    demucs: bool,
    model_name: str,
    language: str,
) -> tuple[float, list[tuple[str, float, float]]]:
    """Run :func:`_compute_alignment` on the deployed Modal GPU class."""
    import modal

    app_name = os.environ.get("LYRICSYNC_MODAL_APP", "lyricsync")
    # 16 kHz mono Opus, same as the RunPod path: lossless vs the pipeline and a
    # fraction of the upload (Modal has no hard body cap, just no reason to ship
    # bytes the worker immediately downsamples away).
    audio_bytes = _encode_align_audio(audio_path)
    aligner = modal.Cls.from_name(app_name, "Aligner")()
    res = aligner.transcribe_align.remote(
        audio_bytes, ".opus", alignment_text, language, demucs, model_name)
    return res["dur"], [tuple(x) for x in res["aligned"]]


def _align_on_runpod(
    audio_path: str,
    alignment_text: str,
    *,
    demucs: bool,
    model_name: str,
    language: str,
) -> tuple[float, list[tuple[str, float, float]]]:
    """Run :func:`_compute_alignment` on a RunPod serverless endpoint.

    Posts the audio (base64 in JSON, the only thing RunPod's queue accepts) to
    ``/run`` and polls ``/status`` until the worker returns. Needs
    ``RUNPOD_ENDPOINT_ID`` + ``RUNPOD_API_KEY``; see ``runpod_handler.py`` for
    the matching worker side. Uses stdlib urllib so the API venv stays lean.
    """
    import base64
    import json
    import time as _time
    import urllib.request

    endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
    key = os.environ["RUNPOD_API_KEY"]
    base = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _call(path: str, body: dict | None = None, method: str = "POST") -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    # Ship 16 kHz mono Opus, not the raw master: RunPod's /run body caps at
    # 20 MB and a base64'd full track sails past it (-> 502 at the gateway).
    payload = {"input": {
        "audio_b64": base64.b64encode(_encode_align_audio(audio_path)).decode(),
        "suffix": ".opus",
        "alignment_text": alignment_text,
        "language": language,
        "demucs": demucs,
        "model_name": model_name,
    }}
    job_id = _call("/run", payload)["id"]

    deadline = _time.time() + 600
    while _time.time() < deadline:
        st = _call(f"/status/{job_id}", method="GET")
        status = st.get("status")
        if status == "COMPLETED":
            out = st["output"]
            return out["dur"], [tuple(x) for x in out["aligned"]]
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod job {status}: {st.get('error') or st}")
        _time.sleep(2)
    raise TimeoutError(f"RunPod job {job_id} did not finish within 600s")


# --------------------------------------------------------------------------- #
# Word -> token mapping (robust to tokenizer differences via char offsets)
# --------------------------------------------------------------------------- #
def _token_spans(tokens: list[str], target: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for tok in tokens:
        idx = target.find(tok, pos)
        if idx < 0:                      # shouldn't happen; target is built from tokens
            idx = pos
        spans.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return spans


def _aligned_spans(aligned: list[tuple[str, float, float]], target: str):
    low, cur, out = target.lower(), 0, []
    for text, s, e in aligned:
        t = text.strip().lower()
        if not t:
            continue
        idx = low.find(t, cur)
        if idx < 0:
            idx = low.find(t)
        if idx < 0:
            continue
        out.append((idx, idx + len(t), s, e))
        cur = idx + len(t)
    return out


def _map_times(tokens: list[str], aligned: list[tuple[str, float, float]]):
    """Return [(begin, end) | None] per token."""
    target = " ".join(tokens)
    tspans = _token_spans(tokens, target)
    aspans = _aligned_spans(aligned, target)
    result: list[tuple[float, float] | None] = []
    for ts, te in tspans:
        overl = [(s, e) for a0, a1, s, e in aspans if a0 < te and a1 > ts]
        result.append((min(s for s, _ in overl), max(e for _, e in overl))
                      if overl else None)
    return result


def _fill_gaps(times: list[tuple[float, float] | None], dur: float):
    """Interpolate begin/end for tokens stable-ts failed to place."""
    n = len(times)
    # forward-fill a floor and backward-fill a ceiling, then split evenly.
    for i in range(n):
        if times[i] is not None:
            continue
        prev_end = next((times[j][1] for j in range(i - 1, -1, -1)
                         if times[j] is not None), 0.0)
        nxt_begin = next((times[j][0] for j in range(i + 1, n)
                          if times[j] is not None), dur)
        # count the contiguous run of None to share the gap
        run = i
        while run < n and times[run] is None:
            run += 1
        span = (nxt_begin - prev_end) / (run - i + 1)
        for k in range(i, run):
            b = prev_end + span * (k - i)
            times[k] = (b, b + span)
    return times


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def align(
    audio_path: str,
    parsed: ParsedLyrics,
    *,
    demucs: bool = True,
    model_name: str = "small",
    language: str | None = None,
) -> AlignedLyrics:
    t_start = time.perf_counter()
    lang = language or parsed.lang or "en"

    backend = _backend()
    if backend == "modal":
        with _stage("modal"):    # GPU compute on Modal: decode + demucs + align
            dur, aligned = _align_on_modal(
                audio_path, parsed.alignment_text,
                demucs=demucs, model_name=model_name, language=lang)
    elif backend == "runpod":
        with _stage("runpod"):   # GPU compute on a RunPod serverless endpoint
            dur, aligned = _align_on_runpod(
                audio_path, parsed.alignment_text,
                demucs=demucs, model_name=model_name, language=lang)
    else:
        dur, aligned = _compute_alignment(
            audio_path, parsed.alignment_text,
            demucs=demucs, model_name=model_name, language=lang, device="cpu")

    with _stage("map+rebuild"):
        # Map onto our canonical lexical tokens.
        tokens = [w for ln in parsed.lines if not ln.is_nonlexical
                  for w in ln.words]
        times = _fill_gaps(_map_times(tokens, aligned), dur)

        # Rebuild lines.
        out = AlignedLyrics(dur=dur, lang=lang, songwriters=parsed.songwriters)
        cursor = 0
        for n, pl in enumerate(parsed.lines, start=1):
            line = Line(key=f"L{n}", agent="v1", text=pl.text,
                        is_nonlexical=pl.is_nonlexical)
            if not pl.is_nonlexical:
                for tok in pl.words:
                    b, e = times[cursor]
                    line.words.append(Word(text=tok, begin=b, end=e))
                    cursor += 1
                line.begin = line.words[0].begin
                line.end = line.words[-1].end
            out.lines.append(line)

        _time_nonlexical(out, dur)

    _tlog(f"{'TOTAL align':<13} {time.perf_counter() - t_start:7.2f}s "
          f"(demucs={'on' if demucs else 'off'}, model={model_name})")
    return out


def _time_nonlexical(out: AlignedLyrics, dur: float) -> None:
    """Give empty / ♪ lines begin/end by interpolating between neighbours."""
    lines = out.lines
    for i, ln in enumerate(lines):
        if not ln.is_nonlexical:
            continue
        prev_end = next((lines[j].end for j in range(i - 1, -1, -1)
                         if not lines[j].is_nonlexical), 0.0)
        nxt_begin = next((lines[j].begin for j in range(i + 1, len(lines))
                          if not lines[j].is_nonlexical), dur)
        ln.begin = prev_end
        ln.end = max(nxt_begin, prev_end)
