"""RunPod serverless worker for LyricSync's GPU compute stage.

The RunPod counterpart to ``modal_app.py``: same `_compute_alignment` code, just
behind RunPod's serverless queue instead of Modal. Audio arrives base64-encoded
in the job input (RunPod queues are JSON-only); we decode, run decode + Demucs +
forced alignment on the GPU, and return the raw word timings as JSON.

The models are loaded once at module import, *outside* the handler. RunPod's
**FlashBoot** keeps a warmed worker process alive between requests, so a reused
worker already has htdemucs (CPU-resident) and whisper (VRAM-resident) loaded --
no per-request disk load. This is RunPod's analogue to Modal's memory snapshots.

Build + push with ``scripts/build_runpod.sh``, then create a Serverless endpoint
from the pushed image (FlashBoot is on by default) and note its Endpoint ID.
"""

import base64
import os
import tempfile

import runpod

from lyricsync.align import _compute_alignment, _load_demucs, _load_model

# Warm both models at import so FlashBoot-reused workers skip the load entirely.
# A serverless worker owns its GPU for its whole life, so loading whisper onto
# CUDA here is fine. `model_name` other than the default still loads on demand.
_PREFETCH_WHISPER = "small"
_load_demucs("htdemucs")                       # torch weights (CPU-resident)
_load_model(_PREFETCH_WHISPER, device="cuda")  # CTranslate2 weights (VRAM)


def handler(job: dict) -> dict:
    """Decode + (Demucs) + forced-align the job's audio on the GPU.

    Returns ``{"dur": <seconds>, "aligned": [[word, begin, end], ...]}``.
    """
    inp = job["input"]
    audio = base64.b64decode(inp["audio_b64"])
    suffix = inp.get("suffix", ".opus")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        dur, aligned = _compute_alignment(
            path, inp["alignment_text"],
            demucs=inp.get("demucs", True),
            model_name=inp.get("model_name", "small"),
            language=inp.get("language", "en"),
            device="cuda",
        )
    finally:
        os.unlink(path)
    return {"dur": dur, "aligned": [list(w) for w in aligned]}


runpod.serverless.start({"handler": handler})
