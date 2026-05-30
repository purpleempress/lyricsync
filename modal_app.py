"""Modal GPU deployment of LyricSync's compute stage.

Demucs vocal isolation and faster-whisper forced alignment are the only slow,
GPU-friendly parts of the pipeline -- tens of seconds each on CPU, a few
seconds on a cheap GPU. This file deploys them as a single Modal function;
``lyricsync.align`` calls it (via ``LYRICSYNC_BACKEND=modal``) and keeps the
cheap parse / map / rebuild work local.

One-time setup (the user already has ~/.modal.toml):
    pip install modal           # or:  pip install -e ".[modal]"
    modal deploy modal_app.py   # build image + deploy the function

Then run the API with offload enabled:
    LYRICSYNC_BACKEND=modal bash scripts/run_api.sh

The function returns plain JSON (``{"dur": float, "aligned": [[word, begin,
end], ...]}``) so nothing lyricsync-specific has to be unpacked on return.
"""

import os

import modal

# faster-whisper's default model size; pre-baked so the common request never
# pays a cold-start download. Other sizes download on first use at runtime.
_PREFETCH_WHISPER = "small"

# How long a warm GPU container lingers after a job before scaling to zero.
# Read at deploy time (scripts/deploy_modal.sh sources .env), so changing it
# means re-deploying. RunPod's equivalent is the endpoint's Idle Timeout in the
# console -- the worker image can't set it, so it isn't driven from here.
_IDLE_SECS = int(os.environ.get("LYRICSYNC_GPU_IDLE_SECS", "300"))


def _download_models() -> None:
    """Bake htdemucs + the default whisper weights + Silero VAD into the layer."""
    from demucs.pretrained import get_model

    get_model("htdemucs")
    import stable_whisper

    stable_whisper.load_faster_whisper(
        _PREFETCH_WHISPER, device="cpu", compute_type="int8")

    # vad=True (default for music) loads Silero via torch.hub on first use;
    # bake it into the image layer so a cold container never downloads it. Call
    # stable-ts directly, not lyricsync._load_silero_vad: this build step runs
    # before add_local_python_source, so `lyricsync` isn't importable here yet.
    from stable_whisper.stabilization.silero_vad import load_silero_vad_model

    load_silero_vad_model()


# Base on NVIDIA's CUDA *runtime* image rather than debian_slim: it ships
# libcublas + cuDNN registered with ldconfig system-wide. faster-whisper's
# CTranslate2 backend dlopen's libcublas.so.12 from the default loader path and
# can't see the copies torch bundles in site-packages, so on debian_slim it dies
# with "libcublas.so.12 is not found". The `cudnn` (unversioned) tag on CUDA
# 12.x is cuDNN 9, matching current CTranslate2. add_python gives us a clean
# 3.12 to pip into. Torch's own CUDA wheel still works on top (it RPATHs its
# bundled libs); the base image is only there to satisfy CTranslate2.
#
# lxml is pulled in because importing lyricsync.align transitively imports
# lyricsync.ttml_in.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    # torch built for CUDA 12.4 (matches the base). The default PyPI wheel is now
    # a CUDA 13 (+cu130) build that fails on GPU drivers older than CUDA 13
    # ("NVIDIA driver too old"); cu124 runs on any >=12.4 driver. Pinned first so
    # demucs's torch dep resolves to this build.
    .pip_install(
        "torch", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "stable-ts>=2.17.0",
        "faster-whisper>=1.0",
        "demucs>=4.0",
        "soundfile>=0.12",
        "lxml>=5.0",
    )
    # Pre-download weights (heavy, cached) before adding local source (cheap,
    # changes often) so editing lyricsync doesn't invalidate the model layer.
    .run_function(_download_models)
    # Ship our package so the GPU function reuses the exact same compute code as
    # the local path -- no logic drift between the CPU and GPU branches.
    .add_local_python_source("lyricsync")
)

app = modal.App("lyricsync", image=image)


# A class (not a bare function) so we get an @modal.enter lifecycle hook, and
# memory snapshots so cold starts *restore* the loaded models instead of
# re-reading them off disk + re-initializing torch/CTranslate2.
#
# enable_gpu_snapshot (alpha) extends the snapshot to GPU memory: the weights we
# load onto the GPU in `_warm` are captured in VRAM, so a cold-started container
# comes back with both models already resident -- no disk load, no host->device
# copy. Without it, only CPU state snapshots and the GPU weights would reload
# (CTranslate2 in particular can't be moved CPU->GPU after load, so a CPU-only
# snapshot wouldn't help the whisper side at all).
@app.cls(
    gpu="T4",
    timeout=900,
    scaledown_window=_IDLE_SECS,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Aligner:
    @modal.enter(snap=True)
    def _warm(self):
        """Load both models into (GPU) memory so the snapshot captures them.

        GPU snapshots make the device available here, so we load whisper
        straight onto CUDA. Both loads populate the module-level lru_cache in
        ``lyricsync.align``; the snapshot freezes that cache (and the VRAM
        behind it), and ``_compute_alignment`` later hits it for free.
        """
        from lyricsync.align import _load_demucs, _load_model, _load_silero_vad

        _load_demucs("htdemucs")                      # torch weights (CPU-resident)
        _load_model(_PREFETCH_WHISPER, device="cuda")  # CTranslate2 weights (VRAM)
        _load_silero_vad()                            # Silero VAD (frozen in snapshot)

    @modal.method()
    def transcribe_align(
        self,
        audio_bytes: bytes,
        suffix: str,
        alignment_text: str,
        language: str,
        demucs: bool,
        model_name: str,
    ) -> dict:
        """Decode + (Demucs) + forced-align ``audio_bytes`` on a GPU.

        Returns ``{"dur": <seconds>, "aligned": [[word, begin, end], ...]}``.
        """
        import os
        import tempfile

        from lyricsync.align import _compute_alignment

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            dur, aligned = _compute_alignment(
                path, alignment_text,
                demucs=demucs, model_name=model_name, language=language,
                device="cuda",
            )
        finally:
            os.unlink(path)
        return {"dur": dur, "aligned": [list(w) for w in aligned]}
