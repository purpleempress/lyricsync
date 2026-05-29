# LyricSync API — forced-alignment service.
FROM python:3.12-slim

# ffmpeg: decode audio + probe duration. libsndfile1: soundfile (Demucs stem write).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch/torchaudio first, from PyTorch's CPU index — keeps the image
# from pulling several GB of CUDA wheels it would never use.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio

# App + remaining deps (torch is already satisfied, so the project's torch pin
# resolves to the CPU build installed above).
COPY pyproject.toml README.md ./
COPY lyricsync ./lyricsync
COPY api ./api
RUN pip install --no-cache-dir ".[api]"

# Whisper/Demucs model weights and the sync cache all live under /app/.cache,
# which compose mounts as a named volume so they survive restarts.
ENV LIBRESPOT_CREDENTIALS=/app/credentials.json \
    LYRICSYNC_CACHE_DIR=/app/.cache/sync \
    LYRICSYNC_CORS=* \
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
