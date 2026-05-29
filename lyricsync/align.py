"""Forced alignment of known lyrics to song audio.

Pipeline: ffmpeg decode -> (optional) Demucs vocal isolation -> stable-ts
``align()`` of the *known* lyric text -> per-word timestamps mapped back onto
the original line/token structure.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from .model import AlignedLyrics, Line, Word
from .ttml_in import ParsedLyrics


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load a faster-whisper (CTranslate2) model, cached for the process.

    CTranslate2 with ``compute_type="int8"`` is markedly faster than the
    openai-whisper/torch backend on CPU, and for *forced alignment* (the words
    are already known) the int8 quantization costs basically no accuracy. The
    cache means a long-lived service loads weights once rather than per job;
    safe here because the API runs alignment single-threaded (one worker).
    """
    import stable_whisper

    return stable_whisper.load_faster_whisper(
        model_name, device="cpu", compute_type="int8")


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


def _isolate_vocals(wav_path: str, workdir: str) -> str:
    """Isolate the vocal stem with Demucs; return the written WAV path.

    Driven through Demucs's Python API and saved via ``soundfile`` rather than
    the CLI: torchaudio 2.x delegates saving to ``torchcodec`` (often absent),
    so the ``demucs`` CLI's save step fails. In-process also avoids a second
    model load.
    """
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.cpu().eval()

    wav = AudioFile(wav_path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    vocals = sources[model.sources.index("vocals")]

    out = str(Path(workdir) / "vocals.wav")
    sf.write(out, vocals.T.numpy(), model.samplerate)
    return out


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
    dur = probe_duration(audio_path)
    lang = language or parsed.lang or "en"

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio.wav")
        _decode_wav(audio_path, wav)
        align_audio = _isolate_vocals(wav, tmp) if demucs else wav

        model = _load_model(model_name)
        result = model.align(align_audio, parsed.alignment_text, language=lang)

    # Flatten aligned words.
    if hasattr(result, "all_words"):
        words = result.all_words()
    else:
        words = [w for seg in result.segments for w in seg.words]
    aligned = [(w.word, float(w.start), float(w.end)) for w in words]

    # Map onto our canonical lexical tokens.
    tokens = [w for ln in parsed.lines if not ln.is_nonlexical for w in ln.words]
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
