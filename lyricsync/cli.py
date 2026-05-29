"""Command-line entry: ``lyricsync audio.opus lyrics.ttml``.

Writes ``<lyrics>.synced.ttml`` and ``<lyrics>.synced.json`` next to the input
(or to ``--out-ttml`` / ``--out-json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ttml_in
from .align import align
from .spicy_json import to_spicy
from .ttml_out import to_ttml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lyricsync", description=__doc__)
    ap.add_argument("audio", help="audio file (.opus/.mp3/.wav/...)")
    ap.add_argument("lyrics", help="unsynced lyrics TTML file")
    # (the two positionals are auto-detected by extension below, so order
    #  doesn't actually matter)
    ap.add_argument("--out-ttml", help="output synced TTML path")
    ap.add_argument("--out-json", help="output spicy-lyrics JSON path")
    ap.add_argument("--word-level", action="store_true",
                    help="per-word karaoke timing (default: line-level)")
    ap.add_argument("--no-demucs", action="store_true",
                    help="skip Demucs vocal isolation (faster, less accurate)")
    ap.add_argument("--model", default="small", help="Whisper model name")
    ap.add_argument("--language", default=None, help="override language code")
    args = ap.parse_args(argv)

    # Order-agnostic: the lyrics file is the .ttml/.xml one.
    audio_path, lyrics_path = args.audio, args.lyrics
    lyric_exts = (".ttml", ".xml")
    if (not lyrics_path.lower().endswith(lyric_exts)
            and audio_path.lower().endswith(lyric_exts)):
        audio_path, lyrics_path = lyrics_path, audio_path

    parsed = ttml_in.parse(lyrics_path)
    print(f"[lyricsync] {len(parsed.lines)} lines, "
          f"{sum(len(l.words) for l in parsed.lines)} words; aligning…",
          file=sys.stderr)

    aligned = align(audio_path, parsed, demucs=not args.no_demucs,
                    model_name=args.model, language=args.language)

    stem = Path(lyrics_path).with_suffix("")
    out_ttml = Path(args.out_ttml) if args.out_ttml else Path(f"{stem}.synced.ttml")
    out_json = Path(args.out_json) if args.out_json else Path(f"{stem}.synced.json")

    out_ttml.write_text(to_ttml(aligned, word_level=args.word_level),
                        encoding="utf-8")
    out_json.write_text(
        json.dumps(to_spicy(aligned, word_level=args.word_level),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lyricsync] wrote {out_ttml}\n[lyricsync] wrote {out_json}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
