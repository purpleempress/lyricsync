"""Serialize aligned lyrics to spicy-lyrics JSON.

Times are emitted in **seconds** (float). spicy-lyrics compares them directly
against ``progress/1000`` (see ``lyrics.ts``: "Content's StartTime/EndTime (s)")
and multiplies by 1000 via ``ConvertTime`` for animation durations -- so a
millisecond value here reads as thousands of seconds and no line ever activates.
See https://github.com/Spikerko/spicy-lyrics (src/utils/Lyrics).

Default is the **Line** schema (``Type:"Line"``, one ``Content`` item per line
with ``Text``/``StartTime``/``EndTime``) -- the stable, well-rendered path. Pass
``word_level=True`` for the ``Type:"Syllable"`` schema (per-word karaoke).
"""

from __future__ import annotations

from .model import AlignedLyrics


def _sec(t: float) -> float:
    return round(t, 3)


def _to_line(a: AlignedLyrics) -> dict:
    content = [{
        "Text": ln.text,
        "StartTime": _sec(ln.begin),
        "EndTime": _sec(ln.end),
    } for ln in a.lines]
    doc = {
        "Type": "Line",
        "StartTime": _sec(a.first_begin),
        "source": "aml",
        "Content": content,
    }
    if a.songwriters:
        doc["SongWriters"] = a.songwriters
    return doc


def _to_syllable(a: AlignedLyrics) -> dict:
    content = []
    for ln in a.lines:
        if ln.is_nonlexical:
            # Musical-interlude / empty line: a vocal line with no syllables.
            content.append({
                "Type": "Vocal",
                "Lead": {"StartTime": _sec(ln.begin), "EndTime": _sec(ln.end),
                         "Syllables": []},
            })
            continue
        # spicy-lyrics renders syllable Text verbatim with no space inserted
        # between standalone spans, so the word-separating space must live in
        # Text. IsPartOfWord=true means this syllable joins the next with no
        # space (mid-word karaoke split); otherwise it ends a word -> append a
        # trailing space.
        syllables = [{
            "Text": w.text + ("" if w.is_part_of_word else " "),
            "StartTime": _sec(w.begin),
            "EndTime": _sec(w.end),
            "IsPartOfWord": w.is_part_of_word,
        } for w in ln.words]
        content.append({
            "Type": "Vocal",
            "Lead": {"StartTime": _sec(ln.begin), "EndTime": _sec(ln.end),
                     "Syllables": syllables},
        })

    doc = {
        "Type": "Syllable",
        "StartTime": _sec(a.first_begin),
        "source": "aml",
        "Content": content,
    }
    if a.songwriters:
        doc["SongWriters"] = a.songwriters
    return doc


def to_spicy(a: AlignedLyrics, *, word_level: bool = False) -> dict:
    return _to_syllable(a) if word_level else _to_line(a)
