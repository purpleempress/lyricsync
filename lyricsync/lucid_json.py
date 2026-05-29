"""Serialize aligned lyrics to the Lucid Lyrics data model.

Target: https://gitlab.com/sanoojes/lucid-lyrics (src/lib/api/types.ts).

Times are **seconds** (float) -- Lucid multiplies by 1000 to compare against the
ms playback position (``LineLyrics.tsx``: ``c.StartTime * 1000``). Default is
``Type:"Line"``; ``word_level=True`` emits ``Type:"Syllable"`` (karaoke). Empty
/ ``♪`` lines become ``Type:"Interlude"`` content with ``Text:" "`` (matching
Lucid's own LRCLIB parser).
"""

from __future__ import annotations

from .model import AlignedLyrics


def _sec(t: float) -> float:
    return round(t, 3)


def _line_content(a: AlignedLyrics) -> list[dict]:
    content = []
    for ln in a.lines:
        if ln.is_nonlexical:
            content.append({
                "Type": "Interlude",
                "Text": " ",
                "StartTime": _sec(ln.begin),
                "EndTime": _sec(ln.end),
                "OppositeAligned": False,
            })
        else:
            content.append({
                "Type": "Line",
                "Text": ln.text,
                "StartTime": _sec(ln.begin),
                "EndTime": _sec(ln.end),
                "OppositeAligned": False,
            })
    return content


def _syllable_content(a: AlignedLyrics) -> list[dict]:
    content = []
    for ln in a.lines:
        if ln.is_nonlexical:
            syllables = ([] if not ln.text else
                         [{"Text": ln.text, "IsPartOfWord": False,
                           "StartTime": _sec(ln.begin), "EndTime": _sec(ln.end)}])
        else:
            # Lucid spaces words via the `trailing-whitespace` CSS class (driven
            # by IsPartOfWord), rendering Text verbatim -- so NO trailing space
            # here (unlike the spicy-lyrics adapter, which needs it).
            syllables = [{
                "Text": w.text,
                "IsPartOfWord": w.is_part_of_word,
                "StartTime": _sec(w.begin),
                "EndTime": _sec(w.end),
            } for w in ln.words]
        content.append({
            "Type": "Vocal",
            "OppositeAligned": False,
            "Lead": {
                "StartTime": _sec(ln.begin),
                "EndTime": _sec(ln.end),
                "Syllables": syllables,
            },
        })
    return content


def _span(item: dict) -> tuple[float, float]:
    """(StartTime, EndTime) of a content item, line- or syllable-shaped."""
    node = item["Lead"] if "Lead" in item else item
    return node["StartTime"], node["EndTime"]


def to_lucid(a: AlignedLyrics, *, id: str = "", provider: str = "lyricsync",
             word_level: bool = False) -> dict:
    content = _syllable_content(a) if word_level else _line_content(a)
    start = _span(content[0])[0] if content else 0.0
    end = _span(content[-1])[1] if content else 0.0
    return {
        "Id": id,
        "Type": "Syllable" if word_level else "Line",
        "Provider": provider,
        "SongWriters": a.songwriters,
        "Content": content,
        "StartTime": _sec(start),
        "EndTime": _sec(end),
    }
