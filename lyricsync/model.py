"""Internal aligned-lyrics datamodel.

The single source of truth consumed by both output writers (ttml_out,
spicy_json). Times are stored in **seconds** (float).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    begin: float
    end: float
    # True when the token has no trailing space -- it joins to the next word
    # (e.g. a hyphenated fragment "lo-"). Maps to spicy-lyrics IsPartOfWord.
    is_part_of_word: bool = False


@dataclass
class Line:
    key: str                     # itunes:key, e.g. "L1"
    agent: str = "v1"            # ttm:agent xml:id
    begin: float = 0.0
    end: float = 0.0
    # Raw line text as it should be displayed (normalized, spaces preserved).
    text: str = ""
    # Non-lexical lines (empty, "♪") carry no alignable words; timed by
    # interpolation between neighbours.
    is_nonlexical: bool = False
    words: list[Word] = field(default_factory=list)


@dataclass
class AlignedLyrics:
    dur: float = 0.0             # body duration / track length, seconds
    lang: str = "en"
    songwriters: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=lambda: ["v1"])
    lines: list[Line] = field(default_factory=list)

    @property
    def first_begin(self) -> float:
        for ln in self.lines:
            if not ln.is_nonlexical:
                return ln.begin
        return self.lines[0].begin if self.lines else 0.0
