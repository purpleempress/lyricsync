"""Parse an unsynced TTML lyrics file into ordered, tokenized lines.

Handles the reference VENOMOUS shape: ``itunes:timing="None"`` with bare
``<p>`` lines whose words are separated by zero-width spaces (U+200B). Also
tolerates already-synced inputs (we ignore any existing timing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

ZERO_WIDTH = "​"
# Lines that carry no alignable lexical content.
_NONLEXICAL_RE = re.compile(r"^[\s♪♫🎵🎶()]*$")


def _localname(tag: object) -> str:
    if not isinstance(tag, str):  # comments / PIs
        return ""
    return tag.rsplit("}", 1)[-1]


@dataclass
class ParsedLine:
    text: str                                  # display text (normalized)
    words: list[str] = field(default_factory=list)
    is_nonlexical: bool = False


@dataclass
class ParsedLyrics:
    lang: str = "en"
    songwriters: list[str] = field(default_factory=list)
    lines: list[ParsedLine] = field(default_factory=list)

    @property
    def alignment_text(self) -> str:
        """Whitespace-joined lexical words across all lines, in order."""
        return " ".join(w for ln in self.lines if not ln.is_nonlexical
                         for w in ln.words)


def _normalize(text: str) -> str:
    text = text.replace(ZERO_WIDTH, "")
    # lxml already unescaped &apos; etc. Normalize NBSP and collapse runs.
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def from_lines(lines: list[str], *, lang: str = "en",
               songwriters: list[str] | None = None) -> ParsedLyrics:
    """Build ParsedLyrics from plain unsynced text lines.

    Used when the caller already has the lyrics as a list of strings (e.g. Lucid
    Lyrics' ``Static`` data) rather than a TTML document. Applies the same
    normalization / non-lexical detection / tokenization as :func:`parse`.
    """
    out = ParsedLyrics(lang=lang, songwriters=list(songwriters or []))
    for raw in lines:
        text = _normalize(raw)
        nonlex = bool(_NONLEXICAL_RE.match(text)) or text == ""
        words = [] if nonlex else text.split(" ")
        out.lines.append(ParsedLine(text=text, words=words,
                                    is_nonlexical=nonlex))
    return out


def parse(source: str | bytes) -> ParsedLyrics:
    """Parse TTML from a path, string, or bytes."""
    if isinstance(source, str) and ("<" not in source):
        data = open(source, "rb").read()
    elif isinstance(source, str):
        data = source.encode("utf-8")
    else:
        data = source

    # The reference files use a non-standard declaration order (`encoding`
    # before `version`). Left in place it sends lxml's recover parser into a
    # mode that silently drops `&apos;` entities, so strip it first.
    data = re.sub(rb"^\s*<\?xml[^>]*\?>", b"", data)
    parser = etree.XMLParser(recover=True, resolve_entities=True)
    root = etree.fromstring(data, parser=parser)
    if root is None:
        raise ValueError(
            "could not parse lyrics as TTML/XML (is this the right file?)")

    out = ParsedLyrics()

    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
    if lang:
        out.lang = lang

    for el in root.iter():
        ln = _localname(el.tag)
        if ln == "songwriter" and el.text:
            out.songwriters.append(el.text.strip())
        elif ln == "p":
            text = _normalize("".join(el.itertext()))
            nonlex = bool(_NONLEXICAL_RE.match(text)) or text == ""
            words = [] if nonlex else text.split(" ")
            out.lines.append(ParsedLine(text=text, words=words,
                                        is_nonlexical=nonlex))

    return out
