"""Serialize aligned lyrics to TTML (Clearview-internal format).

Matches the reference Clearview file's conventions: the
``music.apple.com/lyric-ttml-internal`` namespace, ``<body>`` before
``<head>``, the ``A:BB.CCC`` second-based time format, and per-line
``itunes:key`` / ``ttm:agent``.

Default is **line-level** (``itunes:timing="Line"``, line text directly inside
each ``<p>``) -- a byte-for-byte structural match for the Clearview reference.
Pass ``word_level=True`` for ``itunes:timing="Word"`` with one ``<span>`` per
word (karaoke).
"""

from __future__ import annotations

from .model import AlignedLyrics
from .timefmt import encode


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("'", "&apos;"))


def to_ttml(a: AlignedLyrics, *, word_level: bool = False) -> str:
    dur = encode(a.dur)
    div_begin = encode(a.first_begin)
    timing = "Word" if word_level else "Line"

    # Standards-compliant declaration (version before encoding). The reference
    # files emit the reversed, malformed order, which sends lxml's recover
    # parser into a mode that silently drops `&apos;`; we avoid that trap while
    # keeping the meaningful internal namespace + time format identical.
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append(
        f'<tt itunes:timing="{timing}" xml:lang="{_esc(a.lang)}" '
        'xmlns="http://www.w3.org/ns/ttml" '
        'xmlns:itunes="http://music.apple.com/lyric-ttml-internal" '
        'xmlns:ttm="http://www.w3.org/ns/ttml#metadata">'
    )
    out.append(f'<body dur="{dur}">')
    out.append(f'<div begin="{div_begin}" end="{dur}">')

    for ln in a.lines:
        attrs = (f'begin="{encode(ln.begin)}" end="{encode(ln.end)}" '
                 f'itunes:key="{ln.key}" ttm:agent="{_esc(ln.agent)}"')
        if ln.is_nonlexical:
            if ln.text:
                out.append(f'<p {attrs}>{_esc(ln.text)}</p>')
            else:
                out.append(f'<p {attrs}/>')
            continue
        if word_level:
            body = " ".join(
                f'<span begin="{encode(w.begin)}" end="{encode(w.end)}">'
                f'{_esc(w.text)}</span>'
                for w in ln.words
            )
        else:
            body = _esc(ln.text)
        out.append(f'<p {attrs}>{body}</p>')

    out.append('</div></body>')

    agents = "".join(f'<ttm:agent type="person" xml:id="{_esc(ag)}"/>'
                     for ag in a.agents)
    out.append(f'<head><metadata>{agents}</metadata></head>')
    out.append('</tt>')
    return "".join(out)
