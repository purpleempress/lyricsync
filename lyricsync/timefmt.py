"""Encode/decode the Clearview-internal TTML time format.

The reference Clearview TTML uses Apple's ``lyric-ttml-internal`` namespace,
whose timestamps are *seconds* rendered as ``A:BB.CCC`` where the value equals
``A + BB/100 + CCC/100000`` -- i.e. drop the colon and concatenate the
fractional digits:

    "12:08.400"  -> 12.084 s
    "164:14.400" -> 164.144 s

This is NOT the standard TTML ``MM:SS.fff`` clock; do not confuse the two.
"""

from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(\d+):(\d{2})\.(\d{3})$")


def encode(t: float) -> str:
    """Seconds -> ``A:BB.CCC`` (5 fractional digits, split 2/3)."""
    if t < 0:
        t = 0.0
    whole = int(t)
    digits = round((t - whole) * 100000)  # 0..99999
    # Carry if rounding pushed us to a full second (e.g. 11.999996 -> 12.00000).
    if digits >= 100000:
        whole += 1
        digits -= 100000
    return f"{whole}:{digits // 1000:02d}.{digits % 1000:03d}"


def decode(s: str) -> float:
    """``A:BB.CCC`` -> seconds. Inverse of :func:`encode`."""
    m = _TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"not a Clearview-internal time: {s!r}")
    a, bb, ccc = m.groups()
    return int(a) + int(bb) / 100 + int(ccc) / 100000
