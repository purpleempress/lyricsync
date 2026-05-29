import math

import pytest

from lyricsync.timefmt import decode, encode

# Exact values lifted from the reference Clearview TTML.
CASES = [
    ("12:08.400", 12.084),
    ("16:14.400", 16.144),
    ("51:00.600", 51.006),
    ("164:14.400", 164.144),
    ("0:00.000", 0.0),
]


@pytest.mark.parametrize("s,sec", CASES)
def test_decode(s, sec):
    assert math.isclose(decode(s), sec, abs_tol=1e-9)


@pytest.mark.parametrize("s,sec", CASES)
def test_encode(s, sec):
    assert encode(sec) == s


@pytest.mark.parametrize("s,_sec", CASES)
def test_roundtrip(s, _sec):
    assert encode(decode(s)) == s


def test_rounding_carry():
    # Fractional rounding that bumps to a whole second must carry, not overflow.
    assert encode(11.999999) == "12:00.000"


def test_bad_input():
    with pytest.raises(ValueError):
        decode("not-a-time")
