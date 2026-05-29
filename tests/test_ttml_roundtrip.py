from pathlib import Path

import pytest
from lxml import etree

from lyricsync import ttml_in
from lyricsync.model import AlignedLyrics, Line, Word
from lyricsync.spicy_json import to_spicy
from lyricsync.timefmt import decode
from lyricsync.ttml_out import to_ttml

ROOT = Path(__file__).resolve().parent.parent
VENOMOUS = ROOT / "VENOMOUS by passengerprincess.ttml"
CLEARVIEW = ROOT / "Clearview by Sophie Powers, NOAHFINNCE.ttml"


@pytest.mark.skipif(not VENOMOUS.is_file(), reason="reference VENOMOUS.ttml not present")
def test_parse_venomous_input():
    p = ttml_in.parse(str(VENOMOUS))
    assert len(p.lines) > 30
    # Zero-width spaces must be stripped, apostrophes unescaped.
    joined = " ".join(l.text for l in p.lines)
    assert "​" not in joined
    assert "&apos;" not in joined
    assert "'" in joined  # e.g. "She's"
    assert "Chris Robinson" in p.songwriters
    assert p.alignment_text.startswith("She knew that you would")


@pytest.mark.skipif(not CLEARVIEW.is_file(), reason="reference Clearview.ttml not present")
def test_parse_clearview_flags_nonlexical():
    p = ttml_in.parse(str(CLEARVIEW))
    texts = [l.text for l in p.lines]
    assert "♪" in texts
    # the trailing empty <p/> becomes a nonlexical line
    assert any(l.is_nonlexical and l.text == "" for l in p.lines)
    note = next(l for l in p.lines if l.text == "♪")
    assert note.is_nonlexical and note.words == []


def _sample() -> AlignedLyrics:
    a = AlignedLyrics(dur=10.0, lang="en", songwriters=["A"])
    a.lines.append(Line(key="L1", begin=1.0, end=2.5, text="hello world",
                        words=[Word("hello", 1.0, 1.6), Word("world", 1.7, 2.5)]))
    a.lines.append(Line(key="L2", begin=2.5, end=5.0, text="♪",
                        is_nonlexical=True))
    a.lines.append(Line(key="L3", begin=5.0, end=6.0, text="bye",
                        words=[Word("bye", 5.0, 6.0)]))
    return a


NS = {"t": "http://www.w3.org/ns/ttml"}
TIMING = "{http://music.apple.com/lyric-ttml-internal}timing"


def test_ttml_line_level_is_default():
    root = etree.fromstring(to_ttml(_sample()).encode("utf-8"),
                            etree.XMLParser(recover=True))
    assert root.get(TIMING) == "Line"
    assert root.findall(".//t:span", NS) == []      # no word spans
    ps = root.findall(".//t:p", NS)
    assert ps[0].text == "hello world"              # full line text inline
    assert ps[1].text == "♪"


def test_ttml_word_level_spans_monotonic():
    root = etree.fromstring(to_ttml(_sample(), word_level=True).encode("utf-8"),
                            etree.XMLParser(recover=True))
    assert root.get(TIMING) == "Word"
    times = []
    for sp in root.findall(".//t:span", NS):
        b, e = decode(sp.get("begin")), decode(sp.get("end"))
        assert b <= e
        times.append(b)
    assert times == sorted(times)
    # nonlexical line still has its glyph and no spans
    ps = root.findall(".//t:p", NS)
    assert ps[1].text == "♪" and ps[1].findall("t:span", NS) == []


def test_ttml_preserves_apostrophes_both_modes():
    a = AlignedLyrics(dur=3.0, lang="en")
    a.lines.append(Line(key="L1", begin=0.0, end=1.0, text="She's gone",
                        words=[Word("She's", 0.0, 0.5), Word("gone", 0.5, 1.0)]))
    for wl in (False, True):
        xml = to_ttml(a, word_level=wl)
        assert "She&apos;s" in xml
        # standards-compliant declaration -> apostrophe survives naive parsing
        root = etree.fromstring(xml.encode("utf-8"),
                                etree.XMLParser(recover=True))
        assert "She's" in "".join(root.itertext())


def test_spicy_line_level_is_default():
    doc = to_spicy(_sample())
    assert doc["Type"] == "Line"
    assert doc["SongWriters"] == ["A"]
    # times in SECONDS (spicy-lyrics compares against progress/1000)
    item = doc["Content"][0]
    assert item == {"Text": "hello world", "StartTime": 1.0, "EndTime": 2.5}
    assert doc["Content"][1]["Text"] == "♪"


def test_spicy_word_level_seconds_and_shape():
    doc = to_spicy(_sample(), word_level=True)
    assert doc["Type"] == "Syllable"
    lead = doc["Content"][0]["Lead"]
    assert lead["StartTime"] == 1.0 and lead["EndTime"] == 2.5
    # times in seconds; word-final syllable carries a trailing space
    assert lead["Syllables"][0] == {"Text": "hello ", "StartTime": 1.0,
                                    "EndTime": 1.6, "IsPartOfWord": False}
    assert lead["Syllables"][1]["Text"] == "world "
    assert doc["Content"][1]["Lead"]["Syllables"] == []
