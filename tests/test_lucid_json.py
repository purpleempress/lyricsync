from lyricsync.lucid_json import to_lucid
from lyricsync.model import AlignedLyrics, Line, Word


def _sample() -> AlignedLyrics:
    a = AlignedLyrics(dur=10.0, lang="en", songwriters=["A"])
    a.lines.append(Line(key="L1", begin=1.0, end=2.5, text="hello world",
                        words=[Word("hello", 1.0, 1.6), Word("world", 1.7, 2.5)]))
    a.lines.append(Line(key="L2", begin=2.5, end=5.0, text="♪",
                        is_nonlexical=True))
    a.lines.append(Line(key="L3", begin=5.0, end=6.0, text="bye",
                        words=[Word("bye", 5.0, 6.0)]))
    return a


def test_line_level_default_lucid_shape():
    d = to_lucid(_sample(), id="track123")
    assert d["Type"] == "Line"
    assert d["Id"] == "track123"
    assert d["Provider"] == "lyricsync"
    assert d["SongWriters"] == ["A"]
    # seconds, top-level span spans first..last content
    assert d["StartTime"] == 1.0 and d["EndTime"] == 6.0
    c0 = d["Content"][0]
    assert c0 == {"Type": "Line", "Text": "hello world", "StartTime": 1.0,
                  "EndTime": 2.5, "OppositeAligned": False}
    # ♪ -> Interlude with Text " "
    assert d["Content"][1] == {"Type": "Interlude", "Text": " ",
                               "StartTime": 2.5, "EndTime": 5.0,
                               "OppositeAligned": False}


def test_word_level_syllable_shape():
    d = to_lucid(_sample(), id="t", word_level=True)
    assert d["Type"] == "Syllable"
    assert d["StartTime"] == 1.0 and d["EndTime"] == 6.0
    c0 = d["Content"][0]
    assert c0["Type"] == "Vocal" and c0["OppositeAligned"] is False
    lead = c0["Lead"]
    assert lead["StartTime"] == 1.0 and lead["EndTime"] == 2.5
    # NO trailing space in Text -- Lucid adds it via CSS (trailing-whitespace)
    assert lead["Syllables"][0] == {"Text": "hello", "IsPartOfWord": False,
                                    "StartTime": 1.0, "EndTime": 1.6}
    # ♪ interlude -> single syllable carrying the glyph
    assert d["Content"][1]["Lead"]["Syllables"][0]["Text"] == "♪"


def test_provider_override():
    assert to_lucid(_sample(), provider="user")["Provider"] == "user"
