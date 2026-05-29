import pytest

from lyricsync.sources import AudioSource, SourceError, TrackRef, source_audio


class FakeSource(AudioSource):
    def __init__(self, name, avail, result=None, boom=False):
        self.name = name
        self._avail = avail
        self._result = result
        self._boom = boom
        self.fetched = False

    def available(self):
        return self._avail

    def fetch(self, track, dst_dir):
        self.fetched = True
        if self._boom:
            raise RuntimeError("kaboom")
        return self._result


def test_trackref_uri():
    assert TrackRef("abc").uri == "spotify:track:abc"
    assert TrackRef("spotify:track:abc").uri == "spotify:track:abc"


def test_prefers_first_available():
    a = FakeSource("primary", True, "/tmp/a.ogg")
    b = FakeSource("fallback", True, "/tmp/b.ogg")
    path, name = source_audio(TrackRef("x"), "/tmp", prefer=(a, b))
    assert (path, name) == ("/tmp/a.ogg", "primary")
    assert b.fetched is False


def test_skips_unavailable_and_falls_back():
    a = FakeSource("primary", False)               # not configured
    b = FakeSource("fallback", True, "/tmp/b.ogg")
    path, name = source_audio(TrackRef("x"), "/tmp", prefer=(a, b))
    assert (path, name) == ("/tmp/b.ogg", "fallback")


def test_falls_back_on_error():
    a = FakeSource("primary", True, boom=True)     # available but throws
    b = FakeSource("fallback", True, "/tmp/b.ogg")
    path, name = source_audio(TrackRef("x"), "/tmp", prefer=(a, b))
    assert name == "fallback" and a.fetched is True


def test_all_fail_raises():
    a = FakeSource("primary", False)
    b = FakeSource("fallback", True, boom=True)
    with pytest.raises(SourceError) as e:
        source_audio(TrackRef("x"), "/tmp", prefer=(a, b))
    assert "primary" in str(e.value) and "fallback" in str(e.value)
