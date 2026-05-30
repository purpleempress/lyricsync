"""Benchmark the LyricSync compute stage across backends and tuning knobs.

    PYTHONPATH=. .venv/bin/python scripts/bench.py VENOMOUS.opus VENOMOUS.ttml
    # add remote backends (need creds + a deployed app/endpoint):
    PYTHONPATH=. .venv/bin/python scripts/bench.py VENOMOUS.opus VENOMOUS.ttml \
        --backends local,modal,runpod

LOCAL: sweeps a grid of knobs (fast_mode, Demucs overlap/shifts, demucs on/off)
on CPU. It first runs the baseline COLD (caches cleared, so you see model-load
cost), then everything WARM, and reports per-stage seconds plus the word-timing
drift each variant introduces versus the warm baseline (the quality signal).

MODAL / RUNPOD: measures cold (first call) then warm (immediate second call)
end-to-end latency at default knobs -- those backends read the knobs from the
worker's env, so they can't be varied per call from here.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import time
from pathlib import Path

from lyricsync import ttml_in
from lyricsync.align import (_align_on_modal, _align_on_runpod,
                             _compute_alignment, _load_demucs, _load_model,
                             probe_duration)

_TIMING_RE = re.compile(r"\[timing\]\s+(\S+)\s+([\d.]+)s")

# (label, kwargs for _compute_alignment). First entry is the baseline/reference.
_GRID = [
    ("baseline (demucs full)", dict(demucs=True, fast_mode=False,
                                    demucs_shifts=1, demucs_overlap=0.25)),
    ("fast_mode",              dict(demucs=True, fast_mode=True,
                                    demucs_shifts=1, demucs_overlap=0.25)),
    ("demucs overlap=0.1",     dict(demucs=True, fast_mode=False,
                                    demucs_shifts=1, demucs_overlap=0.1)),
    ("demucs shifts=0",        dict(demucs=True, fast_mode=False,
                                    demucs_shifts=0, demucs_overlap=0.25)),
    ("all-fast (sh0 ov.1 fm)", dict(demucs=True, fast_mode=True,
                                    demucs_shifts=0, demucs_overlap=0.1)),
    ("no demucs",              dict(demucs=False, fast_mode=False)),
    ("no demucs + fast_mode",  dict(demucs=False, fast_mode=True)),
]


def _run_local(audio, text, lang, opts, *, cold=False):
    if cold:
        _load_model.cache_clear()
        _load_demucs.cache_clear()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(buf):
        _, aligned = _compute_alignment(
            audio, text, language=lang, device="cpu", **opts)
    total = time.perf_counter() - t0
    stages = {n: float(s) for n, s in _TIMING_RE.findall(buf.getvalue())}
    return total, stages, aligned


def _drift(base, other):
    """(median, max) abs timing delta per word vs baseline, or None if counts differ."""
    n = min(len(base), len(other))
    if not n or len(base) != len(other):
        return None
    ds = sorted(max(abs(base[i][1] - other[i][1]), abs(base[i][2] - other[i][2]))
                for i in range(n))
    return ds[n // 2], ds[-1]


def _fmt_drift(d):
    return "  (ref)" if d == "ref" else "  n/a (word count differs)" if d is None \
        else f"  med {d[0]:.2f}s / max {d[1]:.2f}s"


def _bench_local(audio, text, lang):
    print("\n== LOCAL (CPU) ==")
    base_opts = _GRID[0][1]
    cold_total, cold_st, _ = _run_local(audio, text, lang, base_opts, cold=True)
    print(f"baseline COLD: {cold_total:6.1f}s total  "
          f"(demucs_load {cold_st.get('demucs_load', 0):.1f}s + "
          f"model_load {cold_st.get('model_load', 0):.1f}s on first job)")

    print(f"\n{'variant':<24} {'total':>7} {'demucs':>7} {'align':>7} {'words':>6}  drift")
    base_aligned = None
    for label, opts in _GRID:
        total, st, aligned = _run_local(audio, text, lang, opts)
        if base_aligned is None:
            base_aligned = aligned
            d = "ref"
        else:
            d = _drift(base_aligned, aligned)
        print(f"{label:<24} {total:6.1f}s {st.get('demucs', 0):6.1f}s "
              f"{st.get('align', 0):6.1f}s {len(aligned):6d}{_fmt_drift(d)}")


def _bench_remote(name, fn, audio, text, lang):
    print(f"\n== {name.upper()} (cold then warm, default knobs) ==")
    try:
        for label in ("cold", "warm"):
            t0 = time.perf_counter()
            _, aligned = fn(audio, text, demucs=True, model_name="small", language=lang)
            print(f"  {label:<5} {time.perf_counter() - t0:6.1f}s  ({len(aligned)} words)")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("lyrics", help="unsynced lyrics TTML")
    ap.add_argument("--backends", default="local",
                    help="comma list of: local, modal, runpod (default local)")
    args = ap.parse_args(argv)

    parsed = ttml_in.parse(args.lyrics)
    text = parsed.alignment_text
    lang = parsed.lang or "en"
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"track ~{probe_duration(args.audio):.0f}s | {len(text.split())} words "
          f"| backends={backends}")

    if "local" in backends:
        _bench_local(args.audio, text, lang)
    if "modal" in backends:
        _bench_remote("modal", _align_on_modal, args.audio, text, lang)
    if "runpod" in backends:
        _bench_remote("runpod", _align_on_runpod, args.audio, text, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
