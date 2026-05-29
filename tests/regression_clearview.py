"""Accuracy regression: re-sync Clearview from scratch and compare line begins
against the real reference timings. Not a pytest (needs models + audio)."""
import re
import statistics
from pathlib import Path

from lyricsync import ttml_in
from lyricsync.align import align

ROOT = Path(__file__).resolve().parent.parent
TTML = ROOT / "Clearview by Sophie Powers, NOAHFINNCE.ttml"
OPUS = ROOT / "Clearview [NfAO5byGFEA].opus"


def ref_begins():
    data = TTML.read_text(encoding="utf-8")
    # reference uses A:BB.CCC -> seconds
    out = []
    for m in re.finditer(r'<p begin="(\d+):(\d{2})\.(\d{3})"', data):
        a, bb, cc = m.groups()
        out.append(int(a) + int(bb) / 100 + int(cc) / 100000)
    return out


def main():
    parsed = ttml_in.parse(str(TTML))   # timing ignored, text reused as "unsynced"
    aligned = align(str(OPUS), parsed, demucs=True, model_name="base")
    pred = [ln.begin for ln in aligned.lines]
    ref = ref_begins()
    n = min(len(pred), len(ref))
    errs = [abs(pred[i] - ref[i]) for i in range(n)]
    print(f"lines compared: {n}")
    print(f"line-begin abs error  mean={statistics.mean(errs):.2f}s  "
          f"median={statistics.median(errs):.2f}s  max={max(errs):.2f}s")
    print(f"within 0.5s: {sum(e<=0.5 for e in errs)}/{n}   "
          f"within 1.0s: {sum(e<=1.0 for e in errs)}/{n}")
    for i in (0, 1, 2, 17, n - 1):
        print(f"  L{i+1}: pred={pred[i]:.2f}s  ref={ref[i]:.2f}s  "
              f"err={abs(pred[i]-ref[i]):.2f}s  text={aligned.lines[i].text[:35]!r}")


if __name__ == "__main__":
    main()
