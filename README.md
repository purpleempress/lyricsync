# LyricSync

LyricSync times unsynced lyrics to a song's audio. You give it the lyrics you
already have (plain lines or TTML) plus a Spotify track id, and it hands back the
same lyrics with per-line or per-word timestamps.

It exists to fill a gap in [Lucid Lyrics](https://gitlab.com/sanoojes/lucid-lyrics):
when a track only has plain, untimed lyrics, a forked Lucid calls this service and
gets synced lyrics back a minute or so later.

## How it works

1. librespot pulls the exact Spotify master for the track id, so the timing can't
   drift from a wrong YouTube rip.
2. ffmpeg decodes it to 16 kHz mono. Demucs optionally isolates the vocal stem.
3. [stable-ts](https://github.com/jianfch/stable-ts) aligns the lyrics you gave it
   against the audio. There's no transcription guesswork; it places words it
   already knows.
4. The word timings get mapped back onto the original lines. `♪` and blank lines
   have nothing to align, so they're interpolated from their neighbours.
5. The result is written as Lucid `Line`/`Syllable` JSON (times in seconds), Apple
   TTML, or both.

## Running the API

With Docker:

```bash
python scripts/librespot_login.py   # one-time Spotify login -> credentials.json
docker compose up -d --build
```

Or locally:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[api,dev]"      # or: uv pip install
.venv/bin/python scripts/librespot_login.py
bash scripts/run_api.sh                     # serves on 0.0.0.0:8000
```

You need ffmpeg on the host (the image already has it). librespot wants your own
Spotify Premium login; those credentials live in `credentials.json`, which is
gitignored and never copied into the image.

### Endpoints

- `POST /sync` with `{spotifyId, title, artist, duration, lyrics:[...], wordLevel,
  demucs, model}` returns `{job_id}`. The service sources the audio itself, aligns,
  and caches the result.
- `POST /align` for local file testing: upload an audio file plus
  `lyrics_ttml`/`lyrics_text`.
- `GET /jobs/{id}` polls a job through `queued`, `running`, then `done` (with
  `result`) or `error`.

Alignment takes tens of seconds (more with Demucs), so jobs run in the background
and you poll. Finished results are cached to disk per `(track, wordLevel, model,
demucs)`, and identical in-flight requests share a single job, so re-plays and
accidental double-fires stay cheap. Only one job runs at a time so the alignment
doesn't fight itself for CPU.

## Lucid Lyrics integration

`lucid-lyrics/` is a fork of Lucid Lyrics on the `lyricsync-integration` branch.
When a track resolves to only `Static` (unsynced) lyrics, the fork posts them to
`/sync`, keeps the plain lyrics on screen with a small "Syncing lyrics…" pill
pinned at the top, and swaps in the synced version once the job lands. That result
goes into IndexedDB, so the next play is instant.

A few behaviours worth knowing:

- It waits about 2 seconds after a track settles before syncing, and runs one sync
  at a time. Flicking through songs won't spawn a job per track; it syncs whatever
  you actually land on.
- Turn on "Upgrade line-synced lyrics" and it will also re-align a provider's
  line-level lyrics into word-by-word.
- The API URL, auto-sync, word-by-word, and the upgrade toggle all live in
  Settings under "Lyric Sync".

Build and install it with `bash scripts/build_extension.sh`, which compiles the
extension and applies it to Spotify (needs bun, node, and spicetify on PATH).

## CLI

For one-off files, skip the API:

```bash
lyricsync song.opus lyrics.ttml                            # writes .synced.ttml + .json
lyricsync song.opus lyrics.ttml --no-demucs --model base   # faster, looser timing
lyricsync song.opus lyrics.ttml --word-level               # per-word
```

The two arguments are order-agnostic; whichever one is `.ttml`/`.xml` is treated as
the lyrics.

## The time-format gotcha

The reference Clearview TTML stores times in seconds written as `A:BB.CCC`, which
decodes to `A + BB/100 + CCC/100000`. So `12:08.400` is 12.084 seconds, not 12
minutes. It looks like `MM:SS` and isn't. The Lucid JSON uses plain seconds, which
Lucid compares against `progress/1000`. Get this wrong and every line lands
thousands of seconds in the future, so nothing ever highlights. It cost me an
afternoon. See `lyricsync/timefmt.py`.

## Roadmap

- Detect background vocals (`ttm:role="x-bg"`) and duets (a second agent /
  `OppositeAligned`).
- Pass romanization through (Lucid already renders `RomanizedText`).
- Send the position-interpolation and stale-stylesheet fixes back to upstream
  Lucid. They're bugs in the original, not just here.
