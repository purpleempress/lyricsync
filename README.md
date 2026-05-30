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

## GPU offload with Modal

Demucs + whisper are brutal on CPU (tens of seconds each). The decode, Demucs
isolation, and forced alignment can run on a cheap GPU via
[Modal](https://modal.com/docs/guide) instead, while the audio sourcing and the
cheap parse/map/rebuild stay local. It's the same compute code either way
(`_compute_alignment` in `lyricsync/align.py`), just with `device="cuda"`.

One-time setup:

```bash
.venv/bin/pip install -e ".[api,modal]"
modal token new                  # prints token_id/secret -> put them in .env as
                                 # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
bash scripts/deploy_modal.sh     # sources .env, builds the image (pre-bakes
                                 # htdemucs + whisper "small"), deploys
```

Then flip the API into offload mode. The backend is chosen by
`LYRICSYNC_BACKEND` (`local` = CPU, the default; `modal`; or `runpod`):

```bash
LYRICSYNC_BACKEND=modal bash scripts/run_api.sh                       # local venv
docker compose -f compose.yaml -f compose.modal.yaml up -d --build   # Docker
```

The base `compose.yaml` is a pure-CPU service; Modal and RunPod each layer in via
an opt-in overlay (`compose.modal.yaml` / `compose.runpod.yaml`) that turns the
backend on *and* supplies its credentials together — both purely through env from
`.env` (`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, `RUNPOD_*`), so neither overlay
mounts anything. The Docker image installs the `modal` client (no CUDA).

`LYRICSYNC_MODAL_APP` overrides the deployed app name (default `lyricsync`). The
GPU is `T4`, and a warm container lingers `LYRICSYNC_GPU_IDLE_SECS` (default 300)
after a job before scaling to zero — read at deploy time, so re-run
`scripts/deploy_modal.sh` to change it. (RunPod ignores that var; its equivalent
is the endpoint's Idle Timeout in the console.) Model sizes other than `small`
download on first use. Re-run `scripts/deploy_modal.sh` after editing
`modal_app.py` or the `lyricsync` package.

Cold starts are cut with **memory snapshots**: the work is a `modal.Cls` whose
`@modal.enter(snap=True)` loads htdemucs + whisper onto the GPU once, and
`enable_memory_snapshot` + the alpha `enable_gpu_snapshot` freeze that state
(including VRAM). A cold-started container then restores the snapshot instead of
re-reading weights from disk and re-initializing torch/CTranslate2. `T4` GPU
snapshotting is alpha — if a deploy/run errors on the snapshot, drop
`experimental_options={"enable_gpu_snapshot": True}` for a CPU-only snapshot
(still helps; whisper weights then reload onto the GPU per cold start).

### RunPod (alternative backend)

`runpod_handler.py` + `Dockerfile.runpod` run the same `_compute_alignment` on a
RunPod serverless endpoint instead of Modal. Build and register the worker, then
point LyricSync at the endpoint:

```bash
IMAGE=docker.io/youruser/lyricsync-runpod:latest bash scripts/build_runpod.sh
# create a Serverless endpoint from that image in the RunPod console, then:
export LYRICSYNC_BACKEND=runpod RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
bash scripts/run_api.sh
# or under Docker:
#   RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=... \
#     docker compose -f compose.yaml -f compose.runpod.yaml up -d --build
```

The cold-start story is RunPod's **FlashBoot**: it's on by default for the
endpoint and keeps a warmed worker alive between requests, so a reused worker
already has both models resident (loaded once at module import in
`runpod_handler.py`) — the same idea as Modal's snapshots, just achieved by
keeping the process warm rather than freezing it. The client talks to RunPod's
`/run` + `/status` REST API (stdlib `urllib`, so the API venv gains no deps);
audio crosses as base64 in the job JSON.

## Benchmarks

One track (137 s, 344 words), single run — ballpark, not gospel. CPU is
an 8-core Intel Core Ultra 5 228V (no AVX-512/AMX); GPU is a `T4`. Reproduce with
`scripts/bench.py <audio> <lyrics.ttml> --backends local,modal,runpod`.

**Backends** (warm = reused container, Demucs on):

| backend     | cold   | warm  |
|-------------|--------|-------|
| local CPU   | 125 s¹ | 77 s  |
| Modal (T4)  | 22 s   | 12 s  |
| RunPod (T4) | 27 s   | 14 s  |

¹ cold CPU includes ~12 s model load + first-run overhead.

**Local CPU knob sweep** (warm; drift = per-word timing delta vs the Demucs baseline):

| variant              | total | demucs | align | drift vs baseline       |
|----------------------|-------|--------|-------|-------------------------|
| baseline (Demucs)    | 77 s  | 66 s   | 10 s  | (ref)                   |
| Demucs `overlap=0.1` | 68 s  | 57 s   | 10 s  | n/a²                    |
| Demucs `shifts=0`    | 78 s  | 66 s   | 12 s  | med 0.02 s / max 4.4 s  |
| `fast_mode`          | 73 s  | 64 s   | 9 s   | n/a²                    |
| **`--no-demucs`**    | **10 s** | —   | 9 s   | med 0.04 s / max 0.94 s |

² word count differed (345 vs 344), so per-word drift isn't directly comparable.

Takeaways:

- **Demucs is ~85% of the CPU cost** (66 of 77 s); the align stage is ~9–10 s
  everywhere, GPU included. The GPU's entire value is accelerating Demucs.
- **`--no-demucs` is ~8× faster on CPU** (~10 s) and, for a clean-vocal track like
  this, barely moves the timings (max 0.94 s) — so local CPU without Demucs is
  competitive with the GPU backends, and free. Noisy/dense mixes still want it.
- **`overlap=0.1`** is the only Demucs knob worth touching (~15% off). **`shifts=0`**
  saves nothing and threw a 4.4 s outlier. **`fast_mode`** shaves ~1 s on align but
  changes word tokenization, so treat it as a behaviour change, not a free win.
- The idle keep-alive (`LYRICSYNC_GPU_IDLE_SECS` / RunPod Idle Timeout) usually
  dominates GPU cost for sporadic use — you pay for the warm container, not just
  the ~12–14 s of work.

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

The reference TTML stores times in seconds written as `A:BB.CCC`, which
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
