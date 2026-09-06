# Flagship pipeline

[![CI](https://github.com/lewisluc87-hub/flagship-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/lewisluc87-hub/flagship-pipeline/actions/workflows/ci.yml)

Takes one YouTube URL and produces a transcript, an AI scene breakdown, and
a reactive waveform video, all in one consolidated output folder -- by
orchestrating three separately-built tools:

1. **[youtube-transcriber](https://github.com/lewisluc87-hub/youtube-transcriber)** -> `transcript.txt` + `summary.md`
2. **[video-to-prompt](https://github.com/lewisluc87-hub/video-to-prompt)** -> `prompts.md` (scene-by-scene breakdown)
3. **[waveform-generator](https://github.com/lewisluc87-hub/waveform-generator)** -> `waveform.webm` (headless render)

## Why this exists

`youtube-transcriber` only ever downloads audio, and only as a Whisper
fallback for videos with no captions -- it never keeps a video file.
`video-to-prompt` needs an actual local video file for its CV analysis.
So this script owns the video download step itself (via `yt-dlp`), and
hands that same file to both `video-to-prompt` and the waveform renderer --
ffmpeg demuxes audio straight out of the mp4, so no separate extraction
step is needed.

## Requirements

- Python 3, `pip install -r requirements.txt`
- `video-to-prompt` installed as an editable package so `video2prompt` is on PATH
- `youtube-transcriber`'s dependencies installed in the same environment
- `waveform-generator`'s `render-waveform.js` with its `npm install`'d dependencies
- `ffmpeg`/`ffprobe` on PATH
- Node.js on PATH

## Usage

```
python run_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

First run recommendation: use `--no-llm` to sanity-check the integration
without spending on API calls, before running the full LLM-backed
breakdown.

By default, assumes `youtube-transcriber` and `waveform-generator` are
sibling directories:
```
some-parent-folder/
├── flagship-pipeline/
│   └── run_pipeline.py
├── youtube-transcriber/
├── video-to-prompt/
└── waveform-generator/
```
Use `--transcriber-path` / `--waveform-script` if your layout differs.

### Flags

| Flag | Effect |
|---|---|
| `--output-dir DIR` | Base output directory (default: `pipeline-output`) |
| `--transcriber-path PATH` | Path to `transcribe.py` |
| `--waveform-script PATH` | Path to `render-waveform.js` |
| `--mode breakdown\|video_prompt` | video-to-prompt mode (default: `breakdown`) |
| `--no-llm` | Skip API calls (template-only breakdown) |
| `--with-audio` | Mux source audio into the waveform output |
| `--keep-video` | Keep the downloaded source video (deleted by default) |

## Output structure

```
pipeline-output/<video-id>-<title-slug>/
├── index.md
├── transcript.txt
├── summary.md
├── prompts.md
└── waveform.webm
```

`index.md` reports which of the three steps succeeded or failed -- a
failing step doesn't abort the run; the other steps still complete.

## Running the tests

```
pip install -r requirements.txt
pytest -q
```

13 tests: unit tests for the pure folder-naming/index-writing logic, plus
integration tests that run the real, unmodified `run_pipeline.py` as a
subprocess against stub tools matching the real CLIs' exact contracts (and
a faked `yt_dlp` standing in for the network-dependent download), so CI
can verify the whole orchestration flow without needing real network
access or API keys.

## Verified

- Full stub-based integration testing (see above) -- CI-enforced on every push.
- **Real end-to-end run** against an actual YouTube Short, with the real
  tools, confirming: correct video download, correct transcript, a
  `--no-llm` scene breakdown (expected to be noisier/generic without an
  LLM, consistent with `video-to-prompt`'s own known behavior), and a
  waveform video independently verified via `ffprobe` (matching duration,
  resolution, framerate) and by comparing extracted frames at different
  timestamps against the real transcript content -- confirming genuinely
  audio-reactive rendering, not a static placeholder.
- One real bug found and fixed during that end-to-end run: `Path.rename()`
  throws `FileExistsError` on Windows when the destination already exists
  (unlike POSIX, where it overwrites). Fixed with `os.replace()`, which
  overwrites on both platforms. Covered by
  `test_running_twice_in_a_row_does_not_crash_on_overwrite`.

## Known limitations

- Not yet tested with a long-form (non-Shorts) video.
- `summary.md` generation depends on a valid Anthropic API key being
  configured for `youtube-transcriber`; without one, the transcript still
  saves but the summary step reports a clear failure rather than crashing.
