# voiceofgod

Self-hosted live captions for a sermon. The operator sends the pulpit microphone from a browser. Listeners scan a QR code, pick English or Bahasa Indonesia, and follow along. There is no listener account and no app to install.

Phase 1 is captions. Phase 2 adds reconnect, up to four live sessions, a glossary editor, transcript downloads, an overlay page, and operator metrics. Translated audio is later.

## What you need

- A Linux box with an NVIDIA GPU, driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Compose v2
- A domain and HTTPS in front of this service. Browsers will not open the microphone on plain HTTP, except on localhost
- 24 GB of GPU memory for the default models. See [HARDWARE.md](HARDWARE.md)

Model downloads are large (about 2 GB for Whisper large-v3-turbo and about 15 GB for Qwen2.5-7B-Instruct). The steps below are the setup, not the download.

## Run on the GPU server

```bash
git clone <this-repo> voiceofgod
cd voiceofgod
cp .env.example .env
```

Edit `.env` and set `PUBLIC_BASE_URL` to the HTTPS origin listeners will open, for example `https://captions.example.org`. Set `DATABASE_URL`, and set `ADMIN_USERNAME` and `ADMIN_PASSWORD` for the first start. Sign in at `/login`. The admin creates every other account. Listeners never sign in.

If the machine's CUDA driver is 12.x rather than 13, set `VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129` in `.env`.

```bash
docker compose --profile gpu up --build
```

Point the HTTPS proxy at port 8080. Open `https://captions.example.org`, create a session, choose the speaker language, add church names that must not be translated, and press Start. The QR code on the operator page is the listener link. Keep that operator URL private. Anyone with it can end the session.

Check readiness:

```bash
curl -s http://127.0.0.1:8080/health
```

`whisper_loaded` stays false until the first session is created. That first create downloads and loads Whisper, so do it before the service, not as the congregation is sitting down.

## Laptop, without a GPU

The caption pipeline and both pages can be exercised without models:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
$env:DATABASE_URL = "postgresql://voiceofgod:password@127.0.0.1:5433/voiceofgod"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "replace-with-a-long-password"
$env:ALLOW_FAKE_MODELS = "true"
$env:WHISPER_DEVICE = "fake"
$env:REQUIRE_TRANSLATOR = "false"
$env:REQUIRE_REDIS = "false"
$env:VLLM_BASE_URL = ""
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

This mode does not translate. The operator page says so. Do not use it for a service.

A real CPU smoke test, still not the production setup:

```bash
$env:WHISPER_MODEL = "tiny"
$env:WHISPER_DEVICE = "cpu"
$env:WHISPER_COMPUTE_TYPE = "int8"
$env:ALLOW_FAKE_MODELS = "false"
$env:REQUIRE_TRANSLATOR = "false"
$env:VLLM_BASE_URL = ""
```

## Google speech, Gemini, and voice

This path does not need the GPU. Set `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, and a paid `GEMINI_API_KEY` in `.env`. Leave `VLLM_BASE_URL` empty if Gemini should translate. Speech-to-Text v1 hears the microphone, Gemini 3.5 Flash translates with thinking off, and WaveNet speaks the final translation. Listeners tap Listen and should use headphones so the pulpit microphone cannot hear the voice. Captions still work if a clip fails. Pastor name, church name, and verse references are not sent yet. Do not put a Bible text in the prompt.

## OBS and ProPresenter

The operator page can take the pulpit from OBS instead of the browser microphone. Install ffmpeg and [MediaMTX](https://github.com/bluenviron/mediamtx) v1.13.1, and set `RTMP_PUBLIC_URL`, `MEDIAMTX_RTMP_URL`, `RTMP_AUTH_SECRET`, and `RTMP_READ_PASSWORD` in `.env`. Use letters and numbers for both secrets. Point MediaMTX at this repo's `mediamtx.yml` with `RTMP_AUTH_URL` set to `http://127.0.0.1:8080/internal/rtmp-auth/` plus the same auth secret. Docker Compose starts MediaMTX and overrides the pull address to `rtmp://mediamtx:1935`.

On the operator page choose OBS RTMP. OBS uses a custom stream: the server and stream key are on that page. Stream the pulpit or speech bus. A program mix that includes music, videos, or the room will be captioned. RTMP adds delay on top of speech and translation. Port 1935 has to be reachable from the OBS computer. It is not carried on the HTTPS proxy.

The overlay at `/v/{session}` stays available for a ProPresenter web layer or a browser. Add `?bg=black` when that source cannot use transparency. To put the translation in a ProPresenter lower third, enable Network in ProPresenter 7.9 or later, create a Message whose theme contains `{Translation}`, and set Dismiss to manually. Save the ProPresenter address, message name, and token name on the operator page. Use the ProPresenter computer's LAN address. `127.0.0.1` reaches ProPresenter only when the caption service runs on that same computer. From the API container it does not. ProPresenter receives each final translation. The overlay and the phones still show the grey in-progress line. If ProPresenter does not answer, captions continue.

## Tests and eval

```bash
.venv\Scripts\python -m pytest
.venv\Scripts\python -m eval.harness --engine fake
```

`--engine whisper` runs the real recognizer and downloads the model. Speech clips in `eval/fixtures/manifest.json` stay blocked until real sermon recordings are added. Do not commit those recordings.

## Operator notes

- Start loads nothing. Creating the session loads the models.
- Grey text is still changing. Solid text is final and will not be rewritten.
- Hide removes a bad line from phones that are already open.
- Restore puts back a line that was held because it looked like a long scripture reading.
- Changing the speaker language affects only new audio.
- Ending the session keeps the transcript on screen. Delete transcript removes it from the server and from phones that are open.
- A dropped connection resumes from the last caption. Refreshing the page loads the whole transcript again.
- If the API process restarts, Valkey still has the transcript. The session comes back paused. Press Start to continue. Listeners keep the lines they already have.
- The glossary editor saves terms for that session. The next sentence uses them. Earlier captions stay as they were.
- TXT and SRT on the operator page are the listener transcript: hidden lines are left out, and a long scripture reading stays as the reference.
- The overlay page is `/v/{session id}`. It is transparent so OBS or ProPresenter can key it. Add `?bg=black` when the source cannot use transparency.
- Metrics show listener count and the last caption's speech, translation, and lag times. GPU utilisation appears when `nvidia-smi` is installed.
- Up to four sessions can be live. They share one recognizer and one translator. A fifth start is refused.
