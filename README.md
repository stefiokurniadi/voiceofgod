# voiceofgod

Self-hosted live captions for a sermon. The operator sends the pulpit microphone from a browser. Listeners scan a QR code, pick English or Bahasa Indonesia, and follow along. There is no listener account and no app to install.

Phase 1 is captions. Translated audio is later. One live session runs at a time.

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

Edit `.env` and set `PUBLIC_BASE_URL` to the HTTPS origin listeners will open, for example `https://captions.example.org`.

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
- Ending the session keeps the transcript on screen. Delete the session from the API when the church wants the transcript gone: `DELETE /api/sessions/{id}` with the `X-Operator-Token` header.
