# Licences

Checked 18 October 2026 against the current upstream licence text. "Commercial" means a church may run this service, including as part of a paid ministry operation. Pinned versions are what this repository installs or deploys. Transitive packages inside those projects stay under their own licences.

## Used

| Name | Version | Licence | Commercial | Link |
| --- | --- | --- | --- | --- |
| faster-whisper | >=1.1.0 | MIT | yes | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| Whisper large-v3-turbo (default) and large-v3 | OpenAI Whisper weights, September 2024 turbo | MIT | yes | https://github.com/openai/whisper/blob/main/LICENSE |
| CTranslate2 | pulled by faster-whisper | MIT | yes | https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE |
| Silero VAD | bundled by faster-whisper | MIT | yes | https://github.com/snakers4/silero-vad/blob/master/LICENSE |
| ONNX Runtime | pulled by faster-whisper | MIT | yes | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| Qwen2.5-7B-Instruct | default translation weights | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE |
| Qwen2.5-3B-Instruct | optional, only if `VLLM_MODEL` is set to it | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE |
| vLLM | Docker image `vllm/vllm-openai:v0.31.0` | Apache-2.0 | yes | https://github.com/vllm-project/vllm/blob/main/LICENSE |
| Valkey | Docker image `valkey/valkey:8` | BSD-3-Clause | yes | https://github.com/valkey-io/valkey/blob/8.0/COPYING |
| PostgreSQL | Docker image `postgres:16`; this laptop uses 17 | PostgreSQL License | yes | https://www.postgresql.org/about/licence/ |
| asyncpg | 0.32.0 | Apache-2.0 | yes | https://github.com/MagicStack/asyncpg/blob/master/LICENSE |
| redis-py | >=5.2 (client only) | MIT | yes | https://github.com/redis/redis-py/blob/master/LICENSE |
| FastAPI | 0.143.3 | MIT | yes | https://github.com/fastapi/fastapi/blob/master/LICENSE |
| Uvicorn | 0.54.0 | BSD-3-Clause | yes | https://github.com/encode/uvicorn/blob/master/LICENSE |
| Starlette | 1.8.0 (pulled by FastAPI) | BSD-3-Clause | yes | https://github.com/encode/starlette/blob/master/LICENSE |
| Pydantic | 2.14.1 | MIT | yes | https://github.com/pydantic/pydantic/blob/main/LICENSE |
| pydantic-settings | 2.16.0 | MIT | yes | https://github.com/pydantic/pydantic-settings/blob/main/LICENSE |
| HTTPX | 0.28.1 | BSD-3-Clause | yes | https://github.com/encode/httpx/blob/master/LICENSE |
| PyYAML | 6.0.3 | MIT | yes | https://github.com/yaml/pyyaml/blob/master/LICENSE |
| NumPy | 2.5.4 | BSD-3-Clause | yes | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| segno | 1.6.6 | BSD-3-Clause | yes | https://github.com/heuer/segno/blob/master/LICENSE |
| MediaMTX | Docker image `bluenviron/mediamtx:1.13.1` | MIT | yes | https://github.com/bluenviron/mediamtx/blob/main/LICENSE |
| FFmpeg | Debian bookworm `ffmpeg` package, invoked as its own process | GPL-2.0-or-later for the Debian build | yes | https://ffmpeg.org/legal.html |
| Cloud Speech-to-Text V2 (`chirp_3`) | Google Cloud API, checked 18 October 2026 | Google Cloud terms, not an open-source weight | yes, as a paid API | https://cloud.google.com/speech-to-text/docs/models/chirp-3 |
| Gemini 3.5 Flash | `gemini-3.5-flash`, checked 18 October 2026 | Gemini API terms, not an open-source weight | yes, on the paid tier | https://ai.google.dev/gemini-api/docs/pricing |
| Cloud Text-to-Speech WaveNet | `en-US-Wavenet-D`, `id-ID-Wavenet-B`, checked 18 October 2026 | Google Cloud terms, not an open-source weight | yes, as a paid API | https://cloud.google.com/text-to-speech/docs/voices |

Python package versions above are from the development environment used to run the tests. The GPU image installs the same minimums from `pyproject.toml` and may resolve a newer patch. Re-check this file if a dependency is upgraded.

## Considered and not used

| Name | Licence found | Commercial | Why it is not in the default stack |
| --- | --- | --- | --- |
| Redis server | RSALv2 / SSPL (source-available, not OSI) | internal use only under those terms | Replaced by Valkey, which speaks the same protocol. |
| SEA-LION | MIT where the project can, otherwise the base model's licence (Llama or Gemma) | depends on the exact checkpoint | Not selected until a sermon-audio eval exists. Do not add a checkpoint without writing its current licence here. |
| Sahabat-AI Llama 3 8B | Llama 3 Community License | yes, under Meta's acceptable-use terms, with a separate licence above 700 million monthly users | Not selected. The licence is not Apache/MIT. |
| Sahabat-AI Gemma 2 and other Gemma models | Gemma Terms of Use | yes, under Google's prohibited-use policy | Not selected. The licence is not Apache/MIT. |
| Meta MMS-TTS, NLLB, SeamlessStreaming | non-commercial or research terms | no | Excluded. Do not add them. |
