# Hardware

These are sizing notes, not measurements. Phase 1 runs one live session. Do not quote a concurrent-session number until it has been timed on the machine that will serve Sunday.

## Minimum

- NVIDIA GPU with 16 GB of memory, using `Qwen/Qwen2.5-3B-Instruct`, `VLLM_GPU_MEMORY_UTILIZATION` around 0.55, and `WHISPER_COMPUTE_TYPE=int8`
- 24 GB is the default this repository is written for: Whisper large-v3-turbo in float16 beside Qwen2.5-7B-Instruct with vLLM memory utilization 0.70
- A current NVIDIA driver matching the vLLM image. The default image is `vllm/vllm-openai:v0.31.0` (CUDA 13). CUDA 12 hosts should use the `v0.31.0-cu129` tag
- Disk for the weights: about 2 GB for Whisper large-v3-turbo and about 15 GB for the 7B model, plus cache
- An HTTPS proxy in front of port 8080

Whisper and vLLM share one GPU. If a session dies with an out-of-memory error, lower the vLLM memory fraction or the model size. The API is supposed to stop that session rather than publish a half-written line.

## What one GPU can carry

Not measured. The Phase 2 test to run, on the real box, is four pipelines at once and 500 caption subscribers on one of them. Subscribers must not each load a model. Until that test is written up here with the GPU name, the power draw, and the caption latency, the only supported live load is one session.

Listener captions are small text messages. Translated audio in Phase 3 is a separate egress question and is not part of this number.

## Cost per service-hour

Not measured. When the GPU box is known, record the machine's hourly price (or its power draw times the local tariff) and divide by the number of sessions that box actually sustained. A guess does not belong in this file.
