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

Listener captions are small text messages. Spoken translation is generated once per final line and kept in memory. Five hundred listeners each pulling that audio has not been measured. Until it is, listen mode is not a supported load at that occupancy. Captions remain the mode for a full room.

## Cost per service-hour

Not measured on a GPU. The hosted Google path uses published list prices, not a timed service on this machine: about $0.016 per minute of speech for Chirp, about $1.50 / $9 per million Gemini 3.5 Flash tokens with thinking off, and WaveNet at $16 per million characters after one million free characters in a month. A two-hour service with about 70 minutes of talking is about $2.05 while that voice allowance lasts. Listeners do not multiply the API bill. A guess of GPU cost still does not belong here.
