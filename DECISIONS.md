# Decisions

## Translation model

The default is `Qwen/Qwen2.5-7B-Instruct`, served by vLLM. Its current licence is Apache-2.0, so a church can run it without a separate grant.

SEA-LION and Sahabat-AI are the Indonesian-focused models the plan asked to evaluate. Their current instruct checkpoints inherit the Llama 3 Community License or the Gemma Terms. Both allow commercial use for an organisation this size, and both add acceptable-use rules that Apache-2.0 does not. They are not rejected on quality. They are not the default because we do not yet have sermon audio to show that the extra licence terms buy a better caption. Swap one in only by changing `VLLM_MODEL`, recording the checkpoint's current licence in `LICENSES.md`, and appending an `EVAL_LOG.md` entry from a real run.

Gemma is in the same position: commercial use is allowed under Google's terms, and it is not the default.

On a 16 GB GPU, set `VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct` and lower `VLLM_GPU_MEMORY_UTILIZATION`. That checkpoint is also Apache-2.0. Expect worse Indonesian. Measure it. Do not guess.

## Speech model

`large-v3-turbo` is the default. It is the OpenAI Whisper turbo checkpoint, MIT licensed, and it is the one that can plausibly leave room in a 3 second caption budget. `WHISPER_MODEL=large-v3` remains available when a recording shows the turbo model is dropping words. Both are MIT.

Recognition is pinned to the session language. A code-switched phrase does not flip the session.

## Stabilisation

Local agreement is implemented in this repository rather than imported from a streaming-Whisper package. A word is confirmed when two successive transcripts share it as a prefix. A confirmed sentence with a still-changing tail is finalised early. Silence after speech finalises the rest. Continuous speech is cut on a maximum hold so a run-on sentence cannot stall the room. Earlier final lines are not edited.

## Session store

The plan named Redis. Current Redis server releases are RSALv2/SSPL, which are not open-source licences. Valkey is the BSD-licensed continuation and speaks the same protocol, so the client is still redis-py. If `REDIS_URL` is empty, state stays in the API process. That is for tests and a laptop. Production compose always sets Valkey.

## Scripture and singing

No Bible text and no lyrics are shipped. A book name plus chapter and verse is a citation. A long tail after a citation is treated as a reading: listeners see the reference and a short status, and the operator can restore the line. A short mention such as "turn to John 3:16" is left in the caption. Music and silence do not become captions. Sustained non-speech energy shows a Worship banner instead of lyrics.

## Glossary

Terms are masked with `[[T0]]` tokens before the model sees the sentence, then written back. Do-not-translate entries win over shorter glossary terms, so "Grace Church" is not turned into "anugerah Church". The model is also checked afterwards. Divine names are not in the seed list. A church sets those when it creates a session, once the editor exists. Until then, pass them in the create-session JSON or put names in the "do not translate" box.

## One live session

Phase 1 refuses to start a second session while one is live. Session state is already keyed by id, so Phase 2 can raise the limit without a new storage design. Four concurrent pipelines is a Phase 2 measurement, not a number this phase claims.

## Development translator

`ALLOW_FAKE_MODELS` with an empty `VLLM_BASE_URL` applies the glossary and leaves every other word unchanged. The operator page says the captions are not translations. Production `.env.example` sets `REQUIRE_TRANSLATOR=true`.
