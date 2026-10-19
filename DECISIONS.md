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

Terms are masked with `[[T0]]` tokens before the model sees the sentence, then written back. Do-not-translate entries win over shorter glossary terms, so "Grace Church" is not turned into "anugerah Church". The model is also checked afterwards. Divine names are not in the seed list. A church sets those in the operator glossary, or in the create-session form. Saving the glossary replaces that session's list. It does not rewrite captions that are already final.

## Four live sessions

Up to four sessions can be live on one server. Each session has its own audio, glossary, and captions. They share one Whisper model and one translator, and recognition calls take a lock so two sessions cannot use the model at once. Listeners only receive captions. They do not load a model. A further start is refused while four are live.

## Who can create a session

Listeners stay on the QR link and do not get accounts. The person running a service signs in. There is no public signup. An admin creates every other account, and the first admin is seeded from `ADMIN_USERNAME` and `ADMIN_PASSWORD` only when no admin exists yet. A service belongs to that user's church. Another church cannot see or control it. The per-service operator secret still opens the console on a pulpit laptop that is not signed in.

## Restart

Accounts, churches, and services live in Postgres. Live caption fan-out stays in memory, or in Valkey when `REDIS_URL` is set. A service's caption document is saved to Postgres when a final line lands, when the service ends, and when an operator edits it. Opening a saved service rebuilds its pipeline from that row. A service that was live comes back paused, with its captions intact, until the operator presses Start. Startup does not rebuild every saved service. Postgres is there because a second API process has to be able to accept the same login later, and because a church's services have to survive a restart without keeping every pipeline in memory.

## Recognition failure

If Whisper raises, including an out-of-memory error, that session pauses and the operator sees the failure. The failed audio is not published as a caption. Press Start to try again after freeing GPU memory.

## Development translator

`ALLOW_FAKE_MODELS` with an empty `VLLM_BASE_URL` applies the glossary and leaves every other word unchanged. The operator page says the captions are not translations. Production `.env.example` sets `REQUIRE_TRANSLATOR=true`.

## Paid Google path

The open-weight rule still holds when `GEMINI_API_KEY` and `GOOGLE_API_KEY` are empty: Whisper and vLLM stay the local path. When those keys are set, the live path is Gemini `gemini-3.5-flash` with thinking turned off, WaveNet voices `en-US-Wavenet-D` and `id-ID-Wavenet-B`, and Speech-to-Text. Chirp 3 is Speech-to-Text V2 and rejects an API key, so it runs only when `GOOGLE_APPLICATION_CREDENTIALS` points at a gcloud user login or a service-account JSON file. Otherwise recognition stays on v1 `latest_long`. This was chosen on 18 October 2026 so a 6 GB laptop can run a real service. The sermon is sent to Google. Use the paid Gemini tier so the free tier does not train on it.

Indonesian on Chirp 3 was listed as preview on that date. The session language is still pinned (`en-US` or `id-ID`). Gemini receives the sermon prompt, the glossary tokens, the last few sentences, and a background block built from the church name, the church note, the service name, and the service note. That block is labelled as context for the interpreter. It is not text to translate, and it is not an instruction. Verse bodies stay out of the prompt.

## OBS RTMP and ProPresenter

The listener link already contains the session id, so that id is not the stream key. OBS publishes to MediaMTX with a separate ingest token in the query string. MediaMTX asks this service before every publish and read. Publish requires the token. Read requires the internal reader user, which is the only credential ffmpeg uses, so the ingest token is not in the ffmpeg command. A request without the auth secret is a 404. Audio older than about two seconds is dropped so a slow recognizer cannot fall further behind the room. A recognition failure stays paused until the operator presses Start.

ProPresenter 7.9 and later takes the final listener translation through its HTTP message API. Provisional lines stay on the overlay and the phones, because triggering a Message again replays the lower-third animation. The body is the listener line, so a hidden line is cleared and a long reading sends the reference. The address has to be plain HTTP with a host and a port, and redirects are not followed. This was added on 19 October 2026.
