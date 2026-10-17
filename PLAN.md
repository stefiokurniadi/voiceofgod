# Build: open-source live sermon translator (English ↔ Bahasa Indonesia)

## Role

You are a senior engineer building a production-minded MVP. Work in phases, commit working code at the end of each phase, and stop to report before starting the next one.

## What we're building

A self-hosted service that takes a live sermon audio feed and delivers real-time translated captions (and later translated audio) to listeners' phones. Listeners scan a QR code, open a web page, pick a language, and follow along. No app install, no listener account.

- Languages: English and Bahasa Indonesia only, in BOTH directions. The speaker language is set per session.
- Setting: church services in Indonesia and Singapore, with 50–500 listeners per session and several sessions running concurrently on Sunday mornings.

## Hard constraints

1. Use open-source / open-weight components only. No paid AI APIs.
2. Every model and library must be licensed for commercial use. Before adopting a model, check its CURRENT licence and record it in `LICENSES.md` (name, version, licence, link, commercial use yes/no). Reject non-commercial licences such as CC-BY-NC. Known examples to avoid: Meta MMS-TTS, NLLB, SeamlessStreaming.
3. Everything runs self-hosted on one Linux GPU server, deployed with Docker Compose.
4. Do not bundle or display copyrighted Bible translations (for example LAI Terjemahan Baru) or song lyrics. Verse references may be detected and shown as a reference only. Do not ship a Bible or lyrics corpus, and do not look up verse text to fill in what the microphone heard.

## Suggested stack

Verify current versions, then choose. Justify any change in `DECISIONS.md`.

- Speech-to-text: faster-whisper with Whisper large-v3 or large-v3-turbo, using streaming with a local-agreement / stabilisation strategy.
- Translation: an open-weight LLM strong in Indonesian (evaluate SEA-LION and Sahabat-AI against Gemma or Qwen), served with vLLM.
- Text-to-speech (Phase 3): an English voice from Kokoro or Piper; for Indonesian, evaluate the available commercial-safe options and plan fine-tuning if none is good enough.
- Backend: Python, FastAPI, WebSockets, Redis for session state and fan-out.
- Frontend: a lightweight, mobile-first listener page and an operator console.
- Audio ingest: browser microphone capture and a USB audio interface on the operator laptop. RTMP/SRT ingest is a stretch goal.

## Translation requirements

- Translate sentence by sentence and pass the previous 3–5 sentences as context.
- Support a per-church glossary of preferred translations and do-not-translate terms. Seed it with common church terms, for example:
  - Holy Spirit ↔ Roh Kudus
  - grace ↔ anugerah
  - worship service ↔ kebaktian / ibadah
  - pastor ↔ gembala / pendeta
  - Church names stay untranslated.
- Handle code-switching: Indonesian sermons often mix in English words and phrases.
- The output register should be natural and respectful, suitable for worship. Use formal Indonesian, not slang. Address the congregation as "Anda", not "kamu". Keep the preacher's person (I / you) and do not add, soften, or strengthen doctrine. Never invent a verse, a citation, or a closing the speaker did not say.
- Church glossary may set the preferred divine name (Tuhan or Allah) and other house terms. Defaults stay neutral until a church sets them.

## Phases

### Phase 1: captions MVP

- The operator creates a session, picks the speaker language, presses Start, and sees a live transcript preview plus the translation. Models are loaded when the session is created, so Start is not a cold start.
- Listeners get a QR code and URL, a language toggle (original or translated), adjustable font size, and auto-scroll. Joining or refreshing receives the finalised caption history and the current provisional line.
- Captions behave as follows: show provisional text greyed out, replace it with finalised text, and never let earlier finalised lines jump around. Each line has a stable id and carries the audio time range, so a later export does not have to invent timing.
- The operator can hide a bad finalised line. Hidden lines disappear for listeners already connected and for anyone who joins later.
- The listener URL is an unguessable capability link. Operator controls use a separate secret and are not available from the listener page.
- Targets: finalised translated captions within 3 seconds of the end of the utterance at p50, and within 5 seconds at p95. Measure pipeline time on the server, and separately measure time until a listener on the same LAN renders the line. Silence and music are not part of the latency sample.

### Phase 2: robustness and operations

- Reconnect on network drops for both the operator and listeners, and resume from the last caption.
- Run multiple concurrent sessions, each with an isolated pipeline.
- Glossary editor in the operator console.
- Save the transcript and translation after each session; downloadable as TXT and SRT.
- An overlay page with a transparent background for OBS/ProPresenter browser sources.
- Basic metrics: latency per stage, listener count, GPU utilisation.

### Phase 3: translated audio

- Text-to-speech of finalised sentences, streamed to listeners who choose "listen".
- Audio must keep playing when the phone screen locks.
- Target: translated audio within 4 seconds of speech at p50.

## Evaluation

Build this in Phase 1, not at the end.

- Create `/eval` with a harness that replays recorded sermon audio through the pipeline and reports:
  - speech-to-text word error rate against reference transcripts, split by English, Indonesian, and code-switched clips
  - translation quality with chrF against reference translations; add COMET only when that checkpoint's current licence allows commercial use
  - latency from utterance end to finalised translation
- Code-switched WER uses the mixed-language transcript as the reference. Do not score code-switched speech against a monolingual translation.
- Placeholder clips, before any real sermons arrive, must include silence, music or singing, fast speech, a run-on sentence, a self-correction, a verse reading, glossary terms, and a personal name that collides with a church term (for example Grace).
- Real recorded sermons in English and Indonesian, including code-switched audio, will be provided later. Treat those recordings as church data: do not commit them, and do not log raw audio in production.
- Re-run the eval after every model or prompt change and log the results in `EVAL_LOG.md`.

## Deliverables

- Repository with `README.md` (setup on a fresh GPU box in under 30 minutes), `docker-compose.yml`, `LICENSES.md`, `DECISIONS.md`, `EVAL_LOG.md`.
- A hardware note covering minimum GPU, how many concurrent sessions one GPU sustains (measured, not guessed), and estimated cost per service-hour.

## Edge cases

Expected behavior. Phase 1 unless a later phase is named.

### Audio and speech-to-text

- Silence, room tone, applause, and singing must not become captions. Gate recognition with voice activity detection. On long non-speech, show nothing new rather than Whisper's invented text. During congregational singing, listeners see a single non-lyric status such as "Worship", never transcribed lyrics.
- If the microphone permission is denied, the USB interface is unplugged, or the capture tab is throttled in the background, pause the pipeline and show that state on the operator console. Do not keep translating the last buffer.
- Normalise ingest to 16 kHz mono before recognition. Ignore the unused stereo channel instead of blending music bleed into the pulpit mic.
- Pin recognition to the session's speaker language. Do not let language auto-detect flip the session because of one English phrase.
- A sustained language change is an operator action: they can switch the speaker language without ending the session. Only new audio uses the new language. Lines already finalised stay as they are.
- Words in a third language (Malay, Mandarin, Tamil, Hokkien, and others) stay as spoken. Do not guess a translation.
- Force-finalise a caption if local agreement has not settled within a short bound (aim: under the p95 latency budget). Waiting forever for agreement is worse than a slightly early final line.
- Very short backchannels ("ya", "amin") can finalise as their own short line. Do not glue them onto the next sentence.

### Caption lifecycle

- Finalised text is immutable. If the preacher corrects themselves ("maksud saya…"), the correction is a new line. It does not rewrite the earlier line.
- Translate each finalised line in order. A late translation for line N never appears after line N+1 and never overwrites another line.
- Provisional translation follows the provisional transcript and is also greyed out. When the transcript finalises, replace the provisional translation in place.
- If the speaker does not pause, split on a maximum hold (about one sentence or a few seconds of speech) so a run-on does not stall the room. Pass the previous 3–5 finalised sentences as context anyway.
- If the pipeline falls behind, keep emitting in order and show the lag on the operator console. Do not drop or reorder sermon text to catch up. If lag stays past a configured ceiling, tell the operator; do not fail silently.
- Auto-scroll follows the live tail. If a listener scrolls up to reread, stop following and offer a way back to live. Changing font size or language keeps their place on the same line.
- The language toggle switches the whole history already on the page, because every line stores both the original and the translation. It does not ask the server to retranslate.
- Ending the session finalises or discards the open provisional line, flushes translations still in flight, then stops. Listeners see a clear ended state and keep the transcript on screen.

### Translation fidelity and glossary

- Apply do-not-translate terms before preferred translations. Longest matching phrase wins, so "Holy Spirit" beats "Spirit", and a church's full name beats a shorter name inside it.
- A listed proper name stays untranslated even when it is also a church word. "Grace" the person, or "Grace Church", is not "anugerah".
- Glossary hits are constraints on the model, then checked on the way out. If the model missed a required term, substitute the preferred term when the source term is unambiguous. If it is ambiguous, leave the source wording.
- One speaker language per session remains the rule. Code-switching inside that language is the translator's job, not a reason to open a second session.
- Two people sharing one microphone (interview, testimony) stay one session. The operator hides a line if the model attributes or translates it badly. Speaker diarisation is out of scope.

### Scripture, copyright, and sensitive content

- When a book plus chapter and verse is detected, show that citation in the language of the listener's toggle ("John 3:16" or "Yohanes 3:16"). Keep the speaker's surrounding commentary. Do not attach verse body from a database.
- A number that is not a citation ("chapter 3 of my story", a budget, a date) stays in the sentence. When detection is unsure, show the spoken transcript and do not suppress it.
- If the speaker reads a passage at length, suppress the verbatim reading and leave the reference plus a short status that scripture is being read. The operator can restore the line if the suppression was wrong.
- Prayer requests, names, illnesses, and phone numbers may be spoken. Do not write raw audio or full transcripts to application logs. Session recordings used for eval stay outside the git repo. The church decides retention; the product must be able to delete a session's transcript.

### Access, clients, and the room

- Operator pages need HTTPS, because browsers only expose the microphone in a secure context (localhost is the exception). Listener pages also use that HTTPS origin so a QR code has one URL shape.
- Session ids are long random tokens. The operator secret is distinct from the listener token. Knowing the QR code does not grant Start, Stop, or Hide.
- Phase 1 is a single live session on the server. The data model still isolates all state by session id so Phase 2 does not redesign storage.
- Listeners on poor church Wi-Fi receive text events only. Keep payloads small. In-app browsers (WhatsApp, Instagram) must at least show the last finalised lines if a WebSocket fails; full resume after a drop is Phase 2.
- Phase 2 reconnect resumes from the last caption id the client ack'd, without duplicating lines. Design the Phase 1 event stream with monotonic ids so resume is possible.
- Phase 2 concurrent-session test is four simultaneous pipelines on the one GPU, plus a fan-out check of 500 caption subscribers on one session. Subscribers must not each run a model.
- If the GPU runs out of memory or a model process dies, stop that session with an operator-visible error. Do not publish partial tokens as a final caption.

### Translated audio (Phase 3)

- Captions remain the mode that has to work for 500 listeners. Measure server egress before offering "listen" at that occupancy. If egress cannot carry 500 audio streams, cap listen mode and say so in the hardware note.
- Audio starts only after a user gesture. It must continue with the screen locked, on iOS and Android, including in-app browsers where that is possible. Where the browser forbids background playback, say so on the page instead of pretending it is playing.
- Synthesise a finalised sentence only after its text is final. If speech runs ahead of synthesis, leave a gap. Do not speak two sentences at once, and do not skip a sentence without leaving the caption.
- Code-switched sentences need a voiced approach that does not mangle the other language. If one voice cannot do both, split the line across the English and Indonesian voices at the language change.
- Do not play translated audio back into the room where the pulpit microphone can hear it.

## Working rules

- Ask before adding any dependency with an unclear licence.
- Prefer simple, boring infrastructure over clever infrastructure.
- At the end of each phase, report what works, measured latency and quality numbers, known issues, and what is proposed next.
