from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.glossary import Glossary
from app.paths import GLOSSARY_PATH
from app.pipeline import SessionPipeline
from app.store import MemoryRepository
from app.captions import Transcript, Word
from app.stt import GoogleRecognizer, ScriptedRecognizer, SonioxRecognizer, WhisperRecognizer, _load_google_credentials, speech_model_label, speech_phrases
from app.translator import GeminiTranslator, GlossaryTranslator, TranslationRequest
from tests.test_phase1 import FAST, document, feed, speak


def test_speech_model_label_names_the_recognizer_used_at_start():
    chirp = GoogleRecognizer("key", "voiceofgod", credentials_path="/tmp/google-login.json")
    fallback = GoogleRecognizer("key", "")
    assert speech_model_label(chirp) == "Google Chirp 3"
    assert speech_model_label(fallback) == "Google latest_long"
    assert speech_model_label(SonioxRecognizer("secret", "stt-rt-v5")) == "Soniox stt-rt-v5"
    assert speech_model_label(WhisperRecognizer("small", "cpu", "int8")) == "Whisper small"
    assert speech_model_label(ScriptedRecognizer([])) == "Development"


def test_chirp_pins_the_session_language():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "alternatives": [
                            {
                                "words": [
                                    {"word": "Halo", "startTime": "0.100s", "endTime": "0.400s"},
                                ]
                            }
                        ]
                    }
                ]
            },
        )

    recognizer = GoogleRecognizer("key", "church-project", client=httpx.Client(transport=httpx.MockTransport(handler)))
    transcript = recognizer.transcribe(np.full(1600, 0.2, dtype=np.float32), "id", "Holy Spirit, pastor")
    assert transcript.words[0].text == "Halo"
    assert transcript.words[0].start == pytest.approx(0.1)
    assert seen["body"]["config"]["model"] == "latest_long"
    assert seen["body"]["config"]["languageCode"] == "id-ID"
    assert seen["body"]["config"]["speechContexts"][0]["phrases"][:2] == ["Holy Spirit", "pastor"]
    assert "Roh Kudus" in seen["body"]["config"]["speechContexts"][0]["phrases"]
    assert "firman Tuhan" in seen["body"]["config"]["speechContexts"][0]["phrases"]
    assert "this Sunday morning" not in seen["body"]["config"]["speechContexts"][0]["phrases"]
    assert seen["url"].startswith("https://speech.googleapis.com/v1/speech:recognize")
    assert "key=" not in seen["url"]
    crowded = speech_phrases(", ".join(f"name {index}" for index in range(45)))
    assert len(crowded) == 40
    assert "let us pray" in crowded
    assert "after the service" in crowded
    indonesian = speech_phrases("Grace Church", language="id")
    assert indonesian[0] == "Grace Church"
    assert "Roh Kudus" in indonesian
    assert "this Sunday morning" not in indonesian
    mixed = speech_phrases(None, language="mix")
    assert "Roh Kudus" in mixed
    assert "Holy Spirit" in mixed


def test_chirp_uses_speech_v2_when_a_service_account_is_available():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "alternatives": [
                            {
                                "transcript": "let us pray",
                                "words": [
                                    {"word": "let", "startOffset": "0s", "endOffset": "0.200s"},
                                    {"word": "us", "startOffset": "0.200s", "endOffset": "0.300s"},
                                    {"word": "pray", "startOffset": "0.300s", "endOffset": "0.600s"},
                                ],
                            }
                        ]
                    }
                ]
            },
        )

    recognizer = GoogleRecognizer(
        "key",
        "church-project",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_provider=lambda: "ya29.test",
    )
    transcript = recognizer.transcribe(np.full(1600, 0.2, dtype=np.float32), "en", "Pastor John")
    assert [word.text for word in transcript.words] == ["let", "us", "pray"]
    assert transcript.words[2].end == pytest.approx(0.6)
    assert seen["url"] == "https://us-speech.googleapis.com/v2/projects/church-project/locations/us/recognizers/_:recognize"
    assert seen["auth"] == "Bearer ya29.test"
    assert seen["body"]["config"]["model"] == "chirp_3"
    assert seen["body"]["config"]["languageCodes"] == ["en-US"]
    assert seen["body"]["config"]["features"]["enableAutomaticPunctuation"] is True
    assert "enableWordTimeOffsets" not in seen["body"]["config"]["features"]
    phrases = seen["body"]["config"]["adaptation"]["phraseSets"][0]["inlinePhraseSet"]["phrases"]
    assert {"value": "Pastor John"} in phrases
    assert "key" not in seen["url"]


def test_a_user_login_file_can_call_chirp(tmp_path):
    path = tmp_path / "adc.json"
    path.write_text(
        json.dumps({
            "type": "authorized_user",
            "client_id": "client",
            "client_secret": "secret",
            "refresh_token": "refresh",
        }),
        encoding="utf-8",
    )
    creds = _load_google_credentials(str(path), "church-project")
    assert creds.quota_project_id == "church-project"


@pytest.mark.asyncio
async def test_gemini_restores_glossary_and_leaves_session_context_empty():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "[[T0]] penuh [[T1]]"}]}}]},
        )

    glossary = Glossary.load(GLOSSARY_PATH).merged(["Grace Church"])
    translator = GeminiTranslator("key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    restored, missing = await translator.translate(
        TranslationRequest("Grace Church is full of grace", "en", glossary, [])
    )
    user = seen["body"]["contents"][0]["parts"][0]["text"]
    assert seen["body"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    assert "Background for the interpreter" in user
    assert "(none)" in user
    assert "[[T0]]" in user
    assert restored == "Grace Church penuh anugerah"
    assert missing == []
    await translator.aclose()


@pytest.mark.asyncio
async def test_revise_parses_json_and_falls_back_to_plain_text():
    from app.translator import REVISE_RULES

    replies = [
        json.dumps({"original": "mungkin [[T0]]", "translation": "perhaps [[T0]]"}),
        "just the translation",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "mungkin kalian" in body["systemInstruction"]["parts"][0]["text"]
        assert REVISE_RULES.splitlines()[0] in body["systemInstruction"]["parts"][0]["text"]
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": replies.pop(0)}]}}]},
        )

    glossary = Glossary.load(GLOSSARY_PATH).merged(["Grace Church"])
    translator = GeminiTranslator("key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = TranslationRequest("morning kalian Grace Church", "id", glossary, [], chosen_language="en")
    corrected, translation, missing = await translator.revise(request)
    assert corrected == "mungkin Grace Church"
    assert translation == "perhaps Grace Church"
    assert missing == []
    same, plain, _missing = await translator.revise(request)
    assert same == request.text
    assert plain == "just the translation"
    await translator.aclose()


class MemorySpeaker:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text: str, language: str) -> bytes:
        self.calls.append((text, language))
        return b"mp3-" + text.encode("utf-8")


class CountingRecognizer(ScriptedRecognizer):
    billed_once = True

    def __init__(self):
        super().__init__(["Hello"])
        self.lengths = []

    def transcribe(self, audio, language, hotwords=None):
        self.lengths.append(len(np.asarray(audio)))
        return Transcript([Word("Hello", 0.0, 0.4)])


@pytest.mark.asyncio
async def test_growing_audio_is_sent_once():
    recognizer = CountingRecognizer()
    pipe = SessionPipeline(
        document(),
        MemoryRepository(),
        recognizer,
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    first = np.full(16000, 0.2, dtype=np.float32)
    second = np.concatenate([first, np.full(16000, 0.3, dtype=np.float32)])
    opening = await pipe._recognize(first, "en", None)
    grown = await pipe._recognize(second, "en", None)
    skipped = await pipe._recognize(np.concatenate([second, np.full(800, 0.1, dtype=np.float32)]), "en", None)
    assert recognizer.lengths == [16000, 16000]
    assert opening.words[0].start == pytest.approx(0.0)
    assert grown.words[-1].start == pytest.approx(1.0)
    assert skipped.words == grown.words
    whole = await pipe._recognize(second, "en", None, whole=True)
    assert recognizer.lengths[-1] == 32000
    assert whole.words[0].text == "Hello"


@pytest.mark.asyncio
async def test_voice_runs_for_a_final_line_only():
    speaker = MemorySpeaker()
    glossary = Glossary.load(GLOSSARY_PATH)
    spoken = document()
    spoken["voice"] = True
    pipe = SessionPipeline(
        spoken,
        MemoryRepository(),
        ScriptedRecognizer(["Hello", "church", "this", "morning."]),
        GlossaryTranslator(),
        glossary,
        FAST,
        speaker,
    )
    await speak(pipe, 1.2)
    finals = [line for line in pipe.doc["lines"] if line.get("state") == "final"]
    assert finals
    assert len(speaker.calls) == len(finals)
    assert all(language == "id" for _text, language in speaker.calls)
    assert finals[0]["audio"] is True
    assert pipe.clips[finals[0]["id"]]


@pytest.mark.asyncio
async def test_a_growing_sentence_is_spoken_to_the_end_before_the_next_one():
    from app.pipeline import speech_addition

    assert speech_addition("Halo gereja.", "Halo gereja pagi ini.") == "pagi ini."
    assert speech_addition("Halo gereja.", "Halo gereja.") is None
    speaker = MemorySpeaker()
    spoken = document()
    spoken["voice"] = True
    pipe = SessionPipeline(
        spoken,
        MemoryRepository(),
        ScriptedRecognizer(["Hello"]),
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
        speaker,
    )

    def line(line_id: str, text: str) -> dict:
        return {
            "id": line_id,
            "state": "final",
            "original": text,
            "translation": text,
            "listener_original": text,
            "listener_translation": text,
            "hidden": False,
            "suppressed": False,
            "t0": 0,
            "t1": 1,
        }

    pipe.doc["lines"] = [line("L00001", "Halo gereja."), line("L00002", "Kalimat berikutnya.")]
    pipe._enqueue_speech("L00001", "Halo gereja.", pipe.audio_epoch)
    pipe._enqueue_speech("L00001", "Halo gereja pagi ini.", pipe.audio_epoch)
    pipe._enqueue_speech("L00002", "Kalimat berikutnya.", pipe.audio_epoch)
    await pipe.idle()
    assert speaker.calls == [
        ("Halo gereja.", "id"),
        ("pagi ini.", "id"),
        ("Kalimat berikutnya.", "id"),
    ]
    assert b"Halo gereja." in pipe.clips["L00001"]
    assert b"pagi ini." in pipe.clips["L00001#2"]
    assert pipe.doc["lines"][0]["audio_id"] == "L00001#2"


@pytest.mark.asyncio
async def test_the_caption_is_sent_before_the_spoken_voice_is_ready():
    speaker = MemorySpeaker()
    glossary = Glossary.load(GLOSSARY_PATH)
    spoken = document()
    spoken["voice"] = True
    pipe = SessionPipeline(
        spoken,
        MemoryRepository(),
        ScriptedRecognizer(["Hello", "church", "this", "morning."]),
        GlossaryTranslator(),
        glossary,
        FAST,
        speaker,
    )
    audio_flags = []
    emit = pipe._emit

    async def spy(event_type, line=None):
        if event_type == "line" and line is not None:
            audio_flags.append(bool(line.get("audio")))
        await emit(event_type, line)

    pipe._emit = spy
    await speak(pipe, 1.2)
    assert audio_flags[:2] == [False, True]
    assert pipe.doc["lines"][0]["audio"] is True


@pytest.mark.asyncio
async def test_audio_route_returns_the_stored_clip():
    from tests.church import make_app, sign_in

    app = await make_app(["Hello"], max_live=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        created = await client.post("/api/sessions", json={"speaker_language": "en"})
        assert created.status_code == 200
        session_id = created.json()["id"]
        pipe = app.state.director.pipeline(session_id)
        pipe.clips["L00001"] = b"fake-mp3"
        found = await client.get(f"/api/sessions/{session_id}/audio/L00001")
        missing = await client.get(f"/api/sessions/{session_id}/audio/L00002")
    assert found.status_code == 200
    assert found.content == b"fake-mp3"
    assert found.headers["content-type"].startswith("audio/mpeg")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_spoken_translation_stays_off_until_the_service_turns_it_on():
    from tests.church import make_app, sign_in

    speaker = MemorySpeaker()
    app = await make_app(["Hello", "church", "this", "morning."], max_live=1, speaker=speaker)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        current = await client.get("/api/settings")
        created = await client.post("/api/sessions", json={"speaker_language": "en"})
        session_id = created.json()["id"]
        pipe = app.state.director.pipeline(session_id)
        await speak(pipe, 1.2)
        assert speaker.calls == []
        saved = await client.post(f"/api/sessions/{session_id}/voice", json={"tts_enabled": True})
        clip = await pipe._store_clip("L00009", "Halo gereja")
        assert clip is True
        assert speaker.calls == [("Halo gereja", "id")]
        captions = await client.get(f"/api/sessions/{session_id}/captions")
        page = await client.get("/settings")
        denied = await client.put("/api/settings", json={"tts_enabled": "yes"})
    finals = [line for line in pipe.doc["lines"] if line.get("state") == "final"]
    assert current.status_code == 200
    assert current.json()["tts_enabled"] is False
    assert finals and finals[0]["audio"] is False
    assert saved.status_code == 200
    assert saved.json() == {"tts_enabled": True}
    assert pipe.voice_enabled is True
    assert captions.json()["voice"] is True
    assert page.status_code == 200
    assert "Spoken translation" in page.text
    assert denied.status_code == 400


@pytest.mark.asyncio
async def test_settings_require_a_login():
    from tests.church import make_app, sign_in

    app = await make_app([])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.put("/api/settings", json={"tts_enabled": True})
        await sign_in(client)
        allowed = await client.put("/api/settings", json={"tts_enabled": True, "speaker_language": "en"})
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["tts_enabled"] is True
