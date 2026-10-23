from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.glossary import Glossary
from app.paths import GLOSSARY_PATH
from app.director import Director
from app.store import MemoryRepository
from app.stt import GoogleRecognizer, ScriptedRecognizer, SonioxRecognizer, WhisperRecognizer, soniox_request, words_from_soniox
from app.translator import GlossaryTranslator
from app.tts import SonioxSpeaker
from app.api import _standard_label
from tests.test_phase1 import FAST


def test_soniox_request_uses_language_hints_and_church_names():
    mixed = soniox_request("stt-rt-v5", "mix", "Pastor John")
    assert mixed["model"] == "stt-rt-v5"
    assert mixed["audio_format"] == "pcm_s16le"
    assert mixed["sample_rate"] == 16000
    assert mixed["num_channels"] == 1
    assert mixed["language_hints"] == ["id", "en"]
    assert mixed["context"]["terms"][0] == "Pastor John"
    assert soniox_request("stt-rt-v5", "id", None)["language_hints"] == ["id"]
    plain = soniox_request("stt-rt-v5", "en", None)
    assert plain["language_hints"] == ["en"]
    assert "Holy Spirit" in plain["context"]["terms"]
    indonesian = soniox_request("stt-rt-v5", "id", None)
    assert "Roh Kudus" in indonesian["context"]["terms"]
    assert "this Sunday morning" not in indonesian["context"]["terms"]
    assert "syiar" in mixed["context"]["terms"]
    assert "Holy Spirit" in mixed["context"]["terms"]


def test_soniox_tokens_join_into_words():
    words = words_from_soniox([
        {"text": "Hel", "start_ms": 0, "end_ms": 100},
        {"text": "lo", "start_ms": 100, "end_ms": 200},
        {"text": " church", "start_ms": 220, "end_ms": 500},
    ])
    assert [word.text for word in words] == ["Hello", "church"]
    assert words[0].start == pytest.approx(0)
    assert words[0].end == pytest.approx(0.2)
    assert words[1].start == pytest.approx(0.22)
    assert words[1].end == pytest.approx(0.5)
    assert words_from_soniox([
        {"text": "Thanks", "start_ms": 0, "end_ms": 200},
        {"text": " for", "start_ms": 200, "end_ms": 300},
        {"text": " watching", "start_ms": 300, "end_ms": 600},
    ]) == []


class _Socket:
    def __init__(self, messages):
        self.sent = []
        self.messages = list(messages)

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout=None):
        del timeout
        return self.messages.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_soniox_recognizer_sends_one_clip_and_keeps_final_tokens():
    seen = {}
    sock = _Socket([
        json.dumps({"tokens": [{"text": "Hel", "is_final": False, "start_ms": 0, "end_ms": 40}]}),
        json.dumps({
            "tokens": [
                {"text": "Hello", "is_final": True, "start_ms": 0, "end_ms": 200},
                {"text": " church", "is_final": True, "start_ms": 200, "end_ms": 480},
            ],
            "finished": True,
        }),
    ])

    def connect(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs["additional_headers"]
        return sock

    recognizer = SonioxRecognizer("secret", connect=connect)
    transcript = recognizer.transcribe(np.full(1600, 0.2, dtype=np.float32), "en", "Grace Church")
    assert [word.text for word in transcript.words] == ["Hello", "church"]
    assert recognizer.billed_once is True
    assert seen["url"] == "wss://stt-rt.soniox.com/transcribe-websocket"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    config = json.loads(sock.sent[0])
    assert config["language_hints"] == ["en"]
    assert config["context"]["terms"][0] == "Grace Church"
    assert isinstance(sock.sent[1], bytes) and sock.sent[1]
    assert sock.sent[-1] == ""


@pytest.mark.asyncio
async def test_soniox_speaker_asks_for_mp3():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"mp3-bytes")

    speaker = SonioxSpeaker("secret", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    clip = await speaker.synthesize("Halo gereja", "id")
    await speaker.aclose()
    assert clip == b"mp3-bytes"
    assert seen["url"] == "https://tts-rt.soniox.com/tts"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"] == {
        "model": "tts-rt-v2",
        "language": "id",
        "voice": "Adrian",
        "audio_format": "mp3",
        "text": "Halo gereja",
    }


def test_standard_recognizer_label():
    assert _standard_label(GoogleRecognizer("key", "project")) == "Google"
    assert _standard_label(WhisperRecognizer("tiny", "cpu", "int8")) == "Whisper"
    assert _standard_label(ScriptedRecognizer([])) == "Standard"


@pytest.mark.asyncio
async def test_a_session_uses_the_saved_speech_providers():
    default = ScriptedRecognizer(["Default"])
    soniox = ScriptedRecognizer(["Soniox"])
    google = object()
    voice = object()
    director = Director(
        Settings(allow_fake_models=True),
        MemoryRepository(),
        default,
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
        google,
        recognizers={"default": default, "soniox": soniox},
        speakers={"google": google, "soniox": voice},
    )
    user = {
        "church_id": "church",
        "church_name": "Grace Church",
        "church_note": "",
        "keep_names": [],
        "stt_provider": "soniox",
        "tts_provider": "soniox",
    }
    created = await director.create_session(user, "Sunday", "", None, "en", False, "http://test")
    pipeline = director.pipeline(created["id"])
    assert pipeline.recognizer is soniox
    assert pipeline.speaker is voice
    assert pipeline.doc["stt_provider"] == "soniox"
    assert pipeline.doc["tts_provider"] == "soniox"
    await director.apply_speech("church", "default", "google")
    assert pipeline.doc["stt_provider"] == "default"
    assert pipeline.doc["tts_provider"] == "google"
    assert pipeline.recognizer is default
    assert pipeline.speaker is google

    bare = Director(
        Settings(allow_fake_models=True),
        MemoryRepository(),
        default,
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    created = await bare.create_session(user, "Sunday", "", None, "en", False, "http://test")
    pipeline = bare.pipeline(created["id"])
    assert pipeline.recognizer is default
    assert pipeline.speaker is None
    assert pipeline.doc["stt_provider"] == "soniox"
    assert pipeline.doc["tts_provider"] == "soniox"


@pytest.mark.asyncio
async def test_settings_keep_soniox_off_until_the_key_is_set():
    from httpx import ASGITransport, AsyncClient

    from tests.church import make_app, sign_in

    app = await make_app(["Hello"])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        current = await client.get("/api/settings")
        denied = await client.put(
            "/api/settings",
            json={"speaker_language": "en", "tts_enabled": False, "stt_provider": "soniox", "tts_provider": "soniox"},
        )
        unknown = await client.put(
            "/api/settings",
            json={"speaker_language": "en", "tts_enabled": False, "stt_provider": "whisper"},
        )
        page = await client.get("/settings")
    body = current.json()
    assert body["stt_provider"] == "default"
    assert body["tts_provider"] == "google"
    assert body["providers"]["stt"][0]["label"] == "Standard"
    assert body["providers"]["stt"][1]["available"] is False
    assert body["providers"]["tts"][1]["available"] is False
    assert denied.status_code == 400
    assert "API key" in denied.json()["detail"]
    assert unknown.status_code == 400
    assert "Speech recognition" in page.text
    assert "Spoken voice" in page.text
    assert "Soniox needs an API key on this server." in page.text


@pytest.mark.asyncio
async def test_settings_save_soniox_for_the_next_service():
    from httpx import ASGITransport, AsyncClient

    from tests.church import make_app, sign_in

    app = await make_app(["Hello"], extra_settings={"soniox_api_key": "test-key"})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        saved = await client.put(
            "/api/settings",
            json={
                "speaker_language": "en",
                "tts_enabled": False,
                "stt_provider": "soniox",
                "tts_provider": "soniox",
            },
        )
        again = await client.get("/api/settings")
        created = await client.post("/api/sessions", json={"speaker_language": "en", "name": "Sunday"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["stt_provider"] == "soniox"
    assert saved.json()["tts_provider"] == "soniox"
    assert saved.json()["providers"]["stt"][1]["available"] is True
    assert again.json()["stt_provider"] == "soniox"
    pipeline = app.state.director.pipeline(created.json()["id"])
    assert pipeline.doc["stt_provider"] == "soniox"
    assert pipeline.doc["tts_provider"] == "soniox"
    assert pipeline.recognizer is app.state.recognizers["soniox"]
    assert pipeline.speaker is app.state.speakers["soniox"]
    assert type(pipeline.recognizer).__name__ == "SonioxRecognizer"
    assert type(pipeline.speaker).__name__ == "SonioxSpeaker"
    switched = await client.put(
        "/api/settings",
        json={
            "speaker_language": "en",
            "tts_enabled": False,
            "stt_provider": "default",
            "tts_provider": "google",
        },
    )
    assert switched.status_code == 200
    assert pipeline.doc["stt_provider"] == "default"
    assert pipeline.doc["tts_provider"] == "google"
    assert pipeline.recognizer is app.state.recognizers["default"]
    assert pipeline.speaker is app.state.director._speaker_for({"tts_provider": "google"})
