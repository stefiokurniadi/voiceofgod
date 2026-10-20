from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.glossary import Glossary
from app.paths import GLOSSARY_PATH
from app.pipeline import finish_translation
from app.stt import GoogleRecognizer
from app.translator import language_name, target_language


def test_mixed_speech_keeps_the_english_and_indonesian_pair():
    assert target_language("en") == "id"
    assert target_language("id") == "en"
    assert target_language("id", "id") == "en"
    assert target_language("mix") == "id"
    assert target_language("mix", "en") == "en"
    assert language_name("mix") == "mixed English and Bahasa Indonesia"
    kept = finish_translation(
        "The grace of God menyertai kamu",
        "The grace of God is with you today.",
        target="en",
    )
    assert kept.casefold() == "the grace of god is with you today."


def test_a_mixed_sentence_uses_the_chosen_glossary_side():
    glossary = Glossary.load(GLOSSARY_PATH)
    toward_indonesian = glossary.protect("the Holy Spirit is with the congregation", "mix", "id")
    indonesian, _missing = glossary.restore(toward_indonesian.text, toward_indonesian)
    assert "Roh Kudus" in indonesian
    assert "jemaat" in indonesian
    assert "Holy Spirit" not in indonesian
    toward_english = glossary.protect("Roh Kudus menyertai jemaat", "mix", "en")
    english, _missing = glossary.restore(toward_english.text, toward_english)
    assert "Holy Spirit" in english
    assert "congregation" in english
    hints = glossary.hotwords("mix")
    assert "Holy Spirit" in hints
    assert "Roh Kudus" in hints


def test_chirp_can_hear_either_language_in_a_mixed_clip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    chirp = GoogleRecognizer(
        "key",
        "church-project",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_provider=lambda: "ya29.test",
    )
    chirp.transcribe(np.full(1600, 0.2, dtype=np.float32), "mix", "Holy Spirit")
    assert seen["body"]["config"]["languageCodes"] == ["id-ID", "en-US"]
    assert seen["body"]["config"]["model"] == "chirp_3"

    fallback = GoogleRecognizer("key", "church-project", client=httpx.Client(transport=httpx.MockTransport(handler)))
    fallback.transcribe(np.full(1600, 0.2, dtype=np.float32), "mix")
    assert seen["body"]["config"]["languageCode"] == "id-ID"
    assert seen["body"]["config"]["alternativeLanguageCodes"] == ["en-US"]
    assert seen["body"]["config"]["model"] == "latest_long"


@pytest.mark.asyncio
async def test_a_service_can_translate_mixed_speech_either_way():
    from tests.church import make_app, sign_in

    app = await make_app([])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        rejected = await client.post("/api/sessions", json={"speaker_language": "fr"})
        english = await client.post("/api/sessions", json={"speaker_language": "en", "translation_language": "en"})
        created = await client.post("/api/sessions", json={"speaker_language": "mix", "translation_language": "en"})
        session_id = created.json()["id"]
        captions = await client.get(f"/api/sessions/{session_id}/captions")
        changed = await client.post(
            f"/api/sessions/{session_id}/language",
            json={"speaker_language": "mix", "translation_language": "id"},
        )
        again = await client.get(f"/api/sessions/{session_id}/captions")
        listed = await client.get("/api/services")
    assert rejected.status_code == 400
    assert english.status_code == 200
    assert english.json()["translation_language"] == "id"
    assert created.status_code == 200
    assert created.json()["speaker_language"] == "mix"
    assert created.json()["translation_language"] == "en"
    assert captions.json()["translation_language"] == "en"
    assert changed.status_code == 200
    assert again.json()["speaker_language"] == "mix"
    assert again.json()["translation_language"] == "id"
    row = next(item for item in listed.json()["services"] if item["id"] == session_id)
    assert row["speaker_language"] == "mix"
    assert row["translation_language"] == "id"
