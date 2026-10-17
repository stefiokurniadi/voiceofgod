from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.captions import lines_after
from app.config import Settings
from app.export import render_srt, render_txt
from app.glossary import Glossary
from app.main import create_app
from app.store import MemoryRepository
from app.stt import ScriptedRecognizer, WhisperRecognizer
from app.translator import GlossaryTranslator
from tests.test_phase1 import FAST, feed


def test_lines_after_an_acknowledged_caption():
    lines = [{"id": "L00001"}, {"id": "L00002"}]
    assert lines_after(lines, None) == lines
    assert lines_after(lines, "L00001") == [{"id": "L00002"}]
    assert lines_after(lines, "missing") is None


def test_export_skips_hidden_lines_and_formats_cues():
    lines = [
        {"id": "L00001", "state": "final", "hidden": False, "t0": 1.25, "t1": 3.5, "original": "Hello church", "translation": "Halo jemaat"},
        {"id": "L00002", "state": "final", "hidden": True, "t0": 4, "t1": 5, "original": "secret", "translation": "rahasia"},
        {"id": "L00003", "state": "provisional", "hidden": False, "t0": 6, "t1": 6, "original": "still", "translation": "masih"},
    ]
    text = render_txt(lines, "en")
    assert "Hello church" in text
    assert "Halo jemaat" in text
    assert "secret" not in text
    assert "still" not in text
    srt = render_srt(lines)
    assert "00:00:01,250 --> 00:00:03,500" in srt
    assert "secret" not in srt
    held = render_srt([{"state": "final", "hidden": False, "t0": 2, "t1": 2, "original": "Amen", "translation": "Amin"}])
    assert "00:00:02,000 --> 00:00:03,000" in held


def test_public_glossary_keeps_aliases_and_rejects_a_long_phrase():
    glossary = Glossary.from_public(
        [{"en": "worship service", "id": "kebaktian", "id_aliases": ["ibadah"]}],
        ["Grace Church"],
    )
    assert glossary.terms[0].id_aliases == ("ibadah",)
    with pytest.raises(ValueError):
        Glossary.from_public([{"en": "a" * 81, "id": "b"}], [])


def test_whisper_calls_are_serialized_on_one_model():
    recognizer = WhisperRecognizer("large-v3-turbo", "cpu", "int8")
    assert recognizer._lock is not None


def app_for(words: list[str], max_live: int = 2):
    return create_app(
        Settings(allow_fake_models=True, require_translator=False, require_redis=False, max_live_sessions=max_live),
        repository=MemoryRepository(),
        recognizer=ScriptedRecognizer(words),
        translator=GlossaryTranslator(),
        pipeline_config=FAST,
    )


async def open_session(client: AsyncClient, language: str = "en") -> dict:
    created = await client.post("/api/sessions", json={"speaker_language": language})
    assert created.status_code == 200
    body = created.json()
    started = await client.post(
        f"/api/sessions/{body['id']}/start",
        headers={"X-Operator-Token": body["operator_secret"]},
    )
    assert started.status_code == 200
    return body


@pytest.mark.asyncio
async def test_two_sessions_share_the_server_and_a_third_is_refused():
    app = app_for(["Hello", "church", "this", "morning"], max_live=2)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await open_session(client)
        second = await open_session(client)
        third = await client.post("/api/sessions", json={"speaker_language": "en"})
        refused = await client.post(
            f"/api/sessions/{third.json()['id']}/start",
            headers={"X-Operator-Token": third.json()["operator_secret"]},
        )
        assert refused.status_code == 409
        assert "2" in refused.json()["detail"]
        assert first["id"] != second["id"]


@pytest.mark.asyncio
async def test_glossary_edit_changes_the_next_utterance_only_for_that_session():
    app = app_for(["full", "of", "grace"])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await open_session(client)
        second = await open_session(client)
        headers = {"X-Operator-Token": first["operator_secret"], "content-type": "application/json"}
        saved = await client.put(
            f"/api/sessions/{first['id']}/glossary",
            headers=headers,
            json={"terms": [{"en": "grace", "id": "kasih"}], "do_not_translate": ["Grace Church"]},
        )
        assert saved.status_code == 200
        too_long = await client.put(
            f"/api/sessions/{first['id']}/glossary",
            headers=headers,
            json={"terms": [{"en": "a" * 81, "id": "b"}], "do_not_translate": []},
        )
        assert too_long.status_code == 400
        for session_id in (first["id"], second["id"]):
            pipe = app.state.director.pipeline(session_id)
            await feed(pipe, 1.0, 0.25)
            await feed(pipe, 0.4, 0.0)
            await pipe.idle()
        first_line = app.state.director.pipeline(first["id"]).doc["lines"][0]["translation"]
        second_line = app.state.director.pipeline(second["id"]).doc["lines"][0]["translation"]
        assert "kasih" in first_line
        assert "anugerah" in second_line


@pytest.mark.asyncio
async def test_resume_export_overlay_and_metrics():
    app = app_for(["Hello", "church", "this", "morning"], max_live=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session = await open_session(client)
        headers = {"X-Operator-Token": session["operator_secret"]}
        pipe = app.state.director.pipeline(session["id"])
        await feed(pipe, 1.0, 0.25)
        await feed(pipe, 0.4, 0.0)
        await pipe.idle()
        captions = await client.get(f"/api/sessions/{session['id']}/captions")
        line_id = captions.json()["lines"][0]["id"]
        resumed = await client.get(f"/api/sessions/{session['id']}/captions", params={"after": line_id})
        assert resumed.json()["type"] == "resume"
        assert resumed.json()["lines"] == []
        unknown = await client.get(f"/api/sessions/{session['id']}/captions", params={"after": "nope"})
        assert unknown.json()["type"] == "snapshot"
        assert unknown.json()["lines"]
        hidden = await client.post(f"/api/sessions/{session['id']}/lines/{line_id}/hide", headers=headers)
        assert hidden.status_code == 200
        exported = await client.get(f"/api/sessions/{session['id']}/export.txt", params={"k": session["operator_secret"]})
        assert exported.status_code == 200
        assert b"Hello" not in exported.content
        srt = await client.get(f"/api/sessions/{session['id']}/export.srt", headers=headers)
        assert srt.status_code == 200
        overlay = await client.get(f"/v/{session['id']}")
        assert overlay.status_code == 200
        assert b'class="overlay"' in overlay.content
        metrics = await client.get(f"/api/sessions/{session['id']}/metrics", headers=headers)
        assert metrics.status_code == 200
        assert "listeners" in metrics.json()
        assert "gpu" in metrics.json()
        operator = await client.get(f"/api/sessions/{session['id']}", headers=headers)
        assert operator.json()["glossary"]["terms"]
