from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.captions import lines_after
from app.config import Settings
from app.director import Director
from app.export import render_srt, render_txt
from app.glossary import Glossary
from app.main import create_app
from app.paths import GLOSSARY_PATH
from app.pipeline import SessionPipeline
from app.store import MemoryRepository
from app.stt import ScriptedRecognizer, WhisperRecognizer
from app.translator import GlossaryTranslator
from tests.test_phase1 import FAST, document, feed


def test_a_missing_session_is_not_reported_as_a_bad_token():
    from starlette.testclient import TestClient

    from tests.church import sign_in_sync

    app = app_for(["Hello", "church"])
    with TestClient(app) as client:
        missing = client.post("/api/sessions/missing/start", headers={"X-Operator-Token": "nope"})
        assert missing.status_code == 404
        assert "home page" in missing.json()["detail"]
        sign_in_sync(client)
        created = client.post("/api/sessions", json={"speaker_language": "en"}).json()
        client.post("/api/logout")
        wrong = client.post(
            f"/api/sessions/{created['id']}/start",
            headers={"X-Operator-Token": "wrong"},
        )
        assert wrong.status_code == 401


def test_a_missing_session_tells_the_page_to_stop_reconnecting():
    from starlette.testclient import TestClient

    from tests.church import sign_in_sync

    app = app_for(["Hello", "church"])
    with TestClient(app) as client:
        with client.websocket_connect("/ws/listen/missing") as socket:
            assert socket.receive_json()["type"] == "deleted"
        sign_in_sync(client)
        created = client.post("/api/sessions", json={"speaker_language": "en"}).json()
        client.post("/api/logout")
        with client.websocket_connect(f"/ws/operate/{created['id']}?k=wrong") as socket:
            assert socket.receive_json()["type"] == "unauthorized"
        with client.websocket_connect("/ws/operate/missing?k=nope") as socket:
            assert socket.receive_json()["type"] == "deleted"


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
    from tests.church import reset_database, test_settings

    settings = test_settings(max_live_sessions=max_live)
    asyncio.run(reset_database(settings))
    return create_app(
        settings,
        repository=MemoryRepository(),
        recognizer=ScriptedRecognizer(words),
        translator=GlossaryTranslator(),
        pipeline_config=FAST,
    )


async def open_session(client: AsyncClient, language: str = "en") -> dict:
    from tests.church import sign_in

    if "voiceofgod_session" not in client.cookies:
        await sign_in(client)
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
    from tests.church import make_app

    app = await make_app(["Hello", "church", "this", "morning"], max_live=2)
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
    from tests.church import make_app

    app = await make_app(["full", "of", "grace"])
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
    from tests.church import make_app

    app = await make_app(["Hello", "church", "this", "morning"], max_live=1)
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


def test_creating_a_service_requires_a_login():
    from starlette.testclient import TestClient

    from tests.church import sign_in_sync

    app = app_for(["Hello", "church"])
    with TestClient(app) as client:
        denied = client.post("/api/sessions", json={"speaker_language": "en"})
        assert denied.status_code == 401
        sign_in_sync(client)
        created = client.post("/api/sessions", json={"speaker_language": "en"})
        assert created.status_code == 200
        missing = client.get("/l/missing", headers={"accept": "text/html"})
        assert missing.status_code == 404
    assert b"Home" in missing.content


@pytest.mark.asyncio
async def test_restart_keeps_the_transcript_and_pauses_a_live_session():
    repository = MemoryRepository()
    first = Director(
        Settings(allow_fake_models=True),
        repository,
        ScriptedRecognizer(["Hello", "church", "this", "morning"]),
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    user = {
        "church_id": "church",
        "church_name": "Grace Church",
        "church_note": "",
        "keep_names": [],
    }
    created = await first.create_session(user, "Sunday", "", None, "en", False, "http://test")
    await first.start(created["id"])
    pipe = first.pipeline(created["id"])
    await feed(pipe, 1.0, 0.25)
    await feed(pipe, 0.4, 0.0)
    await pipe.idle()
    assert pipe.doc["lines"]
    second = Director(
        Settings(allow_fake_models=True),
        repository,
        ScriptedRecognizer(["Hello", "church", "this", "morning"]),
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    assert await second.restore() == 1
    restored = second.pipeline(created["id"])
    assert restored.doc["status"] == "paused"
    assert restored.doc["lines"][0]["original"]
    assert "restarted" in restored.doc["banner"]


@pytest.mark.asyncio
async def test_five_hundred_listeners_share_one_event():
    repository = MemoryRepository()
    subscriptions = [repository.subscribe("sermon") for _ in range(500)]
    entered = [await subscription.__aenter__() for subscription in subscriptions]
    await repository.publish("sermon", {"type": "line", "seq": 1, "line": {"id": "L00001"}})
    for subscription in entered:
        event = await asyncio.wait_for(subscription.__anext__(), timeout=2)
        assert event["seq"] == 1
        assert event["line"]["id"] == "L00001"
    for subscription in subscriptions:
        await subscription.__aexit__(None, None, None)


class FailingRecognizer(ScriptedRecognizer):
    def transcribe(self, audio, language: str, hotwords: str | None = None):
        raise RuntimeError("cuda out of memory")


@pytest.mark.asyncio
async def test_a_recognition_failure_pauses_without_inventing_a_caption():
    pipe = SessionPipeline(
        document(),
        MemoryRepository(),
        FailingRecognizer(["Hello"]),
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    await pipe.start()
    await feed(pipe, 1.0, 0.25)
    await feed(pipe, 0.4, 0.0)
    await pipe.idle()
    assert pipe.doc["lines"] == []
    assert pipe.doc["status"] == "paused"
    assert "Speech recognition failed" in pipe.doc["banner"]
    assert pipe.started is False
