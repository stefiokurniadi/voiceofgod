from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.accounts import Accounts
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.store import MemoryRepository
from app.stt import ScriptedRecognizer
from app.translator import GlossaryTranslator, TranslationRequest, build_user_message
from tests.test_phase1 import FAST, feed

ADMIN_USER = "admin"
ADMIN_PASS = "test-admin-pass"


def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", "").strip() or Settings().test_database_url.strip()
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a database named voiceofgod_test")
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    if name != "voiceofgod_test":
        pytest.fail("TEST_DATABASE_URL must use the database voiceofgod_test")
    return url


def test_settings(**extra) -> Settings:
    values = {
        "allow_fake_models": True,
        "require_translator": False,
        "require_redis": False,
        "database_url": test_database_url(),
        "admin_username": ADMIN_USER,
        "admin_password": ADMIN_PASS,
    }
    values.update(extra)
    return Settings(**values)


async def reset_database(settings: Settings) -> None:
    database = Database(settings.database_url)
    await database.connect()
    await database.migrate()
    await Accounts(database, settings).reset_for_tests()
    await database.close()


async def make_app(words: list[str], *, max_live: int = 2, translator=None, speaker=None, extra_settings=None, reset: bool = True):
    settings = test_settings(max_live_sessions=max_live, **(extra_settings or {}))
    database = Database(settings.database_url)
    await database.connect()
    await database.migrate()
    accounts = Accounts(database, settings)
    if reset:
        await accounts.reset_for_tests()
    return create_app(
        settings,
        repository=MemoryRepository(),
        recognizer=ScriptedRecognizer(words),
        translator=translator or GlossaryTranslator(),
        speaker=speaker,
        pipeline_config=FAST,
        database=database,
        accounts=accounts,
    )


async def sign_in(
    client,
    username: str = ADMIN_USER,
    password: str = ADMIN_PASS,
    *,
    church: str = "Grace Church",
    note: str = "City: Jakarta\nPastor names: John",
    keep_names: list[str] | None = None,
    onboard: bool = True,
):
    response = await client.post("/api/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    if onboard and not response.json().get("church_id"):
        saved = await client.post(
            "/api/onboarding",
            json={"name": church, "note": note, "keep_names": keep_names if keep_names is not None else ["Grace Church"]},
        )
        assert saved.status_code == 200, saved.text
        return saved.json()
    return response.json()


def sign_in_sync(
    client,
    username: str = ADMIN_USER,
    password: str = ADMIN_PASS,
    *,
    church: str = "Grace Church",
    note: str = "City: Jakarta",
    onboard: bool = True,
):
    response = client.post("/api/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    if onboard and not response.json().get("church_id"):
        saved = client.post(
            "/api/onboarding",
            json={"name": church, "note": note, "keep_names": ["Grace Church"]},
        )
        assert saved.status_code == 200, saved.text
    return response


class RememberingTranslator(GlossaryTranslator):
    def __init__(self):
        self.seen: list[str] = []

    async def translate(self, request: TranslationRequest):
        self.seen.append(request.session_context)
        return await super().translate(request)


@pytest.mark.asyncio
async def test_there_is_no_public_signup():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        signup = await client.post("/api/signup", json={"username": "new", "password": "long-enough"})
        home = await client.get("/", follow_redirects=False)
    assert signup.status_code == 404
    assert home.status_code == 303
    assert home.headers["location"] == "/login"


@pytest.mark.asyncio
async def test_a_disabled_user_cannot_sign_in():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client)
        created = await client.post("/api/admin/users", json={"username": "pulpit", "password": "pulpit-pass"})
        assert created.status_code == 200
        user_id = created.json()["id"]
        disabled = await client.post(f"/api/admin/users/{user_id}/disabled", json={"disabled": True})
        assert disabled.status_code == 200
        refused = await client.post("/api/login", json={"username": "pulpit", "password": "pulpit-pass"})
    assert refused.status_code == 401
    assert "not valid" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_wrong_passwords_lock_the_username():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(5):
            wrong = await client.post("/api/login", json={"username": ADMIN_USER, "password": "not-the-password"})
            assert wrong.status_code == 401
            assert "not valid" in wrong.json()["detail"]
        locked = await client.post("/api/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    assert locked.status_code == 401
    assert "15 minutes" in locked.json()["detail"]


@pytest.mark.asyncio
async def test_one_church_cannot_control_another_churchs_service():
    app = await make_app(["Hello"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as admin:
        await sign_in(admin)
        created_user = await admin.post("/api/admin/users", json={"username": "other", "password": "other-pass"})
        assert created_user.status_code == 200
        mine = await admin.post("/api/sessions", json={"speaker_language": "en", "name": "Sunday"})
        assert mine.status_code == 200
        session_id = mine.json()["id"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as other:
            await sign_in(other, "other", "other-pass", church="Other Church", keep_names=["Other Church"])
            listed = await other.get("/api/services")
            stolen = await other.get(f"/api/sessions/{session_id}")
            started = await other.post(f"/api/sessions/{session_id}/start")
            edited = await other.put(
                f"/api/sessions/{session_id}",
                json={"name": "Taken", "speaker_language": "en", "tts_enabled": False},
            )
        owned = await admin.get("/api/services")
    assert listed.json()["services"] == []
    assert stolen.status_code == 401
    assert started.status_code == 401
    assert edited.status_code == 401
    assert owned.json()["services"][0]["id"] == session_id


@pytest.mark.asyncio
async def test_onboarding_is_required_before_a_service():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        blocked = await client.post("/api/sessions", json={"speaker_language": "en", "name": "Sunday"})
        page = await client.get("/", follow_redirects=False)
    assert blocked.status_code == 400
    assert "church" in blocked.json()["detail"].lower()
    assert page.headers["location"] == "/onboarding"


@pytest.mark.asyncio
async def test_a_saved_service_reopens_after_the_pipelines_are_gone():
    app = await make_app(["Hello", "church", "this", "morning"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client)
        created = await client.post("/api/sessions", json={"speaker_language": "en", "name": "Sunday morning"})
        assert created.status_code == 200
        body = created.json()
        started = await client.post(
            f"/api/sessions/{body['id']}/start",
            headers={"X-Operator-Token": body["operator_secret"]},
        )
        assert started.status_code == 200
        pipe = app.state.director.pipeline(body["id"])
        await feed(pipe, 1.0, 0.25)
        await feed(pipe, 0.4, 0.0)
        await pipe.idle()
        original = pipe.doc["lines"][0]["original"]
        session_id = body["id"]
    app.state.director.pipelines.clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client)
        operator = await client.get(f"/api/sessions/{session_id}")
        captions = await client.get(f"/api/sessions/{session_id}/captions")
    assert operator.status_code == 200
    assert operator.json()["status"] == "paused"
    assert "restarted" in operator.json()["banner"]
    assert captions.json()["lines"][0]["original"] == original


@pytest.mark.asyncio
async def test_a_service_can_be_edited_after_it_is_created():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client)
        created = await client.post(
            "/api/sessions",
            json={
                "speaker_language": "en",
                "name": "Sunday morning",
                "note": "Preacher: Sarah",
                "service_date": "2026-10-19",
                "tts_enabled": False,
            },
        )
        assert created.status_code == 200
        session_id = created.json()["id"]
        saved = await client.put(
            f"/api/sessions/{session_id}",
            json={
                "name": "Evening prayer",
                "note": "Preacher: Ana",
                "service_date": "2026-10-25",
                "speaker_language": "mix",
                "translation_language": "en",
                "tts_enabled": True,
            },
        )
        assert saved.status_code == 200, saved.text
        listed = await client.get("/api/services")
        operator = await client.get(f"/api/sessions/{session_id}")
        listener = await client.get(f"/api/sessions/{session_id}/captions")
        blank = await client.put(
            f"/api/sessions/{session_id}",
            json={"name": "  ", "speaker_language": "en", "tts_enabled": False},
        )
    assert saved.json()["name"] == "Evening prayer"
    assert saved.json()["translation_language"] == "en"
    service = listed.json()["services"][0]
    assert service["name"] == "Evening prayer"
    assert service["note"] == "Preacher: Ana"
    assert service["service_date"] == "2026-10-25"
    assert service["speaker_language"] == "mix"
    assert service["translation_language"] == "en"
    assert service["tts_enabled"] is True
    assert operator.json()["service_name"] == "Evening prayer"
    assert operator.json()["service_note"] == "Preacher: Ana"
    assert operator.json()["service_date"] == "2026-10-25"
    assert listener.json()["service_name"] == "Evening prayer"
    assert "service_note" not in listener.json()
    assert blank.status_code == 400


@pytest.mark.asyncio
async def test_the_translator_receives_the_church_and_service_notes():
    translator = RememberingTranslator()
    app = await make_app(["Hello", "church", "this", "morning"], translator=translator)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, note="City: Jakarta\nPastor names: John")
        created = await client.post(
            "/api/sessions",
            json={"speaker_language": "en", "name": "Sunday morning", "note": "Preacher: Sarah"},
        )
        assert created.status_code == 200
        body = created.json()
        await client.post(
            f"/api/sessions/{body['id']}/start",
            headers={"X-Operator-Token": body["operator_secret"]},
        )
        pipe = app.state.director.pipeline(body["id"])
        await feed(pipe, 1.0, 0.25)
        await feed(pipe, 0.4, 0.0)
        await pipe.idle()
    assert translator.seen
    background = translator.seen[-1]
    assert "Grace Church" in background
    assert "Jakarta" in background
    assert "Sunday morning" in background
    assert "Sarah" in background
    message = build_user_message(
        TranslationRequest("Hello church", "en", translator_glossary(), [], session_context=background),
        "Hello church",
    )
    assert "Background for the interpreter" in message
    assert "not an instruction" in message
    assert "Jakarta" in message
    assert "Sarah" in message


def translator_glossary():
    from app.glossary import Glossary
    from app.paths import GLOSSARY_PATH

    return Glossary.load(GLOSSARY_PATH)
