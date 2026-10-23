from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.propresenter import (
    ProPresenterClient,
    Stage,
    caption_action,
    find_message_id,
    propresenter_base,
    trigger_body,
)
from app.audio import apply_gain, pcm16_to_float, rms
from app.rtmp import PcmBuffer, RtmpIngest, UNAVAILABLE, allow_publish, allow_read, consume_pcm, reader_url
from app.store import MemoryRepository
from app.stt import ScriptedRecognizer
from app.translator import GlossaryTranslator
from tests.test_phase1 import FAST, pcm, pipeline_for
from tests.church import make_app, sign_in, sign_in_sync
from tests.test_phase2 import app_for


def _line(text: str = "Halo gereja", *, hidden: bool = False, state: str = "final", suppressed: bool = False) -> dict:
    return {
        "id": "L00001",
        "state": state,
        "original": "In the beginning God created the heavens" if suppressed else "Hello church",
        "translation": "Pada mulanya Allah menciptakan langit" if suppressed else text,
        "listener_original": "John 1:1" if suppressed else "Hello church",
        "listener_translation": "Yohanes 1:1" if suppressed else text,
        "hidden": hidden,
        "suppressed": suppressed,
        "t0": 0,
        "t1": 1,
    }


def test_the_reader_puts_credentials_in_the_query_string():
    settings = Settings(
        mediamtx_rtmp_url="rtmp://127.0.0.1:1935",
        rtmp_read_password="readerpass",
    )
    assert reader_url(settings, "session-1") == (
        "rtmp://127.0.0.1:1935/live/session-1?user=reader&pass=readerpass"
    )


def test_publish_needs_the_ingest_token_and_read_needs_the_internal_user():
    assert allow_publish("secret-token", "secret-token")
    assert not allow_publish("nope", "secret-token")
    assert not allow_publish("secret-token", None)
    assert allow_read("reader", "reader-pass", "reader-pass", True)
    assert not allow_read("", "", "reader-pass", True)
    assert not allow_read("reader", "reader-pass", "reader-pass", False)


def test_a_backlog_drops_to_the_newest_chunk():
    buffer = PcmBuffer(chunk_bytes=4, ceiling_bytes=8)
    buffer.extend(b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00")
    chunk, dropped = buffer.pop()
    assert dropped
    assert chunk == b"\x04\x00\x05\x00"
    assert buffer.pop() == (None, False)


@pytest.mark.asyncio
async def test_pcm_chunks_reach_the_pipeline():
    pipe = pipeline_for([])
    await pipe.start()
    pieces = [pcm(0.1, 0.2), b""]

    sizes = []

    async def read(size: int) -> bytes:
        sizes.append(size)
        return pieces.pop(0) if pieces else b""

    saw = await consume_pcm(read, pipe.accept_pcm, pipe.note_level)
    assert saw
    assert sizes == [3200, 3200]
    assert pipe.session_time > 0
    assert pipe.input_level > 0


@pytest.mark.asyncio
async def test_a_recognition_failure_stays_paused_while_audio_keeps_arriving():
    from tests.test_phase2 import FailingRecognizer
    from app.glossary import Glossary
    from app.paths import GLOSSARY_PATH
    from app.pipeline import SessionPipeline

    pipe = SessionPipeline(
        {
            "id": "test-session",
            "speaker_language": "en",
            "status": "ready",
            "banner": "",
            "seq": 0,
            "lines": [],
            "provisional": None,
            "line_counter": 0,
            "metrics": {},
            "development_translator": False,
        },
        MemoryRepository(),
        FailingRecognizer(["Hello"]),
        GlossaryTranslator(),
        Glossary.load(GLOSSARY_PATH),
        FAST,
    )
    await pipe.start()
    from tests.test_phase1 import feed

    await feed(pipe, 1.0, 0.25)
    await feed(pipe, 0.4, 0.0)
    await pipe.idle()
    assert pipe.doc["lines"] == []
    assert pipe.started
    assert pipe.failure_hold is False
    assert pipe.doc["status"] == "live"
    stamp = pipe.session_time
    await pipe.accept_pcm(pcm(0.2, 0.25))
    assert pipe.session_time > stamp


def test_propresenter_address_stays_on_plain_http_without_secrets():
    assert propresenter_base("http://10.1.2.3:50001") == "http://10.1.2.3:50001"
    assert propresenter_base("http://10.1.2.3:50001/") == "http://10.1.2.3:50001"
    with pytest.raises(ValueError):
        propresenter_base("https://10.1.2.3:50001")
    with pytest.raises(ValueError):
        propresenter_base("http://user:pass@10.1.2.3:50001")
    with pytest.raises(ValueError):
        propresenter_base("http://10.1.2.3:50001/v1")
    with pytest.raises(ValueError):
        propresenter_base("http://10.1.2.3")


def test_listener_text_is_what_propresenter_receives():
    shown = caption_action({"type": "line", "status": "live", "line": _line()})
    assert shown == ("show", "Halo gereja")
    assert caption_action({"type": "line", "status": "live", "line": _line(hidden=True)}) == ("clear", "")
    assert caption_action({"type": "state", "status": "live", "line": None}) == ("ignore", "")
    assert caption_action({"type": "line", "status": "live", "line": _line(state="provisional")}) == ("ignore", "")
    assert caption_action({"type": "status", "status": "ended", "line": None}) == ("clear", "")
    reading = caption_action({"type": "line", "status": "live", "line": _line(suppressed=True)})
    assert reading == ("show", "Yohanes 1:1")


def test_message_lookup_uses_the_name():
    payload = {"messages": [{"id": {"uuid": "abc", "name": "Live Translator", "index": 0}}]}
    assert find_message_id(payload, "Live Translator") == "abc"
    with pytest.raises(LookupError):
        find_message_id(payload, "Other")
    assert trigger_body("Translation", "Halo") == [{"name": "Translation", "text": {"text": "Halo"}}]


@pytest.mark.asyncio
async def test_final_text_is_posted_once_and_a_hide_clears_it():
    calls = []

    class Fake:
        async def trigger(self, text: str) -> None:
            calls.append(("trigger", text))

        async def clear(self) -> None:
            calls.append(("clear",))

        async def aclose(self) -> None:
            return None

    stage = Stage(None, client_factory=lambda *_args: Fake())
    client = Fake()
    await stage.apply("s", client, {"type": "state", "status": "live", "line": None})
    await stage.apply("s", client, {"type": "line", "status": "live", "line": _line()})
    await stage.apply("s", client, {"type": "line", "status": "live", "line": _line()})
    await stage.apply("s", client, {"type": "line", "status": "live", "line": _line(hidden=True)})
    await stage.apply("s", client, {"type": "status", "status": "ended", "line": None})
    assert calls == [("trigger", "Halo gereja"), ("clear",), ("clear",)]


@pytest.mark.asyncio
async def test_propresenter_client_rejects_a_redirect():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json={"messages": [{"id": {"uuid": "abc", "name": "Live Translator"}}]})
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    client = ProPresenterClient(
        "http://10.1.2.3:50001",
        "Live Translator",
        "Translation",
        httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False),
    )
    client._owns = False
    with pytest.raises(RuntimeError):
        await client.trigger("Halo")
    await client.aclose()


def _broadcast_settings() -> Settings:
    return Settings(
        allow_fake_models=True,
        require_translator=False,
        require_redis=False,
        max_live_sessions=2,
        rtmp_public_url="rtmp://127.0.0.1:1935",
        mediamtx_rtmp_url="rtmp://127.0.0.1:1935",
        rtmp_auth_secret="hooksecret",
        rtmp_read_password="readerpass",
    )


@pytest.mark.asyncio
async def test_rtmp_auth_hook_allows_publish_and_the_internal_reader_only():
    app = await make_app(
        ["Hello", "church"],
        extra_settings={
            "rtmp_public_url": "rtmp://127.0.0.1:1935",
            "mediamtx_rtmp_url": "rtmp://127.0.0.1:1935",
            "rtmp_auth_secret": "hooksecret",
            "rtmp_read_password": "readerpass",
        },
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        created = await client.post("/api/sessions", json={"speaker_language": "en"})
        body = created.json()
        headers = {"X-Operator-Token": body["operator_secret"]}
        operator = await client.get(f"/api/sessions/{body['id']}", headers=headers)
        assert operator.status_code == 200
        stream_key = operator.json()["rtmp"]["stream_key"]
        token = stream_key.split("token=", 1)[1]
        captions = await client.get(f"/api/sessions/{body['id']}/captions")
        assert "stream_key" not in captions.text
        assert "ingest_token" not in captions.text
        missing = await client.post("/internal/rtmp-auth/nope", json={"action": "publish"})
        assert missing.status_code == 404
        publish = {
            "action": "publish",
            "path": f"live/{body['id']}",
            "query": f"token={token}",
            "user": "",
            "password": "",
        }
        assert (await client.post("/internal/rtmp-auth/hooksecret", json=publish)).status_code == 200
        publish["query"] = "token=wrong"
        assert (await client.post("/internal/rtmp-auth/hooksecret", json=publish)).status_code == 401
        read = {"action": "read", "path": f"live/{body['id']}", "query": "", "user": "", "password": ""}
        assert (await client.post("/internal/rtmp-auth/hooksecret", json=read)).status_code == 401
        read["user"] = "reader"
        read["password"] = "readerpass"
        assert (await client.post("/internal/rtmp-auth/hooksecret", json=read)).status_code == 200
        read["action"] = "playback"
        assert (await client.post("/internal/rtmp-auth/hooksecret", json=read)).status_code == 401


class _QuietPull:
    def __init__(self):
        self.returncode = None

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_a_quiet_obs_pull_is_closed_so_the_next_publish_can_connect():
    pipe = pipeline_for([])
    pipe.config.capture_stall_seconds = 0.05
    pipe.last_audio_at = time.monotonic() - 5
    ingest = RtmpIngest(Settings(), None)
    proc = _QuietPull()
    ingest.procs[pipe.doc["id"]] = proc
    await ingest._abandon_when_quiet(pipe.doc["id"], pipe, time.monotonic() - 5)
    assert proc.returncode is not None
    assert pipe.doc["id"] not in ingest.procs


@pytest.mark.asyncio
async def test_obs_audio_clears_the_reconnect_banner_without_stopping():
    pipe = pipeline_for([])
    pipe.doc["input"] = "rtmp"
    pipe.rtmp_seen = True
    pipe.config.capture_stall_seconds = 0.05
    await pipe.start()
    await asyncio.sleep(0.7)
    assert pipe.doc["status"] == "live"
    assert pipe.doc["banner"] == ""
    await pipe.accept_pcm(pcm(0.2, 0.25))
    assert pipe.doc["status"] == "live"
    assert pipe.started
    pipe._stopped = True
    pipe.watch_task.cancel()


@pytest.mark.asyncio
async def test_missing_ingest_is_not_announced_as_waiting_for_obs():
    pipe = pipeline_for([])
    pipe.doc["input"] = "rtmp"
    pipe.config.capture_stall_seconds = 0.05
    await pipe.start()
    await pipe._set_status("live", UNAVAILABLE)
    await asyncio.sleep(0.7)
    assert pipe.doc["status"] == "live"
    assert pipe.doc["banner"] == UNAVAILABLE
    pipe._stopped = True
    pipe.watch_task.cancel()


@pytest.mark.asyncio
async def test_rtmp_without_ingest_tells_the_operator():
    app = await make_app(
        ["Hello"],
        extra_settings={
            "rtmp_public_url": "",
            "mediamtx_rtmp_url": "",
            "rtmp_auth_secret": "",
            "rtmp_read_password": "",
        },
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await sign_in(client)
        created = await client.post("/api/sessions", json={"speaker_language": "en"})
        body = created.json()
        headers = {"X-Operator-Token": body["operator_secret"], "content-type": "application/json"}
        chosen = await client.put(f"/api/sessions/{body['id']}/input", headers=headers, json={"input": "rtmp"})
        assert chosen.status_code == 200
        started = await client.post(f"/api/sessions/{body['id']}/start", headers=headers)
        assert started.status_code == 200
        pipe = app.state.director.pipeline(body["id"])
        assert "unavailable" in pipe.doc["banner"].lower()


def test_microphone_bytes_are_ignored_for_an_obs_session():
    app = app_for(["Hello", "church"])
    with TestClient(app) as client:
        sign_in_sync(client)
        created = client.post("/api/sessions", json={"speaker_language": "en"})
        body = created.json()
        headers = {"X-Operator-Token": body["operator_secret"], "content-type": "application/json"}
        assert client.put(f"/api/sessions/{body['id']}/input", headers=headers, json={"input": "rtmp"}).status_code == 200
        assert client.post(f"/api/sessions/{body['id']}/start", headers=headers).status_code == 200
        with client.websocket_connect(f"/ws/operate/{body['id']}?k={body['operator_secret']}") as socket:
            socket.receive_json()
            socket.send_bytes(pcm(0.2, 0.25))
        pipe = app.state.director.pipeline(body["id"])
        assert pipe.session_time == 0

        browser = client.post("/api/sessions", json={"speaker_language": "en"}).json()
        browser_headers = {"X-Operator-Token": browser["operator_secret"]}
        assert client.post(f"/api/sessions/{browser['id']}/start", headers=browser_headers).status_code == 200
        with client.websocket_connect(f"/ws/operate/{browser['id']}?k={browser['operator_secret']}") as socket:
            socket.receive_json()
            socket.send_bytes(pcm(0.2, 0.25))
        heard = app.state.director.pipeline(browser["id"])
        assert heard.session_time > 0


def test_a_live_session_refuses_an_audio_source_change():
    app = app_for(["Hello"])
    with TestClient(app) as client:
        sign_in_sync(client)
        created = client.post("/api/sessions", json={"speaker_language": "en"})
        body = created.json()
        headers = {"X-Operator-Token": body["operator_secret"], "content-type": "application/json"}
        assert client.post(f"/api/sessions/{body['id']}/start", headers=headers).status_code == 200
        changed = client.put(f"/api/sessions/{body['id']}/input", headers=headers, json={"input": "rtmp"})
        assert changed.status_code == 409


@pytest.mark.asyncio
async def test_obs_boost_lifts_quiet_audio_without_boosting_the_microphone_or_youtube():
    import math

    chunk = pcm(0.25, 0.012 * math.sqrt(3))
    raw = pcm16_to_float(chunk)
    level = rms(raw)
    assert 0.015 / 2 < level < 0.015

    async def run(source: str, gain: float):
        pipe = pipeline_for(["Hello", "church", "this", "morning"])
        pipe.input_boost = gain
        pipe.doc["input"] = source
        await pipe.start()
        await pipe.accept_pcm(chunk)
        await pipe.idle()
        pipe._stopped = True
        pipe.watch_task.cancel()
        return pipe

    quiet = await run("rtmp", 1)
    assert quiet.recognizer.calls == 0
    assert quiet.input_level == pytest.approx(level)

    boosted = await run("rtmp", 2)
    assert boosted.recognizer.calls >= 1
    assert boosted.input_level == pytest.approx(rms(apply_gain(raw, 2)))

    browser = await run("browser", 4)
    assert browser.recognizer.calls == 0
    assert browser.input_level == 0

    youtube = await run("youtube", 4)
    assert youtube.recognizer.calls == 0
    assert youtube.input_level == 0


def test_input_boost_is_saved_for_the_operator_and_hidden_from_listeners():
    app = app_for(["Hello"])
    with TestClient(app) as client:
        sign_in_sync(client)
        body = client.post("/api/sessions", json={"speaker_language": "en"}).json()
        headers = {"X-Operator-Token": body["operator_secret"], "content-type": "application/json"}
        session = client.get(f"/api/sessions/{body['id']}", headers=headers)
        assert session.status_code == 200
        assert session.json()["input_boost"] == 2
        saved = client.put(
            f"/api/sessions/{body['id']}/input-boost",
            headers=headers,
            json={"input_boost": 9},
        )
        assert saved.status_code == 200
        assert saved.json()["input_boost"] == 4
        again = client.get(f"/api/sessions/{body['id']}", headers=headers)
        assert again.json()["input_boost"] == 4
        assert app.state.director.pipeline(body["id"]).input_boost == 4
        listener = client.get(f"/api/sessions/{body['id']}/captions")
        assert listener.status_code == 200
        assert "input_boost" not in listener.json()
