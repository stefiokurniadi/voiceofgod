from __future__ import annotations

import asyncio
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.youtube import ENDED, FAILED, LOADING, UNAVAILABLE, WAITING, Media, YouTubeIngest, check_youtube_url, ffmpeg_command, parse_media
from tests.church import make_app, sign_in, sign_in_sync
from tests.test_phase1 import pcm, pipeline_for
from tests.test_phase2 import app_for


WATCH = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_youtube_links_are_limited_to_a_video():
    assert check_youtube_url(WATCH) == WATCH
    assert check_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert check_youtube_url("https://m.youtube.com/shorts/dQw4w9WgXcQ")
    assert check_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ")
    with pytest.raises(ValueError):
        check_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with pytest.raises(ValueError):
        check_youtube_url("https://example.com/watch?v=dQw4w9WgXcQ")
    with pytest.raises(ValueError):
        check_youtube_url("https://www.youtube.com/")
    with pytest.raises(ValueError):
        check_youtube_url("https://user:pass@www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_microphone_bytes_are_ignored_for_a_youtube_test(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: False)
    app = app_for([])
    with TestClient(app) as client:
        sign_in_sync(client, onboard=False)
        created = client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        assert created.status_code == 200, created.text
        body = created.json()
        with client.websocket_connect(f"/ws/operate/{body['id']}?k={_secret(body['operator_url'])}") as socket:
            socket.receive_json()
            socket.send_bytes(pcm(0.2, 0.25))
        pipe = app.state.director.pipeline(body["id"])
        assert pipe.doc["input"] == "youtube"
        assert pipe.session_time == 0


def test_a_recorded_video_is_read_in_real_time():
    recorded = ffmpeg_command("https://example.test/audio", live=False)
    live = ffmpeg_command("https://example.test/audio", live=True)
    jumped = ffmpeg_command("https://example.test/audio", live=False, offset=12)
    assert recorded[recorded.index("-i") - 1] == "-re"
    assert "-re" not in live
    assert "-ss" not in recorded
    assert jumped.index("-ss") < jumped.index("-i")
    assert jumped[jumped.index("-ss") + 1] == "12.000"
    assert recorded[recorded.index("-ar") + 1] == "16000"
    reconnect = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    assert recorded[recorded.index("-reconnect"):recorded.index("-reconnect") + 6] == reconnect
    assert recorded.index("-reconnect") < recorded.index("-re")
    assert jumped.index("-reconnect_delay_max") < jumped.index("-ss")
    assert live.index("-reconnect") < live.index("-i")


def test_media_json_keeps_an_https_audio_address():
    media = parse_media(b'{"url":"https://rr.example/audio","is_live":false}')
    assert media == Media("https://rr.example/audio", False)
    chosen = parse_media(b'{"url":"","is_live":true,"formats":[{"url":"https://rr.example/live"}]}')
    assert chosen.live is True
    assert chosen.url == "https://rr.example/live"
    with pytest.raises(RuntimeError):
        parse_media(b'{"url":"file:///tmp/sermon.wav"}')


class _Stderr:
    async def readline(self):
        return b""


class _Proc:
    def __init__(self, chunks, hang=False):
        self.chunks = list(chunks)
        self.hang = hang
        self.returncode = None
        self.stdout = self
        self.stderr = _Stderr()
        self.terminated = False
        self.started = asyncio.Event()

    async def read(self, _size):
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        if not self.chunks:
            self.returncode = 0
            return b""
        return self.chunks.pop(0)

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9

    async def wait(self):
        return 0 if self.returncode is None else self.returncode


def _secret(operator_url: str) -> str:
    return parse_qs(urlsplit(operator_url).query)["k"][0]


@pytest.mark.asyncio
async def test_only_an_admin_can_start_a_youtube_test():
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as admin:
        anon = await admin.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        assert anon.status_code == 401
        await sign_in(admin, onboard=False)
        created = await admin.post("/api/admin/users", json={"username": "pulpit", "password": "pulpit-pass"})
        assert created.status_code == 200
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as operator:
            await sign_in(operator, "pulpit", "pulpit-pass")
            refused = await operator.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
    assert refused.status_code == 403


@pytest.mark.asyncio
async def test_a_youtube_test_is_not_a_church_service(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: False)
    app = await make_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        rejected = await client.post("/api/admin/youtube", json={"url": "https://example.com/watch?v=1", "speaker_language": "en"})
        assert rejected.status_code == 400
        created = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        assert created.status_code == 200, created.text
        body = created.json()
        assert "/o/" in body["operator_url"]
        pipe = app.state.director.pipeline(body["id"])
        assert pipe.doc["input"] == "youtube"
        assert pipe.doc["youtube_url"] == WATCH
        assert pipe.doc["church_id"] == ""
        assert pipe.doc["banner"] == UNAVAILABLE
        assert await app.state.accounts.load_service(body["id"]) is None
        again = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "id"})
        assert again.status_code == 409
        headers = {"X-Operator-Token": _secret(body["operator_url"]), "content-type": "application/json"}
        changed = await client.put(f"/api/sessions/{body['id']}/input", headers=headers, json={"input": "browser"})
        assert changed.status_code == 409
        stopped = await client.post("/api/admin/youtube/stop")
        assert stopped.status_code == 200
        assert app.state.director.active_youtube() is None


@pytest.mark.asyncio
async def test_youtube_audio_reaches_the_recognizer(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: True)
    app = await make_app(["Hello", "church", "this", "morning"])
    audio = pcm(1.0, 0.2) + pcm(0.4, 0.0)
    chunks = [audio[index:index + 3200] for index in range(0, len(audio), 3200)]
    seen = {}

    async def resolve(_url):
        return Media("https://rr.example/audio", False)

    async def spawn(command):
        seen["command"] = command
        return _Proc(chunks)

    app.state.director.youtube.resolve = resolve
    app.state.director.youtube.spawn = spawn
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        created = await client.post("/api/admin/youtube", json={"url": "https://youtu.be/dQw4w9WgXcQ", "speaker_language": "en"})
        assert created.status_code == 200, created.text
        body = created.json()
    pipe = app.state.director.pipeline(body["id"])
    await pipe.idle()
    assert "-re" in seen["command"]
    assert pipe.session_time > 0
    assert app.state.recognizer.calls > 0
    assert pipe.doc["lines"]


@pytest.mark.asyncio
async def test_stopping_a_youtube_test_kills_the_pull(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: True)
    app = await make_app([])
    proc = _Proc([], hang=True)

    async def resolve(_url):
        return Media("https://rr.example/live", True)

    async def spawn(_command):
        return proc

    app.state.director.youtube.resolve = resolve
    app.state.director.youtube.spawn = spawn
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        created = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        assert created.status_code == 200, created.text
        await asyncio.wait_for(proc.started.wait(), timeout=2)
        stopped = await client.post("/api/admin/youtube/stop")
        assert stopped.status_code == 200
    assert proc.terminated
    assert app.state.director.active_youtube() is None


@pytest.mark.asyncio
async def test_a_finished_video_can_be_started_again(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: True)
    app = await make_app(["Amen"])
    pulls = 0

    async def resolve(_url):
        return Media("https://rr.example/audio", False)

    async def spawn(_command):
        nonlocal pulls
        pulls += 1
        return _Proc([pcm(0.2, 0.0)])

    app.state.director.youtube.resolve = resolve
    app.state.director.youtube.spawn = spawn
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        created = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        body = created.json()
        pipe = app.state.director.pipeline(body["id"])
        for _ in range(30):
            if pipe.doc.get("banner") in {ENDED, FAILED} or not app.state.director.youtube.running(body["id"]):
                if pipe.doc.get("status") == "paused":
                    break
            await asyncio.sleep(0.05)
        assert pipe.doc["status"] == "paused"
        headers = {"X-Operator-Token": _secret(body["operator_url"])}
        again = await client.post(f"/api/sessions/{body['id']}/start", headers=headers)
        assert again.status_code == 200
    assert pulls == 2


@pytest.mark.asyncio
async def test_seek_restarts_playback_at_that_time(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: True)
    app = await make_app([])
    commands = []
    proc = _Proc([], hang=True)

    async def resolve(_url):
        return Media("https://rr.example/audio", False, 90)

    async def spawn(command):
        commands.append(list(command))
        return proc if len(commands) == 1 else _Proc([pcm(0.1, 0.0)])

    app.state.director.youtube.resolve = resolve
    app.state.director.youtube.spawn = spawn
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        created = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        body = created.json()
        await asyncio.wait_for(proc.started.wait(), timeout=2)
        headers = {"X-Operator-Token": _secret(body["operator_url"]), "content-type": "application/json"}
        too_far = await client.post(f"/api/sessions/{body['id']}/youtube/seek", headers=headers, json={"seconds": 120})
        assert too_far.status_code == 400
        moved = await client.post(f"/api/sessions/{body['id']}/youtube/seek", headers=headers, json={"seconds": 12})
        assert moved.status_code == 200, moved.text
        assert moved.json()["position"] == 12
        for _ in range(50):
            if len(commands) >= 2:
                break
            await asyncio.sleep(0.05)
    assert commands[-1].index("-ss") < commands[-1].index("-i")
    assert "12.000" in commands[-1]
    pipe = app.state.director.pipeline(body["id"])
    pipe.doc["youtube_live"] = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        headers = {"X-Operator-Token": _secret(body["operator_url"]), "content-type": "application/json"}
        refused = await client.post(f"/api/sessions/{body['id']}/youtube/seek", headers=headers, json={"seconds": 3})
    assert refused.status_code == 400


@pytest.mark.asyncio
async def test_youtube_audio_is_offered_for_playback():
    pipe = pipeline_for(["Hello"])
    pipe.doc["input"] = "youtube"
    await pipe.start()
    queue = pipe.attach_monitor()
    chunk = pcm(0.2, 0.2)
    await pipe.accept_pcm(chunk)
    heard = await asyncio.wait_for(queue.get(), timeout=1)
    assert heard == chunk
    await pipe.drop_partial()
    assert queue.empty()
    pipe.detach_monitor(queue)
    quiet = pipeline_for(["Hello"])
    await quiet.start()
    other = quiet.attach_monitor()
    await quiet.accept_pcm(chunk)
    assert other.empty()


@pytest.mark.asyncio
async def test_pause_holds_the_place_and_start_continues(monkeypatch):
    monkeypatch.setattr("app.director.youtube_ready", lambda: True)
    app = await make_app([])
    proc = _Proc([], hang=True)
    commands = []

    async def resolve(_url):
        return Media("https://rr.example/audio", False, 90)

    async def spawn(command):
        commands.append(list(command))
        return proc if len(commands) == 1 else _Proc([pcm(0.1, 0.0)])

    app.state.director.youtube.resolve = resolve
    app.state.director.youtube.spawn = spawn
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await sign_in(client, onboard=False)
        missing = await client.post("/api/admin/youtube/pause")
        assert missing.status_code == 404
        created = await client.post("/api/admin/youtube", json={"url": WATCH, "speaker_language": "en"})
        assert created.status_code == 200, created.text
        body = created.json()
        await asyncio.wait_for(proc.started.wait(), timeout=2)
        pipe = app.state.director.pipeline(body["id"])
        pipe.youtube_played = 12
        paused = await client.post("/api/admin/youtube/pause")
        assert paused.status_code == 200, paused.text
        assert paused.json()["paused"] is True
        assert paused.json()["status"] == "paused"
        assert paused.json()["position"] == 12
        assert pipe.doc["youtube_offset"] == 12
        await pipe.accept_pcm(pcm(0.2, 0.2))
        assert pipe.youtube_played == 0
        started = await client.post("/api/admin/youtube/resume")
        assert started.status_code == 200, started.text
        assert started.json()["paused"] is False
        assert started.json()["status"] == "live"
        for _ in range(50):
            if len(commands) >= 2:
                break
            await asyncio.sleep(0.05)
    assert commands[-1].index("-ss") < commands[-1].index("-i")
    assert "12.000" in commands[-1]
    assert pipe.doc["youtube_paused"] is False


@pytest.mark.asyncio
async def test_loading_a_youtube_video_is_not_a_stall():
    pipe = pipeline_for([])
    pipe.doc["input"] = "youtube"
    pipe.doc["youtube_url"] = WATCH
    pipe.config.capture_stall_seconds = 0.2
    await pipe.start()
    pipe.youtube_loading = True
    pipe.last_audio_at = time.monotonic() - 5
    await asyncio.sleep(1.0)
    assert pipe.doc["status"] == "live"
    assert pipe.doc["banner"] != WAITING

    seen = {}

    class _Director:
        def pipeline(self, session_id):
            del session_id
            return pipe

    ingest = YouTubeIngest(_Director())

    async def resolve(_url):
        seen["loading"] = pipe.youtube_loading
        seen["banner"] = pipe.doc["banner"]
        return Media("https://rr.example/audio", False)

    async def spawn(_command):
        return _Proc([pcm(0.2, 0.0)])

    ingest.resolve = resolve
    ingest.spawn = spawn
    await ingest._pull(pipe.doc["id"], pipe)
    assert seen == {"loading": True, "banner": LOADING}
    assert pipe.youtube_loading is False
    assert pipe.doc["banner"] != LOADING
    pipe.last_audio_at = time.monotonic() - 5
    await asyncio.sleep(1.0)
    assert pipe.doc["banner"] == WAITING
    await pipe.stop()
