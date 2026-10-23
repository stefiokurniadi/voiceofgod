"""OBS publishes RTMP. ffmpeg pulls 16 kHz audio into the caption pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import shutil
import time
from urllib.parse import parse_qs, quote, urlsplit

from app.audio import pcm16_to_float, rms

logger = logging.getLogger(__name__)

WAITING = "Waiting for OBS."
DROPPED = "OBS stream stopped. It will reconnect when OBS publishes again."
UNAVAILABLE = "OBS ingest is unavailable."
READER_USER = "reader"
CHUNK_BYTES = 3200
CEILING_BYTES = 64000


def secret_matches(presented: str, expected: str) -> bool:
    if not presented or not expected:
        return False
    left = hashlib.sha256(presented.encode("utf-8")).hexdigest()
    right = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    return hmac.compare_digest(left, right)


def rtmp_ready(settings) -> bool:
    return bool(
        (settings.rtmp_public_url or "").strip()
        and (settings.mediamtx_rtmp_url or "").strip()
        and (settings.rtmp_auth_secret or "").strip()
        and (settings.rtmp_read_password or "").strip()
        and shutil.which("ffmpeg")
    )


def public_server(settings) -> str:
    base = (settings.rtmp_public_url or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/live"


def stream_session(path: str, query: str) -> tuple[str, str]:
    raw_path, _, embedded = (path or "").partition("?")
    pieces = [part for part in raw_path.split("/") if part]
    session_id = ""
    if len(pieces) == 2 and pieces[0] == "live":
        session_id = pieces[1]
    params = parse_qs(embedded or query or "", keep_blank_values=False)
    token = (params.get("token") or [""])[0]
    return session_id, token


def allow_publish(token: str, expected: str | None) -> bool:
    return bool(expected) and secret_matches(token, expected)


def allow_read(user: str, password: str, read_password: str, session_exists: bool) -> bool:
    return bool(session_exists) and user == READER_USER and secret_matches(password, read_password)


def reader_url(settings, session_id: str) -> str:
    base = (settings.mediamtx_rtmp_url or "").strip()
    parts = urlsplit(base)
    if parts.scheme != "rtmp" or not parts.hostname:
        raise RuntimeError("MEDIAMTX_RTMP_URL must be an rtmp:// address.")
    port = parts.port or 1935
    password = quote(settings.rtmp_read_password, safe="")
    session = quote(session_id, safe="")
    # MediaMTX 1.13 reads the RTMP user and password from the query string.
    # Credentials in the URL userinfo are ignored, so the pull is refused.
    return f"rtmp://{parts.hostname}:{port}/live/{session}?user={quote(READER_USER, safe='')}&pass={password}"


class PcmBuffer:
    """Keep ingest near realtime. A backlog past the ceiling is dropped."""

    def __init__(self, chunk_bytes: int = CHUNK_BYTES, ceiling_bytes: int = CEILING_BYTES):
        self.buf = bytearray()
        self.chunk_bytes = chunk_bytes
        self.ceiling_bytes = ceiling_bytes

    def extend(self, data: bytes) -> None:
        if data:
            self.buf.extend(data)

    def pop(self) -> tuple[bytes | None, bool]:
        dropped = False
        even = len(self.buf) - (len(self.buf) % 2)
        if even > self.ceiling_bytes:
            dropped = True
            start = even - self.chunk_bytes
            del self.buf[:start]
            even = len(self.buf) - (len(self.buf) % 2)
        if even < self.chunk_bytes:
            return None, dropped
        chunk = bytes(self.buf[: self.chunk_bytes])
        del self.buf[: self.chunk_bytes]
        return chunk, dropped


async def consume_pcm(read, accept, level=None, buffer: PcmBuffer | None = None) -> bool:
    """Read PCM bytes and pass short chunks to accept. Returns whether any audio arrived."""
    buffer = buffer or PcmBuffer()
    saw_audio = False
    while True:
        data = await read(CHUNK_BYTES)
        if not data:
            return saw_audio
        buffer.extend(data)
        while True:
            chunk, _dropped = buffer.pop()
            if chunk is None:
                break
            saw_audio = True
            if level is not None:
                level(rms(pcm16_to_float(chunk)))
            await accept(chunk)


class RtmpIngest:
    def __init__(self, settings, director):
        self.settings = settings
        self.director = director
        self.tasks: dict[str, asyncio.Task] = {}
        self.procs: dict[str, asyncio.subprocess.Process] = {}

    async def start(self, session_id: str) -> None:
        current = self.tasks.get(session_id)
        if current is not None and not current.done():
            return
        self.tasks[session_id] = asyncio.create_task(self._run(session_id))

    async def stop(self, session_id: str) -> None:
        task = self.tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._kill(session_id)

    async def shutdown(self) -> None:
        for session_id in list(self.tasks):
            await self.stop(session_id)

    def _pipeline(self, session_id: str):
        pipeline = self.director.pipeline(session_id)
        if pipeline is None:
            return None
        status = pipeline.doc.get("status")
        if status in {"ended", "deleted"} or pipeline.doc.get("input") != "rtmp":
            return None
        return pipeline

    async def _run(self, session_id: str) -> None:
        try:
            while True:
                pipeline = self._pipeline(session_id)
                if pipeline is None:
                    return
                if not pipeline.started or pipeline.failure_hold:
                    await asyncio.sleep(0.3)
                    continue
                saw_audio = False
                try:
                    saw_audio = await self._pull(session_id, pipeline)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.info("obs ingest stopped for session %s", session_id)
                pipeline = self._pipeline(session_id)
                if pipeline is None or not pipeline.started or pipeline.failure_hold:
                    continue
                if not (saw_audio or pipeline.rtmp_seen) and pipeline.doc.get("status") == "live":
                    if (pipeline.doc.get("banner") or "") != WAITING:
                        await pipeline._set_status("live", WAITING)
                await asyncio.sleep(1.0)
        finally:
            await self._kill(session_id)

    async def _pull(self, session_id: str, pipeline) -> bool:
        url = reader_url(self.settings, session_id)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-rw_timeout",
            "5000000",
            "-i",
            url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.procs[session_id] = proc
        drain = asyncio.create_task(_discard(proc.stderr))
        watch = asyncio.create_task(self._abandon_when_quiet(session_id, pipeline, time.monotonic()))
        try:
            saw_audio = await consume_pcm(proc.stdout.read, pipeline.accept_pcm, pipeline.note_level)
            if saw_audio:
                pipeline.rtmp_seen = True
            return saw_audio or pipeline.rtmp_seen
        finally:
            watch.cancel()
            drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            with contextlib.suppress(asyncio.CancelledError):
                await drain
            await self._kill(session_id)

    async def _abandon_when_quiet(self, session_id: str, pipeline, started_at: float) -> None:
        """A pull that receives nothing is stuck on the old publish. Close it and try again."""
        limit = pipeline.config.capture_stall_seconds
        while True:
            await asyncio.sleep(min(0.4, limit))
            proc = self.procs.get(session_id)
            if proc is None or proc.returncode is not None:
                return
            now = time.monotonic()
            if now - started_at < limit or now - pipeline.last_audio_at < limit:
                continue
            logger.info("obs ingest quiet for session %s; reconnecting", session_id)
            await self._kill(session_id)
            return

    async def _kill(self, session_id: str) -> None:
        proc = self.procs.pop(session_id, None)
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=2)


async def _discard(stream) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.readline()
        if not chunk:
            return
