"""Admin test: play a YouTube video into the caption pipeline at speaking speed."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from app.rtmp import consume_pcm

logger = logging.getLogger(__name__)

WAITING = "Waiting for the YouTube audio."
ENDED = "The YouTube video ended."
FAILED = "The YouTube audio could not be opened."
UNAVAILABLE = "YouTube ingest is unavailable."
ALLOWED_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
)
RESOLVE_TIMEOUT = 45.0


@dataclass(frozen=True)
class Media:
    url: str
    live: bool
    duration: float = 0.0


def youtube_ready() -> bool:
    return bool(shutil.which("ffmpeg") and importlib.util.find_spec("yt_dlp") and _js_runtime())


def _js_runtime() -> bool:
    return bool(shutil.which("deno") or shutil.which("node") or shutil.which("nodejs"))


def check_youtube_url(value: str) -> str:
    raw = (value or "").strip()
    parts = urlsplit(raw)
    if parts.scheme != "https":
        raise ValueError("The YouTube address must start with https://")
    if parts.username or parts.password:
        raise ValueError("The YouTube address cannot include a username or password.")
    host = (parts.hostname or "").lower().rstrip(".")
    if host not in ALLOWED_HOSTS:
        raise ValueError("Use a youtube.com or youtu.be link.")
    if not _has_video(host, parts.path or "", parts.query or ""):
        raise ValueError("That link is not a YouTube video.")
    return raw


def _has_video(host: str, path: str, query: str) -> bool:
    pieces = [part for part in path.split("/") if part]
    if host == "youtu.be":
        return len(pieces) == 1
    if not pieces:
        return False
    if pieces[0] == "watch":
        return bool((parse_qs(query).get("v") or [""])[0].strip())
    if pieces[0] in {"shorts", "live", "embed"} and len(pieces) >= 2 and pieces[1]:
        return True
    return False


def ffmpeg_command(media_url: str, live: bool, offset: float = 0.0) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if not live and offset > 0:
        command.extend(["-ss", f"{offset:.3f}"])
    if not live:
        command.append("-re")
    command.extend(
        [
            "-i",
            media_url,
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
        ]
    )
    return command


def parse_media(raw: bytes) -> Media:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("yt-dlp returned no audio") from exc
    if not isinstance(data, dict):
        raise RuntimeError("yt-dlp returned no audio")
    url = str(data.get("url") or "")
    if not url.startswith("https://"):
        formats = data.get("requested_formats") or data.get("formats") or []
        if isinstance(formats, list):
            for item in formats:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("url") or "")
                if candidate.startswith("https://"):
                    url = candidate
                    break
    if not url.startswith("https://"):
        raise RuntimeError("yt-dlp returned no audio")
    return Media(url, bool(data.get("is_live")), _duration(data.get("duration")))


def _duration(value) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if seconds < 0 or seconds != seconds:
        return 0.0
    return seconds


class YouTubeIngest:
    def __init__(self, director):
        self.director = director
        self.tasks: dict[str, asyncio.Task] = {}
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.resolve = None
        self.spawn = None
        self.media: dict[str, Media] = {}

    def running(self, session_id: str) -> bool:
        task = self.tasks.get(session_id)
        return task is not None and not task.done()

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
        if status in {"ended", "deleted"} or pipeline.doc.get("input") != "youtube":
            return None
        return pipeline

    async def _run(self, session_id: str) -> None:
        try:
            pipeline = self._pipeline(session_id)
            if pipeline is None:
                return
            saw_audio = False
            try:
                saw_audio = await self._pull(session_id, pipeline)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.info("youtube ingest stopped for session %s", session_id)
                pipeline = self._pipeline(session_id)
                if pipeline is not None and pipeline.started and pipeline.doc.get("status") == "live":
                    if (pipeline.doc.get("banner") or "") != UNAVAILABLE:
                        await pipeline._set_status("paused", FAILED)
                return
            pipeline = self._pipeline(session_id)
            if pipeline is None or not pipeline.started:
                return
            if pipeline.doc.get("status") == "live":
                await pipeline._set_status("paused", ENDED if saw_audio else FAILED)
        finally:
            await self._kill(session_id)

    async def _pull(self, session_id: str, pipeline) -> bool:
        page = check_youtube_url(str(pipeline.doc.get("youtube_url") or ""))
        media = self.media.get(session_id)
        if media is None:
            media = await self._resolve(session_id, page)
            self.media[session_id] = media
        offset = 0.0 if media.live else float(pipeline.doc.get("youtube_offset") or 0)
        async with pipeline.lock:
            pipeline.youtube_played = 0.0
            pipeline.doc["youtube_live"] = media.live
            if media.duration:
                pipeline.doc["youtube_duration"] = media.duration
        command = ffmpeg_command(media.url, media.live, offset)
        spawn = self.spawn or _spawn_ffmpeg
        proc = await spawn(command)
        self.procs[session_id] = proc
        drain = asyncio.create_task(_discard(proc.stderr))
        try:
            return await consume_pcm(proc.stdout.read, pipeline.accept_pcm, pipeline.note_level)
        finally:
            drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain
            await self._kill(session_id)

    async def _resolve(self, session_id: str, page_url: str) -> Media:
        if self.resolve is not None:
            return await self.resolve(page_url)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-warnings",
            "--no-progress",
            "--dump-single-json",
            "-f",
            "ba/b",
            page_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.procs[session_id] = proc
        try:
            stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=RESOLVE_TIMEOUT)
        except asyncio.CancelledError:
            await _terminate(proc)
            raise
        except asyncio.TimeoutError:
            await _terminate(proc)
            raise RuntimeError("yt-dlp timed out") from None
        finally:
            if self.procs.get(session_id) is proc:
                self.procs.pop(session_id, None)
        if proc.returncode != 0:
            raise RuntimeError("yt-dlp failed")
        return parse_media(stdout or b"")

    async def _kill(self, session_id: str) -> None:
        proc = self.procs.pop(session_id, None)
        if proc is None:
            return
        await _terminate(proc)


async def _spawn_ffmpeg(command: list[str]):
    return await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _discard(stream) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.readline()
        if not chunk:
            return


async def _terminate(proc) -> None:
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
