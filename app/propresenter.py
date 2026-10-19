"""Push final listener translations into a ProPresenter Message."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from urllib.parse import quote, urlsplit

import httpx

from app.captions import project_line

logger = logging.getLogger(__name__)

PROPRESENTER_BANNER = "The ProPresenter message was not updated."


def propresenter_base(url: str) -> str:
    raw = (url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme != "http":
        raise ValueError("ProPresenter address must start with http://")
    if parsed.username or parsed.password:
        raise ValueError("ProPresenter address cannot include a username or password.")
    if parsed.fragment or parsed.query:
        raise ValueError("ProPresenter address cannot include a query or a fragment.")
    host = parsed.hostname
    if not host or parsed.port is None:
        raise ValueError("ProPresenter address needs a host and a port.")
    if parsed.path not in ("", "/"):
        raise ValueError("ProPresenter address cannot include a path.")
    netloc = f"[{host}]:{parsed.port}" if ":" in host else f"{host}:{parsed.port}"
    return f"http://{netloc}"


def propresenter_settings(raw: dict | None) -> dict:
    stored = raw or {}
    token = str(stored.get("token") or "Translation").strip() or "Translation"
    return {
        "enabled": bool(stored.get("enabled")),
        "base_url": str(stored.get("base_url") or "").strip(),
        "message": str(stored.get("message") or "").strip(),
        "token": token,
    }


def validate_propresenter(raw: dict | None) -> dict:
    settings = propresenter_settings(raw)
    if len(settings["message"]) > 80 or len(settings["token"]) > 40 or len(settings["base_url"]) > 200:
        raise ValueError("The ProPresenter settings are too long.")
    if settings["enabled"] or settings["base_url"]:
        settings["base_url"] = propresenter_base(settings["base_url"]) if settings["base_url"] else ""
    if settings["enabled"] and (not settings["base_url"] or not settings["message"] or not settings["token"]):
        raise ValueError("ProPresenter needs an address, a message name, and a token name.")
    return settings


def find_message_id(payload, name: str) -> str:
    if isinstance(payload, dict):
        items = payload.get("messages") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ident = item.get("id")
        item_name = ""
        item_id = ""
        if isinstance(ident, dict):
            item_name = str(ident.get("name") or "")
            item_id = str(ident.get("uuid") or ident.get("name") or "")
        elif isinstance(ident, str):
            item_id = ident
            item_name = str(item.get("name") or ident)
        else:
            item_name = str(item.get("name") or item.get("messageTitle") or "")
            item_id = str(item.get("uuid") or item_name)
        if item_name == name and item_id:
            return item_id
    raise LookupError(f"ProPresenter has no message named {name}.")


def caption_action(event: dict) -> tuple[str, str]:
    """What to do with one caption event: ignore, clear, or show text."""
    if event.get("type") == "deleted" or event.get("status") == "ended":
        return "clear", ""
    line = event.get("line")
    if not line:
        return "ignore", ""
    projected = project_line(line, "listener", tombstone=True)
    if not projected or projected.get("state") != "final":
        return "ignore", ""
    text = (projected.get("translation") or "").strip()
    if projected.get("hidden") or not text:
        return "clear", ""
    return "show", text


def trigger_body(token_name: str, text: str) -> list[dict]:
    return [{"name": token_name, "text": {"text": text}}]


class ProPresenterClient:
    def __init__(self, base_url: str, message: str, token: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.message = message
        self.token = token
        self._client = client
        self._owns = client is None
        self.message_id = ""

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=2.0, follow_redirects=False)
        return self._client

    async def aclose(self) -> None:
        if self._owns and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def resolve(self) -> str:
        response = await (await self._http()).get(f"{self.base_url}/v1/messages")
        self._check(response)
        self.message_id = find_message_id(response.json(), self.message)
        return self.message_id

    async def trigger(self, text: str) -> None:
        if not self.message_id:
            await self.resolve()
        response = await (await self._http()).post(
            f"{self.base_url}/v1/message/{quote(self.message_id, safe='')}/trigger",
            json=trigger_body(self.token, text),
        )
        self._check(response)

    async def clear(self) -> None:
        if not self.message_id:
            await self.resolve()
        response = await (await self._http()).get(
            f"{self.base_url}/v1/message/{quote(self.message_id, safe='')}/clear"
        )
        self._check(response)

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.status_code >= 300:
            raise RuntimeError("ProPresenter did not accept the request.")


class Stage:
    def __init__(self, director, client_factory=None):
        self.director = director
        self.client_factory = client_factory or ProPresenterClient
        self.tasks: dict[str, asyncio.Task] = {}
        self.clients: dict[str, ProPresenterClient] = {}
        self.last: dict[str, str] = {}

    def _stored(self, session_id: str) -> dict | None:
        pipeline = self.director.pipeline(session_id)
        if pipeline is None:
            return None
        stored = propresenter_settings(pipeline.doc.get("propresenter"))
        if not stored["enabled"]:
            return None
        return stored

    async def start(self, session_id: str) -> None:
        await self.stop(session_id)
        stored = self._stored(session_id)
        if stored is None:
            return
        try:
            base = propresenter_base(stored["base_url"])
        except ValueError:
            await self._note(session_id, PROPRESENTER_BANNER)
            return
        client = self.client_factory(base, stored["message"], stored["token"])
        self.clients[session_id] = client
        self.last[session_id] = ""
        self.tasks[session_id] = asyncio.create_task(self._run(session_id, client))

    async def stop(self, session_id: str) -> None:
        task = self.tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        client = self.clients.pop(session_id, None)
        if client is not None:
            await client.aclose()

    async def shutdown(self) -> None:
        for session_id in list(self.tasks):
            await self.stop(session_id)

    async def clear_session(self, session_id: str) -> None:
        client = self.clients.get(session_id)
        if client is None:
            return
        try:
            await client.clear()
        except Exception:
            logger.info("propresenter clear failed for session %s", session_id)
        self.last[session_id] = ""

    async def _run(self, session_id: str, client: ProPresenterClient) -> None:
        try:
            async with self.director.repository.subscribe(session_id) as subscription:
                async for event in subscription:
                    await self.apply(session_id, client, event)
                    if event.get("type") == "deleted" or event.get("status") == "ended":
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("propresenter sender stopped for session %s", session_id)

    async def apply(self, session_id: str, client, event: dict) -> None:
        action, text = caption_action(event)
        if action == "ignore":
            return
        try:
            if action == "clear":
                await client.clear()
                self.last[session_id] = ""
                return
            if text == self.last.get(session_id):
                return
            await client.trigger(text)
            self.last[session_id] = text
            await self._note(session_id, "")
        except Exception:
            logger.info("propresenter update failed for session %s", session_id)
            await self._note(session_id, PROPRESENTER_BANNER)

    async def _note(self, session_id: str, banner: str) -> None:
        pipeline = self.director.pipeline(session_id) if self.director is not None else None
        if pipeline is None:
            return
        current = pipeline.doc.get("banner") or ""
        if banner:
            if current and current != banner:
                return
            await pipeline._set_status(pipeline.doc.get("status") or "live", banner)
            return
        if current == PROPRESENTER_BANNER:
            await pipeline._set_status(pipeline.doc.get("status") or "live", pipeline._standing_banner())
