from __future__ import annotations

import asyncio
from collections import defaultdict
import copy
import json
import logging

logger = logging.getLogger(__name__)


class MemoryRepository:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save(self, session_id: str, document: dict) -> None:
        async with self._lock:
            self.sessions[session_id] = copy.deepcopy(document)

    async def load(self, session_id: str) -> dict | None:
        async with self._lock:
            document = self.sessions.get(session_id)
            return copy.deepcopy(document) if document else None

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return list(self.sessions)

    async def publish(self, session_id: str, event: dict) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(session_id, []))
        payload = copy.deepcopy(event)
        for queue in queues:
            await queue.put(copy.deepcopy(payload))

    def subscribe(self, session_id: str) -> MemorySubscription:
        return MemorySubscription(self, session_id)


class MemorySubscription:
    def __init__(self, repository: MemoryRepository, session_id: str):
        self.repository = repository
        self.session_id = session_id
        self.queue: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self) -> MemorySubscription:
        async with self.repository._lock:
            self.repository._subscribers[self.session_id].append(self.queue)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        async with self.repository._lock:
            subscribers = self.repository._subscribers.get(self.session_id, [])
            if self.queue in subscribers:
                subscribers.remove(self.queue)

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        return await self.queue.get()


class RedisRepository:
    def __init__(self, url: str):
        self.url = url
        self.redis = None

    async def connect(self) -> None:
        import redis.asyncio as redis

        self.redis = redis.from_url(self.url, decode_responses=True)
        await self.redis.ping()
        logger.info("connected to valkey")

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()

    def _key(self, session_id: str) -> str:
        return f"voiceofgod:session:{session_id}"

    def _channel(self, session_id: str) -> str:
        return f"voiceofgod:events:{session_id}"

    async def save(self, session_id: str, document: dict) -> None:
        await self.redis.set(self._key(session_id), json.dumps(document))
        await self.redis.sadd("voiceofgod:sessions", session_id)

    async def load(self, session_id: str) -> dict | None:
        raw = await self.redis.get(self._key(session_id))
        if not raw:
            return None
        return json.loads(raw)

    async def delete(self, session_id: str) -> None:
        await self.redis.delete(self._key(session_id))
        await self.redis.srem("voiceofgod:sessions", session_id)

    async def list_ids(self) -> list[str]:
        ids = await self.redis.smembers("voiceofgod:sessions")
        if ids:
            return list(ids)
        found = []
        async for key in self.redis.scan_iter("voiceofgod:session:*"):
            found.append(str(key).removeprefix("voiceofgod:session:"))
        return found

    async def publish(self, session_id: str, event: dict) -> None:
        await self.redis.publish(self._channel(session_id), json.dumps(event))

    def subscribe(self, session_id: str) -> RedisSubscription:
        return RedisSubscription(self.redis, self._channel(session_id))


class RedisSubscription:
    def __init__(self, redis, channel: str):
        self.redis = redis
        self.channel = channel
        self.pubsub = None

    async def __aenter__(self) -> RedisSubscription:
        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(self.channel)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.pubsub is not None:
            await self.pubsub.unsubscribe(self.channel)
            await self.pubsub.aclose()

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        assert self.pubsub is not None
        async for message in self.pubsub.listen():
            if message.get("type") != "message":
                continue
            return json.loads(message["data"])
        raise StopAsyncIteration
