from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import hmac
import logging
import secrets

from app.pipeline import PipelineConfig, SessionPipeline

logger = logging.getLogger(__name__)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str, token_hash: str) -> bool:
    if not token or not token_hash:
        return False
    return hmac.compare_digest(hash_token(token), token_hash)


class Director:
    def __init__(self, settings, repository, recognizer, translator, glossary, pipeline_config: PipelineConfig | None = None):
        self.settings = settings
        self.repository = repository
        self.recognizer = recognizer
        self.translator = translator
        self.glossary = glossary
        self.pipeline_config = pipeline_config
        self.pipelines: dict[str, SessionPipeline] = {}
        self.listeners: dict[str, int] = defaultdict(int)

    async def create_session(self, speaker_language: str, do_not_translate: list[str], terms: list[dict], base_url: str) -> dict:
        if speaker_language not in {"en", "id"}:
            raise ValueError("Speaker language must be English or Bahasa Indonesia.")
        do_not_translate = [phrase.strip() for phrase in do_not_translate if phrase and phrase.strip()]
        if len(do_not_translate) > 100 or any(len(phrase) > 80 for phrase in do_not_translate):
            raise ValueError("The do-not-translate list is too large.")
        if len(terms) > 100:
            raise ValueError("The glossary is too large.")
        self.recognizer.ensure_loaded()
        translator_ready = await self.translator.healthy()
        if self.settings.require_translator and not translator_ready:
            raise RuntimeError("Translator is not ready.")
        session_id = secrets.token_urlsafe(12)
        secret = secrets.token_urlsafe(24)
        document = {
            "id": session_id,
            "speaker_language": speaker_language,
            "status": "ready",
            "banner": "",
            "seq": 0,
            "lines": [],
            "provisional": None,
            "line_counter": 0,
            "metrics": {"listeners": 0, "stt_ms": None, "translate_ms": None, "lag_ms": None},
            "token_hash": hash_token(secret),
            "translator_ready": translator_ready,
            "development_translator": bool(getattr(self.translator, "development", False)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if document["development_translator"]:
            document["banner"] = "Development translator is on. These are not real translations."
        glossary = self.glossary.merged(do_not_translate, terms)
        pipeline = SessionPipeline(
            document,
            self.repository,
            self.recognizer,
            self.translator,
            glossary,
            self.pipeline_config,
        )
        self.pipelines[session_id] = pipeline
        await self.repository.save(session_id, document)
        logger.info("session %s created", session_id)
        return {
            "id": session_id,
            "operator_secret": secret,
            "operator_url": f"{base_url}/o/{session_id}?k={secret}",
            "listener_url": f"{base_url}/l/{session_id}",
            "speaker_language": speaker_language,
        }

    def pipeline(self, session_id: str) -> SessionPipeline | None:
        return self.pipelines.get(session_id)

    async def session(self, session_id: str) -> dict | None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None:
            return pipeline.doc
        return await self.repository.load(session_id)

    async def authorize(self, session_id: str, token: str) -> dict | None:
        document = await self.session(session_id)
        if not document or not token_matches(token, document.get("token_hash", "")):
            return None
        return document

    def live_count(self) -> int:
        return sum(1 for pipeline in self.pipelines.values() if pipeline.started and pipeline.doc.get("status") != "ended")

    async def start(self, session_id: str) -> None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        if pipeline.started and pipeline.doc.get("status") != "ended":
            return
        if self.live_count() >= self.settings.max_live_sessions:
            raise RuntimeError("A session is already live on this server.")
        await pipeline.start()

    async def stop(self, session_id: str) -> None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        await pipeline.stop()

    async def delete(self, session_id: str) -> None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None and pipeline.started and pipeline.doc.get("status") != "ended":
            await pipeline.stop()
        if pipeline is not None:
            async with pipeline.lock:
                pipeline.doc["lines"] = []
                pipeline.doc["provisional"] = None
                pipeline.doc["status"] = "deleted"
            await pipeline._emit("deleted")
        await self.repository.delete(session_id)
        self.pipelines.pop(session_id, None)
        logger.info("session %s deleted", session_id)

    async def note_listeners(self, session_id: str, delta: int) -> None:
        self.listeners[session_id] = max(0, self.listeners[session_id] + delta)
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None:
            await pipeline.set_listener_count(self.listeners[session_id])
