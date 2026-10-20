from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import asyncio
import hashlib
import hmac
import logging
import secrets

from app.pipeline import PipelineConfig, SessionPipeline
from app.propresenter import Stage, validate_propresenter
from app.rtmp import UNAVAILABLE, RtmpIngest, rtmp_ready
from app.youtube import UNAVAILABLE as YOUTUBE_UNAVAILABLE
from app.youtube import YouTubeIngest, check_youtube_url, youtube_ready

logger = logging.getLogger(__name__)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str, token_hash: str) -> bool:
    if not token or not token_hash:
        return False
    return hmac.compare_digest(hash_token(token), token_hash)


class Director:
    def __init__(self, settings, repository, recognizer, translator, glossary, pipeline_config: PipelineConfig | None = None, speaker=None, accounts=None):
        self.settings = settings
        self.repository = repository
        self.accounts = accounts
        self.recognizer = recognizer
        self.translator = translator
        self.speaker = speaker
        self.glossary = glossary
        self.pipeline_config = pipeline_config
        self.pipelines: dict[str, SessionPipeline] = {}
        self.listeners: dict[str, int] = defaultdict(int)
        self.ingest = RtmpIngest(settings, self)
        self.youtube = YouTubeIngest(self)
        self.youtube_secrets: dict[str, str] = {}
        self.stage = Stage(self)

    async def create_session(
        self,
        user: dict,
        name: str,
        note: str,
        service_date,
        speaker_language: str,
        voice: bool,
        base_url: str,
        do_not_translate: list[str] | None = None,
        terms: list[dict] | None = None,
        translation_language: str | None = None,
    ) -> dict:
        from app.translator import CAPTION_LANGUAGE_ERROR, SPEAKER_LANGUAGES, SPEAKER_LANGUAGE_ERROR, target_language

        if not user.get("church_id"):
            raise PermissionError("Set the church before creating a service.")
        if speaker_language not in SPEAKER_LANGUAGES:
            raise ValueError(SPEAKER_LANGUAGE_ERROR)
        if speaker_language == "mix" and translation_language not in (None, "", "en", "id"):
            raise ValueError(CAPTION_LANGUAGE_ERROR)
        caption = target_language(speaker_language, translation_language or user.get("translation_language"))
        cleaned_name = (name or "").strip() or "Service"
        if len(cleaned_name) > 120:
            raise ValueError("The service name is too long.")
        if len(note or "") > 4000:
            raise ValueError("The service note is too long.")
        extra_names = [phrase.strip() for phrase in (do_not_translate or []) if phrase and str(phrase).strip()]
        keep_names = [str(phrase) for phrase in list(user.get("keep_names") or []) + extra_names if str(phrase).strip()]
        if len(keep_names) > 100 or any(len(phrase) > 80 for phrase in keep_names):
            raise ValueError("The do-not-translate list is too large.")
        terms = terms or []
        if len(terms) > 100:
            raise ValueError("The glossary is too large.")
        self.recognizer.ensure_loaded()
        translator_ready = await self.translator.healthy()
        if self.settings.require_translator and not translator_ready:
            raise RuntimeError("Translator is not ready.")
        session_id = secrets.token_urlsafe(12)
        secret = secrets.token_urlsafe(24)
        glossary = self.glossary.merged(keep_names, terms)
        document = {
            "id": session_id,
            "speaker_language": speaker_language,
            "translation_language": caption,
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
            "glossary": glossary.to_dict(),
            "voice": bool(voice),
            "church_id": str(user["church_id"]),
            "church_name": user.get("church_name") or "",
            "church_note": user.get("church_note") or "",
            "service_name": cleaned_name,
            "service_note": (note or "").strip(),
            "service_date": service_date.isoformat() if service_date else "",
            "input": "browser",
            "ingest_token": secrets.token_urlsafe(18),
            "propresenter": {"enabled": False, "base_url": "", "message": "", "token": "Translation"},
        }
        if document["development_translator"]:
            document["banner"] = "Development translator is on. These are not real translations."
        pipeline = self._bind(document, glossary)
        self.pipelines[session_id] = pipeline
        await self.repository.save(session_id, document)
        if self.accounts is not None:
            await self.accounts.insert_service(
                session_id,
                str(user["church_id"]),
                cleaned_name,
                (note or "").strip(),
                service_date,
                speaker_language,
                bool(voice),
                document["token_hash"],
                document,
                caption,
            )
        logger.info("session %s created", session_id)
        return {
            "id": session_id,
            "name": cleaned_name,
            "operator_secret": secret,
            "operator_url": f"{base_url}/o/{session_id}?k={secret}",
            "listener_url": f"{base_url}/l/{session_id}",
            "speaker_language": speaker_language,
            "translation_language": caption,
        }

    def _bind(self, document: dict, glossary=None):
        from app.glossary import Glossary

        if glossary is None:
            raw = document.get("glossary") or {}
            try:
                glossary = Glossary.from_public(raw.get("terms") or [], raw.get("do_not_translate") or [])
            except ValueError:
                glossary = self.glossary
        pipeline = SessionPipeline(
            document,
            self.repository,
            self.recognizer,
            self.translator,
            glossary,
            self.pipeline_config,
            self.speaker,
        )
        pipeline.durable = self._durable
        return pipeline

    async def _durable(self, document: dict) -> None:
        if self.accounts is not None:
            await self.accounts.save_document(document["id"], document)

    async def remember(self, session_id: str) -> None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None:
            await self._durable(pipeline.doc)

    async def restore(self) -> int:
        from app.glossary import Glossary

        restored = 0
        for session_id in await self.repository.list_ids():
            if session_id in self.pipelines:
                continue
            document = await self.repository.load(session_id)
            if not document or document.get("status") == "deleted":
                continue
            raw = document.get("glossary") or {}
            try:
                glossary = Glossary.from_public(raw.get("terms") or [], raw.get("do_not_translate") or [])
            except ValueError:
                glossary = self.glossary
            if document.get("status") in {"live", "paused"}:
                document["status"] = "paused"
                document["banner"] = "The caption service restarted. Press Start to continue."
                document.setdefault("metrics", {})["listeners"] = 0
            pipeline = self._bind(document, glossary)
            self.pipelines[session_id] = pipeline
            await self.repository.save(session_id, document)
            restored += 1
        if restored:
            await asyncio.to_thread(self.recognizer.ensure_loaded)
            logger.info("restored %s sessions", restored)
        return restored

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
        already = pipeline.started and pipeline.doc.get("status") != "ended"
        youtube_resume = (
            already
            and pipeline.doc.get("input") == "youtube"
            and not self.youtube.running(session_id)
        )
        if already and not youtube_resume:
            return
        if not already and self.live_count() >= self.settings.max_live_sessions:
            raise RuntimeError(f"This server is already running {self.settings.max_live_sessions} sessions.")
        await pipeline.start()
        if pipeline.doc.get("input") == "rtmp":
            if not rtmp_ready(self.settings):
                await pipeline._set_status("live", UNAVAILABLE)
            else:
                await self.ingest.start(session_id)
        if pipeline.doc.get("input") == "youtube":
            if not youtube_ready() and self.youtube.resolve is None:
                await pipeline._set_status("live", YOUTUBE_UNAVAILABLE)
            else:
                await self.youtube.start(session_id)
        await self.stage.start(session_id)

    async def stop(self, session_id: str) -> None:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        await self.ingest.stop(session_id)
        await self.youtube.stop(session_id)
        await pipeline.stop()
        await self.stage.clear_session(session_id)
        await self.stage.stop(session_id)

    async def delete(self, session_id: str) -> None:
        await self.youtube.stop(session_id)
        self.youtube_secrets.pop(session_id, None)
        await self.ingest.stop(session_id)
        await self.stage.clear_session(session_id)
        await self.stage.stop(session_id)
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

    async def update_glossary(self, session_id: str, terms: list[dict], do_not_translate: list[str]) -> dict:
        from app.glossary import Glossary

        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        glossary = Glossary.from_public(terms, do_not_translate)
        async with pipeline.lock:
            pipeline.glossary = glossary
            pipeline.doc["glossary"] = glossary.to_dict()
        await pipeline._emit("status")
        await self.remember(session_id)
        return glossary.to_dict()

    async def set_input(self, session_id: str, source: str) -> str:
        if source not in {"browser", "rtmp"}:
            raise ValueError("Audio source must be the microphone or OBS.")
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        if pipeline.doc.get("input") == "youtube":
            raise RuntimeError("A YouTube test keeps its audio source.")
        if pipeline.started and pipeline.doc.get("status") != "ended":
            raise RuntimeError("End the session before changing the audio source.")
        async with pipeline.lock:
            pipeline.doc["input"] = source
        await pipeline._emit("status")
        await self.remember(session_id)
        return source

    async def set_propresenter(self, session_id: str, raw: dict) -> dict:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        saved = validate_propresenter(raw)
        async with pipeline.lock:
            pipeline.doc["propresenter"] = saved
        await pipeline._emit("status")
        if pipeline.started and pipeline.doc.get("status") != "ended":
            await self.stage.start(session_id)
        await self.remember(session_id)
        return saved

    async def shutdown(self) -> None:
        await self.ingest.shutdown()
        await self.youtube.shutdown()
        await self.stage.shutdown()

    def active_youtube(self):
        for pipeline in self.pipelines.values():
            if pipeline.doc.get("input") != "youtube":
                continue
            if pipeline.started and pipeline.doc.get("status") != "ended":
                return pipeline
        return None

    async def create_youtube_test(self, url: str, speaker_language: str, base_url: str) -> dict:
        page = check_youtube_url(url)
        from app.translator import SPEAKER_LANGUAGES, SPEAKER_LANGUAGE_ERROR, target_language

        if speaker_language not in SPEAKER_LANGUAGES:
            raise ValueError(SPEAKER_LANGUAGE_ERROR)
        caption = target_language(speaker_language)
        if self.active_youtube() is not None:
            raise RuntimeError("A YouTube test is already running.")
        if self.live_count() >= self.settings.max_live_sessions:
            raise RuntimeError(f"This server is already running {self.settings.max_live_sessions} sessions.")
        self.recognizer.ensure_loaded()
        translator_ready = await self.translator.healthy()
        if self.settings.require_translator and not translator_ready:
            raise RuntimeError("Translator is not ready.")
        session_id = secrets.token_urlsafe(12)
        secret = secrets.token_urlsafe(24)
        glossary = self.glossary.merged([], [])
        document = {
            "id": session_id,
            "speaker_language": speaker_language,
            "translation_language": caption,
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
            "glossary": glossary.to_dict(),
            "voice": False,
            "church_id": "",
            "church_name": "",
            "church_note": "",
            "service_name": "YouTube test",
            "service_note": "",
            "service_date": "",
            "input": "youtube",
            "youtube_url": page,
            "youtube_offset": 0,
            "youtube_duration": 0,
            "youtube_live": False,
            "ingest_token": "",
            "propresenter": {"enabled": False, "base_url": "", "message": "", "token": "Translation"},
        }
        if document["development_translator"]:
            document["banner"] = "Development translator is on. These are not real translations."
        pipeline = self._bind(document, glossary)
        self.pipelines[session_id] = pipeline
        self.youtube_secrets[session_id] = secret
        await self.repository.save(session_id, document)
        await self.start(session_id)
        logger.info("youtube test %s created", session_id)
        return {
            "id": session_id,
            "operator_url": f"{base_url}/o/{session_id}?k={secret}",
            "speaker_language": speaker_language,
        }

    def youtube_view(self, base_url: str) -> dict:
        pipeline = self.active_youtube()
        if pipeline is None:
            return {"active": False}
        session_id = pipeline.doc["id"]
        secret = self.youtube_secrets.get(session_id) or ""
        view = {
            "active": True,
            "id": session_id,
            "url": pipeline.doc.get("youtube_url") or "",
            "status": pipeline.doc.get("status") or "",
            "banner": pipeline.doc.get("banner") or "",
            **self.youtube_position(pipeline),
        }
        if secret:
            view["operator_url"] = f"{base_url}/o/{session_id}?k={secret}"
        return view

    def youtube_position(self, pipeline) -> dict:
        duration = float(pipeline.doc.get("youtube_duration") or 0)
        offset = float(pipeline.doc.get("youtube_offset") or 0)
        position = offset + float(getattr(pipeline, "youtube_played", 0) or 0)
        if duration:
            position = min(position, duration)
        return {
            "position": round(position, 3),
            "duration": round(duration, 3),
            "live": bool(pipeline.doc.get("youtube_live")),
        }

    async def seek_youtube(self, session_id: str, seconds: float) -> dict:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None or pipeline.doc.get("input") != "youtube":
            raise ValueError("Only a YouTube test can move the playback time.")
        if pipeline.doc.get("youtube_live"):
            raise ValueError("A live stream cannot be moved.")
        try:
            target = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("The time has to be a number of seconds.")
        if target < 0:
            raise ValueError("The time has to be zero or later.")
        duration = float(pipeline.doc.get("youtube_duration") or 0)
        if duration and target > duration:
            raise ValueError("That time is past the end of the video.")
        await self.youtube.stop(session_id)
        async with pipeline.lock:
            pipeline.doc["youtube_offset"] = target
        await pipeline.drop_partial()
        if pipeline.doc.get("status") == "ended" or not pipeline.started:
            await self.start(session_id)
        else:
            if pipeline.doc.get("status") != "live":
                await pipeline._set_status("live", pipeline._standing_banner())
            if not youtube_ready() and self.youtube.resolve is None:
                await pipeline._set_status("live", YOUTUBE_UNAVAILABLE)
            else:
                await self.youtube.start(session_id)
        return self.youtube_position(pipeline)

    async def stop_youtube_test(self) -> None:
        pipeline = self.active_youtube()
        if pipeline is None:
            raise KeyError("youtube")
        await self.stop(pipeline.doc["id"])

    async def set_service_voice(self, session_id: str, enabled: bool) -> bool:
        pipeline = self.pipelines.get(session_id)
        if pipeline is None:
            raise KeyError(session_id)
        await pipeline.set_voice(enabled)
        await self.remember(session_id)
        return bool(enabled)

    async def apply_church(self, church_id: str, name: str, note: str) -> None:
        for pipeline in list(self.pipelines.values()):
            if str(pipeline.doc.get("church_id") or "") != str(church_id):
                continue
            async with pipeline.lock:
                pipeline.doc["church_name"] = name
                pipeline.doc["church_note"] = note
            await self._durable(pipeline.doc)

    def _may_open(self, document, *, church_id: str | None, token: str | None, public: bool) -> bool:
        if public:
            return True
        if church_id is not None and str(document.get("church_id") or "") == str(church_id):
            return True
        return bool(token) and token_matches(token, document.get("token_hash", ""))

    async def ensure_loaded(self, session_id: str, *, church_id: str | None = None, token: str | None = None, public: bool = False):
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None:
            document = pipeline.doc
            if self._may_open(document, church_id=church_id, token=token, public=public):
                return document
            return None
        if self.accounts is None:
            return await self.repository.load(session_id)
        row = await self.accounts.load_service(session_id)
        if row is None or row["status"] == "deleted":
            return None
        if not self._may_open(row, church_id=church_id, token=token, public=public):
            return None
        document = row["document"]
        if isinstance(document, str):
            import json
            document = json.loads(document)
        if document.get("status") in {"live", "paused"}:
            document["status"] = "paused"
            document["banner"] = "The caption service restarted. Press Start to continue."
            document.setdefault("metrics", {})["listeners"] = 0
        self.recognizer.ensure_loaded()
        pipeline = self._bind(document)
        self.pipelines[session_id] = pipeline
        await self.repository.save(session_id, document)
        await self._durable(document)
        return document

    async def note_listeners(self, session_id: str, delta: int) -> None:
        self.listeners[session_id] = max(0, self.listeners[session_id] + delta)
        pipeline = self.pipelines.get(session_id)
        if pipeline is not None:
            await pipeline.set_listener_count(self.listeners[session_id])
