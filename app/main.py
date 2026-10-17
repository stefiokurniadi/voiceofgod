from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import Settings
from app.director import Director
from app.glossary import Glossary
from app.paths import GLOSSARY_PATH, STATIC_DIR
from app.store import MemoryRepository, RedisRepository
from app.stt import ScriptedRecognizer, WhisperRecognizer
from app.translator import GlossaryTranslator, UnavailableTranslator, VllmTranslator

logger = logging.getLogger("voiceofgod")


def build_recognizer(settings: Settings):
    if settings.allow_fake_models and not settings.whisper_model:
        return ScriptedRecognizer([])
    if settings.allow_fake_models and settings.whisper_device == "fake":
        return ScriptedRecognizer(["This", "is", "a", "development", "caption."])
    return WhisperRecognizer(
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.model_dir,
    )


def build_translator(settings: Settings):
    if settings.vllm_base_url:
        return VllmTranslator(settings.vllm_base_url, settings.vllm_model, settings.translate_timeout_seconds)
    if settings.allow_fake_models:
        return GlossaryTranslator()
    return UnavailableTranslator()


def build_repository(settings: Settings):
    if settings.redis_url:
        return RedisRepository(settings.redis_url)
    if settings.require_redis:
        raise RuntimeError("REDIS_URL is required.")
    logger.warning("REDIS_URL is empty; session state stays in this process.")
    return MemoryRepository()


def create_app(
    settings: Settings | None = None,
    *,
    repository=None,
    recognizer=None,
    translator=None,
    pipeline_config=None,
) -> FastAPI:
    settings = settings or Settings()
    glossary = Glossary.load(GLOSSARY_PATH)
    repository = repository if repository is not None else build_repository(settings)
    recognizer = recognizer if recognizer is not None else build_recognizer(settings)
    translator = translator if translator is not None else build_translator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.repository.connect()
        yield
        closer = getattr(app.state.translator, "aclose", None)
        if closer:
            await closer()
        await app.state.repository.close()

    app = FastAPI(title="voiceofgod", lifespan=lifespan)
    app.state.settings = settings
    app.state.repository = repository
    app.state.recognizer = recognizer
    app.state.translator = translator
    app.state.director = Director(settings, repository, recognizer, translator, glossary, pipeline_config)
    app.include_router(router)

    @app.get("/")
    async def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/o/{session_id}")
    async def operator_page(session_id: str, request: Request):
        document = await request.app.state.director.session(session_id)
        if not document:
            raise HTTPException(404, "Session not found.")
        return FileResponse(STATIC_DIR / "operator.html")

    @app.get("/l/{session_id}")
    async def listener_page(session_id: str, request: Request):
        document = await request.app.state.director.session(session_id)
        if not document:
            raise HTTPException(404, "Session not found.")
        return FileResponse(STATIC_DIR / "listener.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
