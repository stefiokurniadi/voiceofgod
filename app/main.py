from __future__ import annotations

from contextlib import asynccontextmanager
import html
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.accounts import Accounts, cookie_token
from app.api import router
from app.config import Settings
from app.db import Database
from app.director import Director
from app.glossary import Glossary
from app.paths import GLOSSARY_PATH, STATIC_DIR
from app.pipeline import PipelineConfig
from app.store import MemoryRepository, RedisRepository
from app.stt import GoogleRecognizer, ScriptedRecognizer, WhisperRecognizer
from app.translator import GeminiTranslator, GlossaryTranslator, UnavailableTranslator, VllmTranslator
from app.tts import GoogleSpeaker

logger = logging.getLogger("voiceofgod")


def build_recognizer(settings: Settings):
    if settings.google_api_key and settings.google_cloud_project:
        return GoogleRecognizer(
            settings.google_api_key,
            settings.google_cloud_project,
            settings.google_speech_region,
            settings.google_speech_model,
        )
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
    if settings.gemini_api_key:
        return GeminiTranslator(settings.gemini_api_key, settings.gemini_model)
    if settings.vllm_base_url:
        return VllmTranslator(settings.vllm_base_url, settings.vllm_model, settings.translate_timeout_seconds)
    if settings.allow_fake_models:
        return GlossaryTranslator()
    return UnavailableTranslator()


def build_speaker(settings: Settings):
    if not settings.google_api_key:
        return None
    return GoogleSpeaker(settings.google_api_key, settings.tts_voice_en, settings.tts_voice_id)


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
    speaker=None,
    pipeline_config=None,
    database: Database | None = None,
    accounts: Accounts | None = None,
) -> FastAPI:
    settings = settings or Settings()
    glossary = Glossary.load(GLOSSARY_PATH)
    database = database if database is not None else Database(settings.database_url)
    accounts = accounts if accounts is not None else Accounts(database, settings)
    repository = repository if repository is not None else build_repository(settings)
    recognizer = recognizer if recognizer is not None else build_recognizer(settings)
    translator = translator if translator is not None else build_translator(settings)
    if speaker is None and settings.google_api_key:
        speaker = build_speaker(settings)
    if pipeline_config is None and settings.gemini_api_key:
        pipeline_config = PipelineConfig(translate_timeout_seconds=max(settings.translate_timeout_seconds, 8.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.database.connect()
        await app.state.database.migrate()
        await app.state.accounts.ensure_admin()
        await app.state.repository.connect()
        yield
        await app.state.director.shutdown()
        closer = getattr(app.state.translator, "aclose", None)
        if closer:
            await closer()
        speaker_closer = getattr(app.state.speaker, "aclose", None)
        if speaker_closer:
            await speaker_closer()
        recognizer_closer = getattr(app.state.recognizer, "close", None)
        if recognizer_closer:
            recognizer_closer()
        await app.state.repository.close()
        await app.state.database.close()

    app = FastAPI(title="voiceofgod", lifespan=lifespan)
    app.state.settings = settings
    app.state.repository = repository
    app.state.recognizer = recognizer
    app.state.translator = translator
    app.state.speaker = speaker
    app.state.database = database
    app.state.accounts = accounts
    app.state.director = Director(settings, repository, recognizer, translator, glossary, pipeline_config, speaker, accounts)
    app.include_router(router)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        accept = request.headers.get("accept", "")
        if exc.status_code == 404 and "text/html" in accept:
            detail = html.escape(exc.detail if isinstance(exc.detail, str) else "Not found.")
            page = (
                "<!doctype html><html lang='en'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>Not found · voiceofgod</title>"
                "<link rel='icon' href='/static/icon.svg' type='image/svg+xml'>"
                "<link rel='stylesheet' href='/static/style.css'>"
                "<body class='auth'><main class='auth-card'>"
                "<a class='brand' href='/'><img src='/static/icon.svg' alt=''><span>voiceofgod</span></a>"
                f"<h1>Not found</h1><p class='lede'>{detail}</p>"
                "<a class='btn btn-primary btn-block' href='/'>Home</a>"
                "</main></body></html>"
            )
            return HTMLResponse(page, status_code=404)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    async def _page_user(request: Request):
        return await request.app.state.accounts.user_from_token(cookie_token(request))

    @app.get("/")
    async def home(request: Request):
        user = await _page_user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        if not user.get("church_id"):
            return RedirectResponse("/onboarding", status_code=303)
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/login")
    async def login_page():
        return FileResponse(STATIC_DIR / "login.html")

    @app.get("/onboarding")
    async def onboarding_page(request: Request):
        user = await _page_user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        return FileResponse(STATIC_DIR / "onboarding.html")

    @app.get("/settings")
    async def settings_page(request: Request):
        user = await _page_user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        if not user.get("church_id"):
            return RedirectResponse("/onboarding", status_code=303)
        return FileResponse(STATIC_DIR / "settings.html")

    @app.get("/admin")
    async def admin_page(request: Request):
        user = await _page_user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        if user["role"] != "admin":
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC_DIR / "admin.html")

    @app.get("/o/{session_id}")
    async def operator_page(session_id: str, request: Request):
        director = request.app.state.director
        user = await _page_user(request)
        token = request.query_params.get("k", "")
        document = await director.session(session_id)
        if document is None and user and user.get("church_id"):
            document = await director.ensure_loaded(session_id, church_id=user["church_id"])
        if document is None and token:
            document = await director.ensure_loaded(session_id, token=token)
        if not document:
            raise HTTPException(404, "Session not found.")
        return FileResponse(STATIC_DIR / "operator.html")

    @app.get("/l/{session_id}")
    async def listener_page(session_id: str, request: Request):
        director = request.app.state.director
        document = await director.session(session_id)
        if document is None:
            document = await director.ensure_loaded(session_id, public=True)
        if not document:
            raise HTTPException(404, "Session not found.")
        return FileResponse(STATIC_DIR / "listener.html")

    @app.get("/v/{session_id}")
    async def overlay_page(session_id: str, request: Request):
        director = request.app.state.director
        document = await director.session(session_id)
        if document is None:
            document = await director.ensure_loaded(session_id, public=True)
        if not document:
            raise HTTPException(404, "Session not found.")
        return FileResponse(STATIC_DIR / "overlay.html")

    @app.middleware("http")
    async def revalidate(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
