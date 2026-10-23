from __future__ import annotations

import asyncio
import io
import json
import logging

from datetime import date

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response
import segno

from app.accounts import COOKIE, AuthError, CHURCH_NOTE, cookie_token
from app.audio import clamp_input_boost
from app.stt import GoogleRecognizer, WhisperRecognizer, speech_model_label
from app.captions import lines_after, project_session
from app.director import token_matches
from app.export import render_srt, render_txt
from app.gpu import read_gpu
from app.propresenter import ProPresenterClient, propresenter_settings
from app.rtmp import allow_publish, allow_read, public_server, rtmp_ready, secret_matches, stream_session

logger = logging.getLogger(__name__)
router = APIRouter()

def _event_for(event: dict, role: str) -> dict:
    from app.captions import project_line

    projected = {
        "type": event.get("type"),
        "seq": event.get("seq") or 0,
        "status": event.get("status"),
        "banner": event.get("banner") or "",
        "speaker_language": event.get("speaker_language"),
        "translation_language": event.get("translation_language"),
        "voice": bool(event.get("voice")),
        "line": project_line(event.get("line"), role, tombstone=True) if event.get("line") else None,
        "provisional": project_line(event.get("provisional"), role, tombstone=True) if event.get("provisional") else None,
    }
    projected["service_name"] = event.get("service_name") or ""
    if role == "operator":
        projected["metrics"] = event.get("metrics") or {}
        projected["service_note"] = event.get("service_note") or ""
        projected["service_date"] = event.get("service_date") or ""
    return projected


def _attach_operator(payload: dict, document: dict, settings) -> None:
    token = document.get("ingest_token") or ""
    payload["input"] = document.get("input") or "browser"
    payload["input_boost"] = clamp_input_boost(document.get("input_boost", 2))
    payload["youtube_url"] = document.get("youtube_url") or ""
    payload["rtmp"] = {
        "available": rtmp_ready(settings),
        "server": public_server(settings),
        "stream_key": f"{document['id']}?token={token}" if token else "",
    }
    payload["propresenter"] = propresenter_settings(document.get("propresenter"))
    payload["overlay_note"] = (
        "A ProPresenter web layer or a browser can open the overlay. "
        "Add ?bg=black when the source cannot use transparency."
    )


def _catchup(document: dict, role: str, after: str, settings=None, recognizers=None) -> dict:
    projected = project_session(document, role)
    newer = lines_after(projected["lines"], after or None)
    if newer is None:
        projected["type"] = "snapshot"
    else:
        projected["lines"] = newer
        projected["type"] = "resume" if after else "snapshot"
    if role == "operator" and settings is not None:
        _attach_operator(projected, document, settings)
        projected["speech_model"] = _speech_model_for(document, recognizers)
    return projected


def _speech_model_for(document: dict, recognizers) -> str:
    if not recognizers:
        return ""
    name = document.get("stt_provider") or "default"
    recognizer = recognizers.get(name) or recognizers.get("default")
    return speech_model_label(recognizer) if recognizer is not None else ""


def _base_url(request: Request) -> str:
    configured = request.app.state.settings.public_base_url.strip()
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


async def _reject_socket(websocket: WebSocket, kind: str) -> None:
    """Accept first so the browser receives a message and stops reconnecting.

    Closing before accept becomes an HTTP 403, and the page treats that as a dropped connection.
    """
    await websocket.accept()
    await websocket.send_json({"type": kind})
    await websocket.close(code=1000)


def _token(request: Request) -> str:
    return request.headers.get("x-operator-token", "")


async def _current_user(source):
    accounts = source.app.state.accounts
    return await accounts.user_from_token(cookie_token(source))


def _login_cookie(response: Response, token: str, settings) -> None:
    secure = (settings.public_base_url or "").strip().lower().startswith("https://")
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=secure,
        max_age=30 * 24 * 60 * 60,
        path="/",
    )


async def _require_user(request: Request) -> dict:
    user = await _current_user(request)
    if user is None:
        raise HTTPException(401, "Sign in before continuing.")
    return user


async def _require_admin(request: Request) -> dict:
    user = await _require_user(request)
    if user["role"] != "admin":
        raise HTTPException(403, "Only an admin can do that.")
    return user


async def _operator_document(request: Request, session_id: str, token: str | None = None) -> dict:
    director = request.app.state.director
    document = await director.session(session_id)
    if not document or document.get("status") == "deleted":
        user = await _current_user(request)
        if user and user.get("church_id"):
            document = await director.ensure_loaded(session_id, church_id=user["church_id"])
        if not document or document.get("status") == "deleted":
            raise HTTPException(404, "This session is no longer on this server. Create a new one from the home page.")
    presented = _token(request) if token is None else token
    if token_matches(presented, document.get("token_hash", "")):
        return document
    user = await _current_user(request)
    if user and user.get("church_id") and str(user["church_id"]) == str(document.get("church_id") or ""):
        return document
    raise HTTPException(401, "Operator token is not valid.")


@router.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    redis_ok = True
    if settings.redis_url:
        try:
            document = await request.app.state.repository.load("__health__")
            del document
        except Exception:
            redis_ok = False
    translator_ok = await request.app.state.translator.healthy()
    recognizer = request.app.state.recognizer
    return {
        "ok": redis_ok and (translator_ok or not settings.require_translator),
        "redis": redis_ok,
        "translator": translator_ok,
        "whisper_loaded": bool(getattr(recognizer, "model", None) or getattr(recognizer, "loaded", False)),
        "development_translator": bool(getattr(request.app.state.translator, "development", False)),
        "live_sessions": request.app.state.director.live_count(),
    }


@router.post("/api/login")
async def login(request: Request):
    body = await request.json()
    try:
        token = await request.app.state.accounts.login(str(body.get("username") or ""), str(body.get("password") or ""))
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    user = await request.app.state.accounts.user_from_token(token)
    response = JSONResponse(_me(user, request))
    _login_cookie(response, token, request.app.state.settings)
    return response


@router.post("/api/logout")
async def logout(request: Request):
    await request.app.state.accounts.logout(cookie_token(request))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response


@router.get("/api/me")
async def me(request: Request):
    user = await _require_user(request)
    return _me(user, request)


def _speech_model(request: Request, document: dict) -> str:
    return _speech_model_for(document, request.app.state.recognizers)


def _standard_label(recognizer) -> str:
    if isinstance(recognizer, GoogleRecognizer):
        return "Google"
    if isinstance(recognizer, WhisperRecognizer):
        return "Whisper"
    return "Standard"


def _providers(request: Request) -> dict:
    recognizers = getattr(request.app.state, "recognizers", None) or {}
    speakers = getattr(request.app.state, "speakers", None) or {}
    return {
        "stt": [
            {"id": "default", "label": _standard_label(request.app.state.recognizer), "available": True},
            {"id": "soniox", "label": "Soniox", "available": "soniox" in recognizers},
        ],
        "tts": [
            {"id": "google", "label": "Google WaveNet", "available": True},
            {"id": "soniox", "label": "Soniox", "available": "soniox" in speakers},
        ],
    }


def _me(user: dict, request: Request) -> dict:
    return {
        "username": user["username"],
        "role": user["role"],
        "church_id": user.get("church_id"),
        "church_name": user.get("church_name") or "",
        "church_note": user.get("church_note") or "",
        "keep_names": user.get("keep_names") or [],
        "speaker_language": user["speaker_language"],
        "translation_language": user.get("translation_language") or "id",
        "tts_enabled": user["tts_enabled"],
        "stt_provider": user.get("stt_provider") or "default",
        "tts_provider": user.get("tts_provider") or "google",
        "providers": _providers(request),
        "church_template": CHURCH_NOTE,
    }


@router.post("/api/onboarding")
async def onboarding(request: Request):
    user = await _require_user(request)
    body = await request.json()
    try:
        saved = await request.app.state.accounts.save_church(
            user["id"],
            str(body.get("name") or ""),
            str(body.get("note") or ""),
            body.get("keep_names") or [],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await request.app.state.director.apply_church(saved["church_id"], saved["church_name"], saved["church_note"])
    return _me(saved, request)


@router.get("/api/settings")
async def get_settings(request: Request):
    user = await _require_user(request)
    return _me(user, request)


@router.put("/api/settings")
async def put_settings(request: Request):
    user = await _require_user(request)
    body = await request.json()
    language = body.get("speaker_language", user["speaker_language"])
    enabled = body.get("tts_enabled", user["tts_enabled"])
    caption = body.get("translation_language", user.get("translation_language") or "id")
    stt_provider = body.get("stt_provider", user.get("stt_provider") or "default")
    tts_provider = body.get("tts_provider", user.get("tts_provider") or "google")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "Spoken translation must be on or off.")
    if "stt_provider" in body and stt_provider == "soniox" and "soniox" not in request.app.state.recognizers:
        raise HTTPException(400, "Soniox needs an API key on this server.")
    if "tts_provider" in body and tts_provider == "soniox" and "soniox" not in request.app.state.speakers:
        raise HTTPException(400, "Soniox needs an API key on this server.")
    try:
        saved = await request.app.state.accounts.save_preferences(
            user["id"], language, enabled, caption, stt_provider, tts_provider
        )
        if saved.get("church_id"):
            await request.app.state.director.apply_speech(
                saved["church_id"], saved.get("stt_provider") or "default", saved.get("tts_provider") or "google"
            )
        if "name" in body or "note" in body or "keep_names" in body:
            saved = await request.app.state.accounts.save_church(
                user["id"],
                str(body.get("name") or saved.get("church_name") or ""),
                str(body.get("note") if body.get("note") is not None else saved.get("church_note") or ""),
                body.get("keep_names") if body.get("keep_names") is not None else saved.get("keep_names") or [],
            )
            await request.app.state.director.apply_church(saved["church_id"], saved["church_name"], saved["church_note"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _me(saved, request)


@router.get("/api/admin/youtube")
async def admin_youtube_status(request: Request):
    await _require_admin(request)
    return request.app.state.director.youtube_view(_base_url(request))


@router.post("/api/admin/youtube")
async def admin_youtube_start(request: Request):
    await _require_admin(request)
    body = await request.json()
    language = body.get("speaker_language") or "en"
    try:
        created = await request.app.state.director.create_youtube_test(
            str(body.get("url") or ""),
            str(language),
            _base_url(request),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        text = str(exc)
        status = 503 if "not ready" in text else 409
        raise HTTPException(status, text) from exc
    return created


@router.post("/api/admin/youtube/pause")
async def admin_youtube_pause(request: Request):
    await _require_admin(request)
    try:
        await request.app.state.director.pause_youtube_test()
    except KeyError:
        raise HTTPException(404, "No YouTube test is running.") from None
    return request.app.state.director.youtube_view(_base_url(request))


@router.post("/api/admin/youtube/resume")
async def admin_youtube_resume(request: Request):
    await _require_admin(request)
    try:
        await request.app.state.director.resume_youtube_test()
    except KeyError:
        raise HTTPException(404, "No YouTube test is running.") from None
    except RuntimeError as exc:
        text = str(exc)
        status = 503 if "not ready" in text else 409
        raise HTTPException(status, text) from exc
    return request.app.state.director.youtube_view(_base_url(request))


@router.post("/api/admin/youtube/stop")
async def admin_youtube_stop(request: Request):
    await _require_admin(request)
    try:
        await request.app.state.director.stop_youtube_test()
    except KeyError:
        raise HTTPException(404, "No YouTube test is running.") from None
    return {"ok": True}


@router.post("/api/admin/youtube/seek")
async def admin_youtube_seek(request: Request):
    await _require_admin(request)
    pipeline = request.app.state.director.active_youtube()
    if pipeline is None:
        raise HTTPException(404, "No YouTube test is running.")
    body = await request.json()
    return await _seek_youtube(request, pipeline.doc["id"], body)


@router.get("/api/admin/users")
async def admin_users(request: Request):
    await _require_admin(request)
    return {"users": await request.app.state.accounts.list_users()}


@router.post("/api/admin/users")
async def admin_create_user(request: Request):
    await _require_admin(request)
    body = await request.json()
    try:
        created = await request.app.state.accounts.create_user(str(body.get("username") or ""), str(body.get("password") or ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": created["id"], "username": created["username"]}


@router.post("/api/admin/users/{user_id}/password")
async def admin_reset_password(user_id: str, request: Request):
    await _require_admin(request)
    body = await request.json()
    try:
        await request.app.state.accounts.set_password(user_id, str(body.get("password") or ""))
    except KeyError:
        raise HTTPException(404, "User not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@router.post("/api/admin/users/{user_id}/disabled")
async def admin_disable_user(user_id: str, request: Request):
    await _require_admin(request)
    body = await request.json()
    if not isinstance(body.get("disabled"), bool):
        raise HTTPException(400, "Say whether the user is disabled.")
    try:
        await request.app.state.accounts.set_disabled(user_id, body["disabled"])
    except KeyError:
        raise HTTPException(404, "User not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@router.get("/api/services")
async def list_services(request: Request):
    user = await _require_user(request)
    if not user.get("church_id"):
        return {"services": []}
    return {"services": await request.app.state.accounts.list_services(user["church_id"])}


def _service_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise HTTPException(400, "The service date is not valid.") from exc


@router.post("/api/sessions")
async def create_session(request: Request):
    user = await _require_user(request)
    body = await request.json()
    speaker_language = body.get("speaker_language") or user["speaker_language"]
    translation_language = body.get("translation_language", user.get("translation_language"))
    voice = body.get("tts_enabled")
    if voice is None:
        voice = user["tts_enabled"]
    if not isinstance(voice, bool):
        raise HTTPException(400, "Spoken translation must be on or off.")
    do_not_translate = body.get("do_not_translate") or []
    terms = body.get("terms") or []
    if not isinstance(do_not_translate, list) or not isinstance(terms, list):
        raise HTTPException(400, "Glossary fields must be lists.")
    try:
        created = await request.app.state.director.create_session(
            user,
            str(body.get("name") or ""),
            str(body.get("note") or ""),
            _service_date(body.get("service_date")),
            speaker_language,
            voice,
            _base_url(request),
            [str(item) for item in do_not_translate],
            terms,
            translation_language,
        )
    except PermissionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return created


@router.put("/api/sessions/{session_id}")
async def update_session(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    voice = body.get("tts_enabled")
    if not isinstance(voice, bool):
        raise HTTPException(400, "Spoken translation must be on or off.")
    try:
        return await request.app.state.director.update_session(
            session_id,
            str(body.get("name") or ""),
            str(body.get("note") or ""),
            _service_date(body.get("service_date")),
            body.get("speaker_language") or "",
            voice,
            body.get("translation_language"),
        )
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str, request: Request):
    await _operator_document(request, session_id)
    try:
        await request.app.state.director.start(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    document = await request.app.state.director.session(session_id)
    return {"ok": True, "speech_model": _speech_model(request, document or {})}


@router.post("/api/sessions/{session_id}/pause")
async def pause_session(session_id: str, request: Request):
    await _operator_document(request, session_id)
    try:
        await request.app.state.director.pause(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    return {"ok": True}


@router.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str, request: Request):
    await _operator_document(request, session_id)
    try:
        await request.app.state.director.stop(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    return {"ok": True}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    await _operator_document(request, session_id)
    await request.app.state.director.delete(session_id)
    return {"ok": True}


@router.post("/api/sessions/{session_id}/language")
async def set_language(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    from app.translator import CAPTION_LANGUAGE_ERROR, SPEAKER_LANGUAGES, SPEAKER_LANGUAGE_ERROR

    language = body.get("speaker_language")
    if language not in SPEAKER_LANGUAGES:
        raise HTTPException(400, SPEAKER_LANGUAGE_ERROR)
    caption = body.get("translation_language")
    if language == "mix" and caption not in (None, "", "en", "id"):
        raise HTTPException(400, CAPTION_LANGUAGE_ERROR)
    pipeline = request.app.state.director.pipeline(session_id)
    if pipeline is None:
        raise HTTPException(404, "Session not found.")
    await pipeline.set_language(language, caption)
    await request.app.state.director.remember(session_id)
    return {"ok": True}


@router.post("/api/sessions/{session_id}/voice")
async def set_voice(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    enabled = body.get("tts_enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "Spoken translation must be on or off.")
    try:
        saved = await request.app.state.director.set_service_voice(session_id, enabled)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    return {"tts_enabled": saved}


@router.post("/api/sessions/{session_id}/lines/{line_id}/hide")
async def hide_line(session_id: str, line_id: str, request: Request):
    return await _line_action(request, session_id, line_id, "hide")


@router.post("/api/sessions/{session_id}/lines/{line_id}/restore")
async def restore_line(session_id: str, line_id: str, request: Request):
    return await _line_action(request, session_id, line_id, "restore")


async def _line_action(request: Request, session_id: str, line_id: str, action: str):
    await _operator_document(request, session_id)
    pipeline = request.app.state.director.pipeline(session_id)
    if pipeline is None:
        raise HTTPException(404, "Session not found.")
    found = await (pipeline.hide(line_id) if action == "hide" else pipeline.restore(line_id))
    if not found:
        raise HTTPException(404, "Caption not found.")
    return {"ok": True}


@router.get("/api/sessions/{session_id}/captions")
async def listener_captions(session_id: str, request: Request):
    director = request.app.state.director
    document = await director.session(session_id)
    if not document:
        document = await director.ensure_loaded(session_id, public=True)
    if not document or document.get("status") == "deleted":
        raise HTTPException(404, "Session not found.")
    return _catchup(document, "listener", request.query_params.get("after", ""))


@router.get("/api/sessions/{session_id}/audio/{line_id}")
async def line_audio(session_id: str, line_id: str, request: Request):
    pipeline = request.app.state.director.pipeline(session_id)
    clip = pipeline.clips.get(line_id) if pipeline else None
    if not clip:
        raise HTTPException(404, "This line has no audio.")
    return Response(content=clip, media_type="audio/mpeg")


@router.get("/api/sessions/{session_id}/glossary")
async def get_glossary(session_id: str, request: Request):
    document = await _operator_document(request, session_id)
    return document.get("glossary") or {"terms": [], "do_not_translate": []}


@router.put("/api/sessions/{session_id}/input")
async def put_input(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    try:
        source = await request.app.state.director.set_input(
            session_id, str(body.get("input") or ""), str(body.get("url") or "")
        )
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    document = await request.app.state.director.session(session_id)
    return {"input": source, "youtube_url": (document or {}).get("youtube_url") or ""}


@router.put("/api/sessions/{session_id}/input-boost")
async def put_input_boost(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Input boost has to be a number.")
    try:
        gain = await request.app.state.director.set_input_boost(session_id, body.get("input_boost"))
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    return {"input_boost": gain}


@router.put("/api/sessions/{session_id}/propresenter")
async def put_propresenter(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    try:
        saved = await request.app.state.director.set_propresenter(session_id, body if isinstance(body, dict) else {})
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return saved


@router.post("/api/sessions/{session_id}/propresenter/test")
async def test_propresenter(session_id: str, request: Request):
    document = await _operator_document(request, session_id)
    settings = propresenter_settings(document.get("propresenter"))
    if not settings["base_url"] or not settings["message"]:
        raise HTTPException(400, "Save a ProPresenter address and message name first.")
    client = ProPresenterClient(settings["base_url"], settings["message"], settings["token"])
    try:
        message_id = await client.resolve()
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "ProPresenter did not answer.") from exc
    finally:
        await client.aclose()
    return {"ok": True, "id": message_id}


@router.post("/api/sessions/{session_id}/youtube/pause")
async def session_youtube_pause(session_id: str, request: Request):
    await _operator_document(request, session_id)
    try:
        await request.app.state.director.pause_youtube_test(session_id)
    except KeyError:
        raise HTTPException(404, "No YouTube test is running.") from None
    pipeline = request.app.state.director.pipeline(session_id)
    return request.app.state.director.youtube_position(pipeline) if pipeline else {}


@router.post("/api/sessions/{session_id}/youtube/resume")
async def session_youtube_resume(session_id: str, request: Request):
    await _operator_document(request, session_id)
    try:
        return await request.app.state.director.resume_youtube_test(session_id)
    except KeyError:
        raise HTTPException(404, "No YouTube test is running.") from None
    except RuntimeError as exc:
        text = str(exc)
        status = 503 if "not ready" in text else 409
        raise HTTPException(status, text) from exc


@router.post("/api/sessions/{session_id}/youtube/seek")
async def session_youtube_seek(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    return await _seek_youtube(request, session_id, body)


async def _seek_youtube(request: Request, session_id: str, body: dict) -> dict:
    value = body.get("seconds") if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, "The time has to be a number of seconds.")
    try:
        return await request.app.state.director.seek_youtube(session_id, float(value))
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/sessions/{session_id}/level")
async def session_level(session_id: str, request: Request):
    await _operator_document(request, session_id)
    pipeline = request.app.state.director.pipeline(session_id)
    payload = {"level": float(pipeline.input_level) if pipeline is not None else 0.0}
    if pipeline is not None and pipeline.doc.get("input") == "youtube":
        payload.update(request.app.state.director.youtube_position(pipeline))
    return payload


@router.post("/internal/rtmp-auth/{secret}")
async def rtmp_auth(secret: str, request: Request):
    settings = request.app.state.settings
    if not secret_matches(secret, settings.rtmp_auth_secret):
        raise HTTPException(404, "Not found.")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(401, "Denied.") from exc
    if not isinstance(body, dict):
        raise HTTPException(401, "Denied.")
    session_id, token = stream_session(str(body.get("path") or ""), str(body.get("query") or ""))
    document = await request.app.state.director.session(session_id) if session_id else None
    if not document or document.get("status") == "deleted":
        expected = None
        exists = False
    else:
        expected = document.get("ingest_token") or ""
        exists = True
    action = str(body.get("action") or "")
    allowed = False
    if action == "publish":
        allowed = allow_publish(token, expected)
    elif action == "read":
        allowed = allow_read(str(body.get("user") or ""), str(body.get("password") or ""), settings.rtmp_read_password, exists)
    if not allowed:
        raise HTTPException(401, "Denied.")
    return Response(status_code=200)


@router.put("/api/sessions/{session_id}/glossary")
async def put_glossary(session_id: str, request: Request):
    await _operator_document(request, session_id)
    body = await request.json()
    try:
        saved = await request.app.state.director.update_glossary(
            session_id,
            body.get("terms") or [],
            body.get("do_not_translate") or [],
        )
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return saved


@router.get("/api/sessions/{session_id}/metrics")
async def session_metrics(session_id: str, request: Request):
    document = await _operator_document(request, session_id)
    metrics = dict(document.get("metrics") or {})
    metrics["gpu"] = read_gpu()
    return metrics


@router.get("/api/sessions/{session_id}/export.txt")
async def export_txt(session_id: str, request: Request):
    document = await _export_document(request, session_id)
    projected = project_session(document, "listener")
    body = render_txt(projected["lines"], document["speaker_language"], document.get("translation_language"))
    return Response(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.txt"'},
    )


@router.get("/api/sessions/{session_id}/export.srt")
async def export_srt(session_id: str, request: Request):
    document = await _export_document(request, session_id)
    projected = project_session(document, "listener")
    body = render_srt(projected["lines"])
    return Response(
        body,
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.srt"'},
    )


async def _export_document(request: Request, session_id: str) -> dict:
    token = _token(request) or request.query_params.get("k", "")
    return await _operator_document(request, session_id, token)


@router.get("/api/sessions/{session_id}")
async def operator_session(session_id: str, request: Request):
    document = await _operator_document(request, session_id)
    payload = project_session(document, "operator")
    payload["speech_model"] = _speech_model(request, document)
    _attach_operator(payload, document, request.app.state.settings)
    return payload


@router.get("/api/sessions/{session_id}/qr.svg")
async def session_qr(session_id: str, request: Request):
    document = await _operator_document(request, session_id, request.query_params.get("k", ""))
    listener_url = f"{_base_url(request)}/l/{session_id}"
    qr = segno.make(listener_url, error="m")
    buffer = io.BytesIO()
    qr.save(buffer, kind="svg", scale=4, border=1, xmldecl=False)
    return Response(buffer.getvalue(), media_type="image/svg+xml")


@router.websocket("/ws/listen/{session_id}")
async def listen(websocket: WebSocket, session_id: str):
    director = websocket.app.state.director
    document = await director.session(session_id)
    if not document:
        document = await director.ensure_loaded(session_id, public=True)
    if not document:
        await _reject_socket(websocket, "deleted")
        return
    await websocket.accept()
    await websocket.app.state.director.note_listeners(session_id, 1)
    repository = websocket.app.state.repository
    try:
        async with repository.subscribe(session_id) as subscription:
            current = await websocket.app.state.director.session(session_id)
            if current:
                payload = _catchup(current, "listener", websocket.query_params.get("after", ""))
                await websocket.send_json(payload)
                seen = current.get("seq") or 0
            else:
                seen = 0
            async for event in subscription:
                if (event.get("seq") or 0) <= seen:
                    continue
                seen = event["seq"]
                if event.get("type") == "deleted":
                    await websocket.send_json({"type": "deleted", "seq": seen})
                    break
                await websocket.send_json(_event_for(event, "listener"))
    except Exception:
        logger.info("listener disconnected from session %s", session_id)
    finally:
        await websocket.app.state.director.note_listeners(session_id, -1)


@router.websocket("/ws/operate/{session_id}")
async def operate(websocket: WebSocket, session_id: str):
    director = websocket.app.state.director
    token = websocket.query_params.get("k", "")
    document = await director.authorize(session_id, token) if token else None
    if document is None and token:
        document = await director.ensure_loaded(session_id, token=token)
    if document is None:
        user = await websocket.app.state.accounts.user_from_token(cookie_token(websocket))
        if user and user.get("church_id"):
            document = await director.ensure_loaded(session_id, church_id=user["church_id"])
    if not document:
        exists = await director.session(session_id)
        if exists is None and director.accounts is not None:
            row = await director.accounts.load_service(session_id)
            exists = row if row and row["status"] != "deleted" else None
        await _reject_socket(websocket, "unauthorized" if exists else "deleted")
        return
    pipeline = websocket.app.state.director.pipeline(session_id)
    if pipeline is None:
        await _reject_socket(websocket, "deleted")
        return
    await websocket.accept()
    repository = websocket.app.state.repository
    send_lock = asyncio.Lock()

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def pump_events():
        async with repository.subscribe(session_id) as subscription:
            current = await websocket.app.state.director.session(session_id)
            payload = _catchup(
                current,
                "operator",
                websocket.query_params.get("after", ""),
                websocket.app.state.settings,
                websocket.app.state.recognizers,
            )
            await send_json(payload)
            seen = current.get("seq") or 0
            async for event in subscription:
                if (event.get("seq") or 0) <= seen:
                    continue
                seen = event["seq"]
                await send_json(_event_for(event, "operator"))
                if event.get("type") == "deleted":
                    break

    async def pump_audio():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                if pipeline.doc.get("input") == "browser":
                    await pipeline.accept_pcm(message["bytes"])
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "capture" and payload.get("state") == "denied":
                    await pipeline._set_status("paused", "Microphone permission was denied")

    async def pump_monitor():
        queue = pipeline.attach_monitor()
        try:
            while True:
                chunk = await queue.get()
                if pipeline.doc.get("input") != "youtube":
                    continue
                async with send_lock:
                    await websocket.send_bytes(chunk)
        finally:
            pipeline.detach_monitor(queue)

    event_task = asyncio.create_task(pump_events())
    audio_task = asyncio.create_task(pump_audio())
    monitor_task = asyncio.create_task(pump_monitor())
    done, pending = await asyncio.wait({event_task, audio_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        try:
            task.result()
        except Exception:
            logger.info("operator socket closed for session %s", session_id)
