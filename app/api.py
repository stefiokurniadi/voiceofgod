from __future__ import annotations

import asyncio
import io
import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import Response
import segno

from app.captions import lines_after, project_session
from app.export import render_srt, render_txt
from app.gpu import read_gpu

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
        "line": project_line(event.get("line"), role, tombstone=True) if event.get("line") else None,
        "provisional": project_line(event.get("provisional"), role, tombstone=True) if event.get("provisional") else None,
    }
    if role == "operator":
        projected["metrics"] = event.get("metrics") or {}
    return projected


def _catchup(document: dict, role: str, after: str) -> dict:
    projected = project_session(document, role)
    newer = lines_after(projected["lines"], after or None)
    if newer is None:
        projected["type"] = "snapshot"
        return projected
    projected["lines"] = newer
    projected["type"] = "resume" if after else "snapshot"
    return projected


def _base_url(request: Request) -> str:
    configured = request.app.state.settings.public_base_url.strip()
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


def _token(request: Request) -> str:
    return request.headers.get("x-operator-token", "")


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
    }


@router.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json()
    speaker_language = body.get("speaker_language", "en")
    do_not_translate = body.get("do_not_translate") or []
    terms = body.get("terms") or []
    if not isinstance(do_not_translate, list) or not isinstance(terms, list):
        raise HTTPException(400, "Glossary fields must be lists.")
    try:
        created = await request.app.state.director.create_session(
            speaker_language,
            [str(item) for item in do_not_translate],
            terms,
            _base_url(request),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return created


@router.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str, request: Request):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
    try:
        await request.app.state.director.start(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str, request: Request):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
    try:
        await request.app.state.director.stop(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    return {"ok": True}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
    await request.app.state.director.delete(session_id)
    return {"ok": True}


@router.post("/api/sessions/{session_id}/language")
async def set_language(session_id: str, request: Request):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
    body = await request.json()
    language = body.get("speaker_language")
    if language not in {"en", "id"}:
        raise HTTPException(400, "Speaker language must be English or Bahasa Indonesia.")
    pipeline = request.app.state.director.pipeline(session_id)
    if pipeline is None:
        raise HTTPException(404, "Session not found.")
    await pipeline.set_language(language)
    return {"ok": True}


@router.post("/api/sessions/{session_id}/lines/{line_id}/hide")
async def hide_line(session_id: str, line_id: str, request: Request):
    return await _line_action(request, session_id, line_id, "hide")


@router.post("/api/sessions/{session_id}/lines/{line_id}/restore")
async def restore_line(session_id: str, line_id: str, request: Request):
    return await _line_action(request, session_id, line_id, "restore")


async def _line_action(request: Request, session_id: str, line_id: str, action: str):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
    pipeline = request.app.state.director.pipeline(session_id)
    if pipeline is None:
        raise HTTPException(404, "Session not found.")
    found = await (pipeline.hide(line_id) if action == "hide" else pipeline.restore(line_id))
    if not found:
        raise HTTPException(404, "Caption not found.")
    return {"ok": True}


@router.get("/api/sessions/{session_id}/captions")
async def listener_captions(session_id: str, request: Request):
    document = await request.app.state.director.session(session_id)
    if not document or document.get("status") == "deleted":
        raise HTTPException(404, "Session not found.")
    return _catchup(document, "listener", request.query_params.get("after", ""))


@router.get("/api/sessions/{session_id}/glossary")
async def get_glossary(session_id: str, request: Request):
    document = await request.app.state.director.authorize(session_id, _token(request))
    if not document:
        raise HTTPException(401, "Operator token is not valid.")
    return document.get("glossary") or {"terms": [], "do_not_translate": []}


@router.put("/api/sessions/{session_id}/glossary")
async def put_glossary(session_id: str, request: Request):
    if not await request.app.state.director.authorize(session_id, _token(request)):
        raise HTTPException(401, "Operator token is not valid.")
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
    document = await request.app.state.director.authorize(session_id, _token(request))
    if not document:
        raise HTTPException(401, "Operator token is not valid.")
    metrics = dict(document.get("metrics") or {})
    metrics["gpu"] = read_gpu()
    return metrics


@router.get("/api/sessions/{session_id}/export.txt")
async def export_txt(session_id: str, request: Request):
    document = await _export_document(request, session_id)
    projected = project_session(document, "listener")
    body = render_txt(projected["lines"], document["speaker_language"])
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
    document = await request.app.state.director.authorize(session_id, token)
    if not document:
        raise HTTPException(401, "Operator token is not valid.")
    return document


@router.get("/api/sessions/{session_id}")
async def operator_session(session_id: str, request: Request):
    document = await request.app.state.director.authorize(session_id, _token(request))
    if not document:
        raise HTTPException(401, "Operator token is not valid.")
    return project_session(document, "operator")


@router.get("/api/sessions/{session_id}/qr.svg")
async def session_qr(session_id: str, request: Request):
    token = request.query_params.get("k", "")
    document = await request.app.state.director.authorize(session_id, token)
    if not document:
        raise HTTPException(401, "Operator token is not valid.")
    listener_url = f"{_base_url(request)}/l/{session_id}"
    qr = segno.make(listener_url, error="m")
    buffer = io.BytesIO()
    qr.save(buffer, kind="svg", scale=4, border=1, xmldecl=False)
    return Response(buffer.getvalue(), media_type="image/svg+xml")


@router.websocket("/ws/listen/{session_id}")
async def listen(websocket: WebSocket, session_id: str):
    document = await websocket.app.state.director.session(session_id)
    if not document:
        await websocket.close(code=4404)
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
    token = websocket.query_params.get("k", "")
    document = await websocket.app.state.director.authorize(session_id, token)
    if not document:
        await websocket.close(code=4401)
        return
    pipeline = websocket.app.state.director.pipeline(session_id)
    if pipeline is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    repository = websocket.app.state.repository

    async def pump_events():
        async with repository.subscribe(session_id) as subscription:
            current = await websocket.app.state.director.session(session_id)
            payload = _catchup(current, "operator", websocket.query_params.get("after", ""))
            await websocket.send_json(payload)
            seen = current.get("seq") or 0
            async for event in subscription:
                if (event.get("seq") or 0) <= seen:
                    continue
                seen = event["seq"]
                await websocket.send_json(_event_for(event, "operator"))
                if event.get("type") == "deleted":
                    break

    async def pump_audio():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                await pipeline.accept_pcm(message["bytes"])
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "capture" and payload.get("state") == "denied":
                    await pipeline._set_status("paused", "Microphone permission was denied")

    event_task = asyncio.create_task(pump_events())
    audio_task = asyncio.create_task(pump_audio())
    done, pending = await asyncio.wait({event_task, audio_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        try:
            task.result()
        except Exception:
            logger.info("operator socket closed for session %s", session_id)
