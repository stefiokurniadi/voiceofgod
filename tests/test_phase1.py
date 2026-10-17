from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.captions import project_session
from app.config import Settings
from app.director import Director
from app.glossary import Glossary
from app.main import create_app
from app.paths import GLOSSARY_PATH
from app.pipeline import PipelineConfig, SessionPipeline
from app.scripture import find_citations, is_long_reading, primary_citation
from app.store import MemoryRepository
from app.stt import GrowingRecognizer, ScriptedRecognizer
from app.translator import GlossaryTranslator, build_user_message, TranslationRequest
from eval.scoring import chrf, wer


FAST = PipelineConfig(
    hangover_seconds=0.25,
    decode_interval_seconds=0.2,
    max_hold_seconds=8.0,
    keep_tail_seconds=0.25,
    provisional_debounce_seconds=0.01,
    translate_timeout_seconds=1.0,
    worship_seconds=1.2,
    capture_stall_seconds=30,
    min_decode_seconds=0.15,
    min_sentence_words=4,
)


def pcm(seconds: float, amplitude: float, seed: int = 1) -> bytes:
    count = int(16000 * seconds)
    if amplitude == 0:
        samples = np.zeros(count, dtype=np.float32)
    else:
        samples = np.random.default_rng(seed).uniform(-amplitude, amplitude, count).astype(np.float32)
    return np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()


async def feed(pipeline: SessionPipeline, seconds: float, amplitude: float) -> None:
    remaining = seconds
    while remaining > 0.001:
        step = min(0.2, remaining)
        await pipeline.accept_pcm(pcm(step, amplitude))
        remaining -= step


def document(language: str = "en") -> dict:
    return {
        "id": "test-session",
        "speaker_language": language,
        "status": "ready",
        "banner": "",
        "seq": 0,
        "lines": [],
        "provisional": None,
        "line_counter": 0,
        "metrics": {},
        "development_translator": False,
    }


def pipeline_for(words: list[str], language: str = "en", config: PipelineConfig | None = None, dnt: list[str] | None = None):
    glossary = Glossary.load(GLOSSARY_PATH).merged(dnt or [])
    pipe = SessionPipeline(
        document(language),
        MemoryRepository(),
        ScriptedRecognizer(words),
        GlossaryTranslator(),
        glossary,
        config or FAST,
    )
    return pipe


async def speak(pipe: SessionPipeline, seconds: float = 1.0) -> None:
    await pipe.start()
    await feed(pipe, seconds, 0.25)
    await feed(pipe, 0.4, 0.0)
    await pipe.idle()


def test_glossary_protects_church_name_and_translates_the_word():
    glossary = Glossary.load(GLOSSARY_PATH).merged(["Grace Church"])
    protection = glossary.protect("Grace Church is full of grace", "en")
    assert "Grace Church" not in protection.text
    assert "grace" not in protection.text.casefold()
    restored, missing = glossary.restore(protection.text, protection)
    assert restored == "Grace Church is full of anugerah"
    assert missing == []


def test_glossary_longest_phrase_wins():
    glossary = Glossary([], ["Holy Spirit"]).merged(
        [],
        [{"en": "Holy Spirit", "id": "Roh Kudus"}, {"en": "Spirit", "id": "roh"}],
    )
    # merged() keys by English side, so add Spirit through a fresh glossary.
    glossary = Glossary(
        [],
        [],
    ).merged([], [{"en": "Holy Spirit", "id": "Roh Kudus"}, {"en": "Spirit", "id": "roh"}])
    protection = glossary.protect("The Holy Spirit gives spirit", "en")
    restored, _missing = glossary.restore(protection.text, protection)
    assert "Roh Kudus" in restored
    assert "roh" in restored


def test_indonesian_alias_maps_back_to_english():
    glossary = Glossary.load(GLOSSARY_PATH)
    protection = glossary.protect("ibadah dimulai", "id")
    restored, _missing = glossary.restore(protection.text, protection)
    assert restored == "worship service dimulai"


def test_citations_and_non_citations():
    john = primary_citation("Please open John 3:16 with me")
    assert john is not None
    assert john.display_en == "John 3:16"
    assert john.display_id == "Yohanes 3:16"
    assert primary_citation("In chapter 3 of my story we left") is None
    epistle = primary_citation("1 Yohanes 4:8")
    assert epistle is not None
    assert epistle.display_en == "1 John 4:8"
    assert find_citations("the budget is 3 to 1 today") == []


def test_long_reading_threshold():
    citation = primary_citation("John 3:16 " + " ".join(["alpha"] * 25))
    assert citation is not None
    assert is_long_reading("John 3:16 " + " ".join(["alpha"] * 25), citation)
    short = "Please turn to John 3:16"
    short_citation = primary_citation(short)
    assert short_citation is not None
    assert not is_long_reading(short, short_citation)


def test_prompt_includes_context_and_formal_rule():
    glossary = Glossary.load(GLOSSARY_PATH)
    request = TranslationRequest("grace", "en", glossary, [("Hello church", "Halo jemaat")])
    message = build_user_message(request, "[[T0]]")
    assert "Halo jemaat" in message
    assert "[[T0]]" in message
    assert "Bahasa Indonesia" in message


def test_wer_and_chrf():
    assert wer("", "") == 0
    assert wer("a b", "a b") == 0
    assert wer("a b", "a c") == 0.5
    assert wer("", "hello") == 1
    assert chrf("", "") is None
    assert chrf("anugerah", "anugerah") == pytest.approx(1)
    assert chrf("anugerah", "") == 0


@pytest.mark.asyncio
async def test_silence_does_not_call_recognizer_or_emit_captions():
    pipe = pipeline_for(["should", "not", "appear"])
    await pipe.start()
    await feed(pipe, 1.0, 0.0)
    await pipe.idle()
    assert pipe.recognizer.calls == 0
    assert pipe.doc["lines"] == []


@pytest.mark.asyncio
async def test_speech_finalises_and_applies_glossary():
    pipe = pipeline_for(["Grace", "Church", "is", "full", "of", "grace"], dnt=["Grace Church"])
    await speak(pipe)
    assert pipe.doc["lines"]
    line = pipe.doc["lines"][0]
    assert line["state"] == "final"
    assert "Grace Church" in line["original"]
    assert "anugerah" in line["translation"]
    assert "Grace Church" in line["translation"]
    assert line["t1"] >= line["t0"]


@pytest.mark.asyncio
async def test_final_line_does_not_change_when_the_speaker_continues():
    pipe = pipeline_for(["Hello", "church", "this", "morning"])
    await speak(pipe)
    first = pipe.doc["lines"][0]["original"]
    pipe.recognizer.words = ["A", "completely", "different", "sentence"]
    await feed(pipe, 1.0, 0.25)
    await feed(pipe, 0.4, 0.0)
    await pipe.idle()
    assert pipe.doc["lines"][0]["original"] == first
    assert len(pipe.doc["lines"]) >= 2


@pytest.mark.asyncio
async def test_music_becomes_a_banner_not_lyrics():
    pipe = pipeline_for([])
    await speak(pipe, seconds=1.6)
    assert pipe.doc["lines"] == []
    assert pipe.doc["banner"] == "Worship"
    assert pipe.recognizer.calls >= 1


@pytest.mark.asyncio
async def test_long_scripture_is_hidden_from_listeners():
    words = ["John", "3:16", *["alpha"] * 25]
    pipe = pipeline_for(words)
    await speak(pipe, seconds=1.2)
    assert pipe.doc["lines"]
    line = pipe.doc["lines"][0]
    assert line["suppressed"] is True
    assert "alpha" in line["original"]
    listener = project_session(pipe.doc, "listener")
    assert "alpha" not in listener["lines"][0]["original"]
    assert "John 3:16" in listener["lines"][0]["original"]
    await pipe.restore(line["id"])
    restored = project_session(pipe.doc, "listener")
    assert "alpha" in restored["lines"][0]["original"]


@pytest.mark.asyncio
async def test_hide_removes_the_line_for_listeners():
    pipe = pipeline_for(["Hello", "church", "this", "morning"])
    await speak(pipe)
    line_id = pipe.doc["lines"][0]["id"]
    assert await pipe.hide(line_id)
    listener = project_session(pipe.doc, "listener")
    assert listener["lines"] == []
    operator = project_session(pipe.doc, "operator")
    assert operator["lines"][0]["hidden"] is True


@pytest.mark.asyncio
async def test_sentence_split_keeps_the_tail():
    config = PipelineConfig(**{**FAST.__dict__, "decode_interval_seconds": 0.25, "min_sentence_words": 4})
    glossary = Glossary.load(GLOSSARY_PATH)
    recognizer = GrowingRecognizer(["Please", "open", "your", "Bible.", "Then", "we", "sing"])
    pipe = SessionPipeline(document(), MemoryRepository(), recognizer, GlossaryTranslator(), glossary, config)
    await pipe.start()
    await feed(pipe, 1.5, 0.25)
    await pipe.idle()
    await feed(pipe, 0.8, 0.25)
    await pipe.idle()
    finals = [line["original"] for line in pipe.doc["lines"]]
    assert any("Bible." in line for line in finals)
    assert all(not line.endswith("sing") or "Bible." not in line for line in finals)


@pytest.mark.asyncio
async def test_api_session_projects_listener_without_the_operator_secret():
    app = create_app(
        Settings(allow_fake_models=True, require_translator=False, require_redis=False, max_live_sessions=1),
        repository=MemoryRepository(),
        recognizer=ScriptedRecognizer(["Hello", "church", "this", "morning"]),
        translator=GlossaryTranslator(),
        pipeline_config=FAST,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/sessions", json={"speaker_language": "en", "do_not_translate": ["Grace Church"]})
        assert created.status_code == 200
        body = created.json()
        assert body["operator_secret"] not in body["listener_url"]
        denied = await client.get(f"/api/sessions/{body['id']}")
        assert denied.status_code == 401
        started = await client.post(
            f"/api/sessions/{body['id']}/start",
            headers={"X-Operator-Token": body["operator_secret"]},
        )
        assert started.status_code == 200
        pipe = app.state.director.pipeline(body["id"])
        await feed(pipe, 1.0, 0.25)
        await feed(pipe, 0.4, 0.0)
        await pipe.idle()
        captions = await client.get(f"/api/sessions/{body['id']}/captions")
        assert captions.status_code == 200
        payload = captions.json()
        assert "token_hash" not in payload
        assert payload["lines"]
        qr = await client.get(f"/api/sessions/{body['id']}/qr.svg", params={"k": body["operator_secret"]})
        assert qr.status_code == 200
        assert b"<svg" in qr.content
        hidden = await client.post(
            f"/api/sessions/{body['id']}/lines/{payload['lines'][0]['id']}/hide",
            headers={"X-Operator-Token": body["operator_secret"]},
        )
        assert hidden.status_code == 200
        again = await client.get(f"/api/sessions/{body['id']}/captions")
        assert again.json()["lines"] == []
