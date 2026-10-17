"""Replay caption fixtures through the pipeline.

Synthetic silence and tone clips check that the pipeline does not invent captions.
Speech clips stay blocked until real sermon recordings are added. This harness does
not invent word-error or translation scores for missing audio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import wave

import numpy as np

from app.glossary import Glossary
from app.paths import FIXTURE_DIR, GLOSSARY_PATH, ROOT
from app.pipeline import PipelineConfig, SessionPipeline
from app.store import MemoryRepository
from app.stt import ScriptedRecognizer, WhisperRecognizer
from app.translator import GlossaryTranslator
from eval.scoring import chrf, wer

MANIFEST = FIXTURE_DIR / "manifest.json"


def ensure_synthetic_wavs() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_wav(FIXTURE_DIR / "silence-5s.wav", np.zeros(16000 * 5, dtype=np.float32))
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000 * 4) / 16000)).astype(np.float32)
    _write_wav(FIXTURE_DIR / "tone-music.wav", tone)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    ints = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(ints.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio


async def _feed(pipeline, audio: np.ndarray) -> None:
    chunk = 1600
    for start in range(0, len(audio), chunk):
        piece = audio[start : start + chunk]
        ints = np.clip(piece * 32767, -32768, 32767).astype("<i2")
        await pipeline.accept_pcm(ints.tobytes())
    await pipeline.accept_pcm(np.zeros(16000 // 2, dtype="<i2").tobytes())
    await pipeline.idle()


async def run(engine: str) -> dict:
    ensure_synthetic_wavs()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    glossary = Glossary.load(GLOSSARY_PATH)
    config = PipelineConfig(
        hangover_seconds=0.25,
        decode_interval_seconds=0.3,
        provisional_debounce_seconds=0.01,
        worship_seconds=1.5,
        capture_stall_seconds=30,
    )
    results = []
    for clip in manifest["clips"]:
        if clip.get("expect") == "pending":
            results.append({"id": clip["id"], "status": "blocked", "reason": "Real sermon audio has not been added.", "tags": clip["tags"]})
            continue
        recognizer = _recognizer(engine)
        repository = MemoryRepository()
        document = _document(clip["language"])
        pipeline = make_pipeline(document, repository, recognizer, glossary, config)
        await pipeline.start()
        audio = _read_wav(FIXTURE_DIR / clip["audio"])
        await _feed(pipeline, audio)
        banner = pipeline.doc.get("banner") or ""
        await pipeline.stop()
        hypothesis = " ".join(line["original"] for line in pipeline.doc["lines"] if not line.get("hidden"))
        translation = " ".join(line["translation"] for line in pipeline.doc["lines"] if not line.get("hidden"))
        latencies = [line["latency_ms"] for line in pipeline.doc["lines"] if line.get("latency_ms") is not None]
        results.append(
            {
                "id": clip["id"],
                "status": "ran",
                "engine": engine,
                "tags": clip["tags"],
                "wer": wer(clip.get("reference_transcript") or "", hypothesis),
                "chrf": chrf(clip.get("reference_translation") or "", translation),
                "latency_ms": latencies,
                "banner": banner,
                "recognizer_calls": getattr(recognizer, "calls", None),
                "hypothesis": hypothesis,
            }
        )
    glossary_text = "Grace Church is full of grace"
    protected = glossary.merged(["Grace Church"]).protect(glossary_text, "en")
    restored, missing = glossary.merged(["Grace Church"]).restore(protected.text, protected)
    return {
        "clips": results,
        "glossary_check": {"input": glossary_text, "output": restored, "missing": missing},
    }


def _recognizer(engine: str):
    if engine == "fake":
        return ScriptedRecognizer([])
    if engine == "whisper":
        recognizer = WhisperRecognizer("large-v3-turbo", "cpu", "int8")
        recognizer.ensure_loaded()
        return recognizer
    raise SystemExit(f"Unknown engine {engine}")


def _document(language: str) -> dict:
    return {
        "id": f"eval-{language}",
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


def make_pipeline(document, repository, recognizer, glossary, config):
    return SessionPipeline(document, repository, recognizer, GlossaryTranslator(), glossary, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["fake", "whisper"], default="fake")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "latest.json"))
    args = parser.parse_args()
    payload = asyncio.run(run(args.engine))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
