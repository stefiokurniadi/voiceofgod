from __future__ import annotations

from dataclasses import dataclass
import base64
import inspect
import logging
import threading

import httpx
import numpy as np

from app.audio import SAMPLE_RATE, float_to_pcm16
from app.captions import Transcript, Word

logger = logging.getLogger(__name__)

HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "please subscribe",
    "please subscribe to my channel",
    "subtitles by the amara.org community",
    "amara.org",
    "www.amara.org",
    "mbc 뉴스",
    "ご視聴ありがとうございました",
    "music",
    "applause",
    "[music]",
    "[applause]",
    "(music)",
    "(applause)",
}


def reject_hallucination(text: str) -> bool:
    cleaned = " ".join(text.replace("♪", " ").split()).casefold().strip(" .!?,")
    if not cleaned:
        return True
    return cleaned in HALLUCINATIONS


class ScriptedRecognizer:
    """Test and development recognizer. It does not load a model."""

    def __init__(self, words: list[str] | None = None, empty: bool = False):
        self.words = [] if empty else list(words or [])
        self.calls = 0
        self.loaded = True

    def ensure_loaded(self) -> None:
        self.loaded = True

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        del language, hotwords
        self.calls += 1
        duration = len(audio) / 16000
        if duration < 0.2 or not self.words:
            return Transcript([])
        if duration >= 0.8:
            count = len(self.words)
        else:
            count = max(1, int(len(self.words) * duration / 0.8))
        step = duration / count
        return Transcript(
            [Word(self.words[index], start=index * step, end=min(duration, (index + 1) * step)) for index in range(count)]
        )


class GrowingRecognizer(ScriptedRecognizer):
    """Reveals more of a fixed sentence as the buffer gets longer."""

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        del language, hotwords
        self.calls += 1
        duration = len(audio) / 16000
        if duration <= 0 or not self.words:
            return Transcript([])
        count = min(len(self.words), max(1, int(duration / 0.35)))
        step = duration / count
        return Transcript(
            [Word(self.words[index], start=index * step, end=(index + 1) * step) for index in range(count)]
        )


class WhisperRecognizer:
    def __init__(self, model_name: str, device: str, compute_type: str, download_root: str = ""):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root or None
        self.model = None
        self._supports_hotwords = False
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("loading whisper model %s on %s", self.model_name, self.device)
        kwargs = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if self.download_root:
            kwargs["download_root"] = self.download_root
        self.model = WhisperModel(self.model_name, **kwargs)
        self._supports_hotwords = "hotwords" in inspect.signature(self.model.transcribe).parameters
        logger.info("whisper model ready")

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        self.ensure_loaded()
        with self._lock:
            return self._transcribe_locked(audio, language, hotwords)

    def _transcribe_locked(self, audio, language: str, hotwords: str | None) -> Transcript:
        from faster_whisper.vad import VadOptions

        options = {
            "language": language,
            "task": "transcribe",
            "beam_size": 1,
            "temperature": 0.0,
            "vad_filter": True,
            "vad_parameters": VadOptions(min_silence_duration_ms=300),
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 0.5,
        }
        if hotwords and self._supports_hotwords:
            options["hotwords"] = hotwords
        segments, _info = self.model.transcribe(audio, **options)
        words: list[Word] = []
        for segment in segments:
            if reject_hallucination(segment.text or ""):
                continue
            segment_words = list(segment.words or [])
            if segment_words:
                for word in segment_words:
                    text = (word.word or "").strip()
                    if text:
                        words.append(Word(text, float(word.start), float(word.end)))
            else:
                text = (segment.text or "").strip()
                if text:
                    words.append(Word(text, float(segment.start), float(segment.end)))
        if words and reject_hallucination(" ".join(word.text for word in words)):
            return Transcript([])
        return Transcript(words)


def _duration_seconds(value) -> float:
    text = str(value or "").strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        return max(0.0, float(text))
    except ValueError:
        return 0.0


SERMON_PHRASES = (
    "grace of God",
    "with you today",
    "is with you",
    "is preaching",
    "preaching from",
    "Holy Spirit",
    "let us pray",
    "after the service",
    "quiet hearts",
    "this Sunday morning",
    "next Sunday",
    "open the word",
    "Brothers and sisters",
    "salvation is a gift",
    "own strength",
    "heavy heart",
    "verse sixteen",
    "blessing of the cross",
    "for the first time",
    "the congregation is listening",
)


def speech_phrases(hotwords: str | None, limit: int = 40) -> list[str]:
    """Church names first, then the sermon phrases, and never past the recognizer's limit."""
    custom = []
    for phrase in (hotwords or "").split(","):
        text = phrase.strip()
        if text and text not in custom:
            custom.append(text)
    reserved = [hint for hint in SERMON_PHRASES if hint not in custom]
    room = max(0, limit - len(reserved))
    chosen = custom[:room]
    for hint in reserved:
        if hint not in chosen:
            chosen.append(hint)
    return chosen[:limit]


def _words_from_google(body: dict, duration: float) -> list[Word]:
    words: list[Word] = []
    for result in body.get("results") or []:
        alternatives = result.get("alternatives") or []
        if not alternatives:
            continue
        alternative = alternatives[0]
        for info in alternative.get("words") or []:
            text = (info.get("word") or "").strip()
            if text:
                start = info.get("startTime", info.get("startOffset"))
                end = info.get("endTime", info.get("endOffset"))
                words.append(Word(text, _duration_seconds(start), _duration_seconds(end)))
        if words:
            continue
        transcript = (alternative.get("transcript") or "").strip()
        if not transcript or reject_hallucination(transcript):
            continue
        pieces = transcript.split()
        if not pieces:
            continue
        step = duration / len(pieces) if duration else 0.0
        for index, piece in enumerate(pieces):
            words.append(Word(piece, index * step, (index + 1) * step))
    if words and reject_hallucination(" ".join(word.text for word in words)):
        return []
    return words


class GoogleRecognizer:
    """Speech-to-Text v1. Chirp 3 rejects an API key, so the key path uses latest_long."""

    def __init__(
        self,
        api_key: str,
        project: str,
        region: str = "us",
        model: str = "chirp_3",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.project = project
        self.region = region or "us"
        self.model = model or "chirp_3"
        self.timeout = timeout
        self.loaded = True
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._lock = threading.Lock()
        self.billed_once = True

    def ensure_loaded(self) -> None:
        self.loaded = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        samples = np.asarray(audio if audio is not None else [], dtype=np.float32)
        pcm = float_to_pcm16(samples)
        if not pcm:
            return Transcript([])
        language_code = "id-ID" if language == "id" else "en-US"
        model = self.model if self.model and not self.model.startswith("chirp") else "latest_long"
        phrases = speech_phrases(hotwords)
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": SAMPLE_RATE,
                "languageCode": language_code,
                "model": model,
                "enableWordTimeOffsets": True,
                "enableAutomaticPunctuation": True,
            },
            "audio": {"content": base64.b64encode(pcm).decode("ascii")},
        }
        if phrases:
            payload["config"]["speechContexts"] = [{"phrases": phrases[:40], "boost": 12}]
        with self._lock:
            response = self._client.post(
                "https://speech.googleapis.com/v1/speech:recognize",
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
        response.raise_for_status()
        duration = len(samples) / SAMPLE_RATE
        return Transcript(_words_from_google(response.json(), duration))
