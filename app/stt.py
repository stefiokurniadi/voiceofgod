from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import threading

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
