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
        if language and language != "mix":
            options["language"] = language
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


_SPEECH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class _HttpxAuthResponse:
    def __init__(self, response: httpx.Response):
        self.status = response.status_code
        self.headers = response.headers
        self.data = response.content


class _HttpxAuthRequest:
    """Token refresh transport. google-auth's own transport needs the requests package."""

    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        del kwargs
        response = httpx.request(method, url, content=body, headers=headers, timeout=timeout or 30)
        return _HttpxAuthResponse(response)


def _load_google_credentials(path: str, project: str):
    """A service-account file or a gcloud user login. Both are JSON."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = raw.get("type")
    if kind == "service_account":
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(raw, scopes=[_SPEECH_SCOPE])
    elif kind == "authorized_user":
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_info(raw, scopes=[_SPEECH_SCOPE])
    else:
        raise RuntimeError("The Google credential file is neither a service account nor a user login.")
    if project and hasattr(creds, "with_quota_project"):
        creds = creds.with_quota_project(project)
    return creds


def speech_language_codes(language: str) -> list[str]:
    """Chirp picks the dominant language of a clip. Mixed stays inside Indonesian and English."""
    if language == "id":
        return ["id-ID"]
    if language == "mix":
        return ["id-ID", "en-US"]
    return ["en-US"]


def chirp_request(pcm: bytes, language: str, hotwords: str | None, word_times: bool = False) -> dict:
    """Speech-to-Text V2 body for one Chirp 3 clip.

    Word timestamps and phrase hints together make Chirp return "not found",
    so hints win. The pipeline estimates timing when the response has no words.
    """
    features: dict = {"enableAutomaticPunctuation": True}
    if word_times:
        features["enableWordTimeOffsets"] = True
    config: dict = {
        "explicitDecodingConfig": {
            "encoding": "LINEAR16",
            "sampleRateHertz": SAMPLE_RATE,
            "audioChannelCount": 1,
        },
        "languageCodes": speech_language_codes(language),
        "model": "chirp_3",
        "features": features,
    }
    phrases = speech_phrases(hotwords, limit=200)
    if phrases:
        config["adaptation"] = {
            "phraseSets": [{
                "inlinePhraseSet": {"phrases": [{"value": phrase} for phrase in phrases]},
            }]
        }
    return {"config": config, "content": base64.b64encode(pcm).decode("ascii")}


class GoogleRecognizer:
    """Speech-to-Text. Chirp 3 uses V2 and a service account. An API key stays on v1."""

    def __init__(
        self,
        api_key: str,
        project: str,
        region: str = "us",
        model: str = "chirp_3",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        credentials_path: str = "",
        token_provider=None,
    ):
        self.api_key = api_key
        self.project = project
        self.region = region or "us"
        self.model = model or "chirp_3"
        self.timeout = timeout
        self.credentials_path = credentials_path or ""
        self.token_provider = token_provider
        self.loaded = True
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._credentials = None
        self.billed_once = True

    def ensure_loaded(self) -> None:
        self.loaded = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _bearer(self) -> str | None:
        if self.token_provider is not None:
            token = self.token_provider()
            return token or None
        path = self.credentials_path
        if not path:
            return None
        try:
            from google.oauth2 import service_account  # noqa: F401
        except ImportError:
            logger.warning("google-auth is not installed, so Chirp 3 is unavailable")
            return None
        with self._lock:
            if self._credentials is None:
                self._credentials = _load_google_credentials(path, self.project)
            if not self._credentials.valid:
                self._credentials.refresh(_HttpxAuthRequest())
            return self._credentials.token

    def _chirp_ready(self) -> bool:
        return bool(self.model.startswith("chirp") and self.project and (self.token_provider or self.credentials_path))

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        samples = np.asarray(audio if audio is not None else [], dtype=np.float32)
        pcm = float_to_pcm16(samples)
        if not pcm:
            return Transcript([])
        if self._chirp_ready():
            try:
                return self._transcribe_chirp(samples, pcm, language, hotwords)
            except Exception:
                logger.exception("Chirp 3 failed for this clip; using the previous recognizer")
        return self._transcribe_v1(samples, pcm, language, hotwords)

    def _transcribe_chirp(self, samples, pcm: bytes, language: str, hotwords: str | None) -> Transcript:
        token = self._bearer()
        if not token:
            raise RuntimeError("Chirp 3 has no Google login")
        region = self.region
        url = (
            f"https://{region}-speech.googleapis.com/v2/projects/{self.project}"
            f"/locations/{region}/recognizers/_:recognize"
        )
        headers = {"Authorization": f"Bearer {token}", "x-goog-user-project": self.project}
        with self._lock:
            response = self._client.post(url, headers=headers, json=chirp_request(pcm, language, hotwords), timeout=self.timeout)
        if response.status_code == 400 and "WordTimeOffsets" in response.text:
            with self._lock:
                response = self._client.post(
                    url, headers=headers, json=chirp_request(pcm, language, hotwords, word_times=False), timeout=self.timeout
                )
        response.raise_for_status()
        duration = len(samples) / SAMPLE_RATE
        return Transcript(_words_from_google(response.json(), duration))

    def _transcribe_v1(self, samples, pcm: bytes, language: str, hotwords: str | None) -> Transcript:
        codes = speech_language_codes(language)
        model = self.model if self.model and not self.model.startswith("chirp") else "latest_long"
        phrases = speech_phrases(hotwords)
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": SAMPLE_RATE,
                "languageCode": codes[0],
                "model": model,
                "enableWordTimeOffsets": True,
                "enableAutomaticPunctuation": True,
            },
            "audio": {"content": base64.b64encode(pcm).decode("ascii")},
        }
        if len(codes) > 1:
            payload["config"]["alternativeLanguageCodes"] = codes[1:]
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
