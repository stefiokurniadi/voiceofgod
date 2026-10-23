from __future__ import annotations

from dataclasses import dataclass
import base64
import inspect
import json
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


SERMON_PHRASES_EN = (
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
SERMON_PHRASES_ID = (
    "Roh Kudus",
    "firman Tuhan",
    "Tuhan Yesus",
    "Bapa",
    "Alkitab",
    "penghiburan",
    "pengharapan",
    "saudara-saudara",
    "jemaat",
    "syiar",
    "walaupun",
    "mungkin kalian",
    "puji Tuhan",
    "haleluya",
)


def sermon_hints(language: str) -> tuple[str, ...]:
    """Hints in the preacher's language. Mixed sermons hear both."""
    if language == "id":
        return SERMON_PHRASES_ID
    if language == "mix":
        return SERMON_PHRASES_ID + SERMON_PHRASES_EN
    return SERMON_PHRASES_EN


def speech_phrases(hotwords: str | None, limit: int = 40, language: str = "en") -> list[str]:
    """Church names first, then the sermon phrases, and never past the recognizer's limit."""
    custom = []
    for phrase in (hotwords or "").split(","):
        text = phrase.strip()
        if text and text not in custom:
            custom.append(text)
    reserved = [hint for hint in sermon_hints(language) if hint not in custom]
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
    phrases = speech_phrases(hotwords, limit=200, language=language)
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
        self._client = client or httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
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
        phrases = speech_phrases(hotwords, language=language)
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


SONIOX_STT_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


def soniox_language_hints(language: str) -> list[str]:
    """Soniox takes short language codes. Mixed stays inside Indonesian and English."""
    if language == "id":
        return ["id"]
    if language == "mix":
        return ["id", "en"]
    return ["en"]


def soniox_request(model: str, language: str, hotwords: str | None) -> dict:
    """One real-time Soniox config for a raw 16 kHz clip."""
    config = {
        "model": model or "stt-rt-v5",
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "language_hints": soniox_language_hints(language),
    }
    phrases = speech_phrases(hotwords, language=language)
    if phrases:
        config["context"] = {"terms": phrases}
    return config


def _millis(value) -> float:
    try:
        return max(0.0, float(value)) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def words_from_soniox(tokens: list[dict]) -> list[Word]:
    """Join sub-word tokens. A token that starts with a space begins a new word."""
    words: list[Word] = []
    text = ""
    start: float | None = None
    end = 0.0

    def flush() -> None:
        nonlocal text, start, end
        cleaned = text.strip()
        if cleaned and start is not None:
            words.append(Word(cleaned, start, max(end, start)))
        text = ""
        start = None

    for token in tokens:
        piece = token.get("text") or ""
        if not piece:
            continue
        if piece[:1].isspace():
            flush()
            piece = piece.lstrip()
            if not piece:
                continue
        token_start = _millis(token.get("start_ms"))
        token_end = _millis(token.get("end_ms"))
        if start is None:
            start = token_start
        if token_end:
            end = token_end
        text += piece
    flush()
    if words and reject_hallucination(" ".join(word.text for word in words)):
        return []
    return words


class SonioxRecognizer:
    """One Soniox real-time session per clip. Final tokens become the transcript."""

    def __init__(self, api_key: str, model: str = "stt-rt-v5", timeout: float = 30.0, connect=None):
        self.api_key = api_key
        self.model = model or "stt-rt-v5"
        self.timeout = timeout
        self.connect = connect
        self.loaded = True
        self.billed_once = True
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        self.loaded = True

    def close(self) -> None:
        return None

    def transcribe(self, audio, language: str, hotwords: str | None = None) -> Transcript:
        samples = np.asarray(audio if audio is not None else [], dtype=np.float32)
        pcm = float_to_pcm16(samples)
        if not pcm:
            return Transcript([])
        config = soniox_request(self.model, language, hotwords)
        with self._lock:
            tokens = self._transcribe_locked(pcm, config)
        return Transcript(words_from_soniox(tokens))

    def _transcribe_locked(self, pcm: bytes, config: dict) -> list[dict]:
        connect = self.connect
        if connect is None:
            from websockets.sync.client import connect as default_connect

            connect = default_connect
        tokens: list[dict] = []
        with connect(
            SONIOX_STT_URL,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=self.timeout,
            close_timeout=5,
        ) as ws:
            ws.send(json.dumps(config))
            # Soniox reads raw PCM as a stream of binary frames. An empty text
            # frame, not an empty binary frame, tells it the clip has ended.
            for start in range(0, len(pcm), 3840):
                ws.send(pcm[start:start + 3840])
            ws.send("")
            while True:
                message = ws.recv(timeout=self.timeout)
                body = json.loads(message)
                if body.get("error_code") is not None:
                    raise RuntimeError(f"Soniox {body.get('error_code')}: {body.get('error_message')}")
                for token in body.get("tokens") or []:
                    if token.get("is_final") and token.get("text"):
                        tokens.append(token)
                if body.get("finished"):
                    break
        return tokens


def speech_model_label(recognizer) -> str:
    """The model that hears the service, named the way the operator page shows it."""
    if isinstance(recognizer, GoogleRecognizer):
        if recognizer._chirp_ready():
            return "Google Chirp 3"
        name = recognizer.model if recognizer.model and not recognizer.model.startswith("chirp") else "latest_long"
        return f"Google {name}"
    if isinstance(recognizer, SonioxRecognizer):
        return f"Soniox {recognizer.model}"
    if isinstance(recognizer, WhisperRecognizer):
        return f"Whisper {recognizer.model_name}"
    return "Development"
