from __future__ import annotations

import base64

import httpx

from app.translator import target_language

SONIOX_TTS_URL = "https://tts-rt.soniox.com/tts"


class GoogleSpeaker:
    """One WaveNet clip per final listener line. Bytes stay in memory."""

    def __init__(
        self,
        api_key: str,
        voice_en: str = "en-US-Wavenet-D",
        voice_id: str = "id-ID-Wavenet-B",
        timeout: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.voices = {"en": voice_en, "id": voice_id}
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str, language: str) -> bytes:
        spoken = " ".join(text.split())
        if not spoken:
            return b""
        language_code = "id-ID" if language == "id" else "en-US"
        voice = self.voices.get(language) or self.voices["en"]
        parts: list[bytes] = []
        for chunk in _chunks(spoken, 4500):
            payload = {
                "input": {"text": chunk},
                "voice": {"languageCode": language_code, "name": voice},
                "audioConfig": {"audioEncoding": "MP3"},
            }
            response = await self._client.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                params={"key": self.api_key},
                json=payload,
            )
            response.raise_for_status()
            encoded = (response.json() or {}).get("audioContent") or ""
            if not encoded:
                raise RuntimeError("Google returned no audio")
            parts.append(base64.b64decode(encoded))
        return b"".join(parts)


class SonioxSpeaker:
    """One Soniox clip per final listener line. The response is raw MP3 bytes."""

    def __init__(
        self,
        api_key: str,
        model: str = "tts-rt-v2",
        voice_en: str = "Adrian",
        voice_id: str = "Adrian",
        timeout: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.model = model or "tts-rt-v2"
        self.voices = {"en": voice_en, "id": voice_id}
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(self, text: str, language: str) -> bytes:
        spoken = " ".join(text.split())
        if not spoken:
            return b""
        language_code = "id" if language == "id" else "en"
        voice = self.voices.get(language) or self.voices["en"]
        parts: list[bytes] = []
        for chunk in _chunks(spoken, 4500):
            payload = {
                "model": self.model,
                "language": language_code,
                "voice": voice,
                "audio_format": "mp3",
                "text": chunk,
            }
            response = await self._client.post(
                SONIOX_TTS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("Soniox returned no audio")
            parts.append(response.content)
        return b"".join(parts)


def voice_language(speaker_language: str, chosen: str | None = None) -> str:
    return target_language(speaker_language, chosen)


def _chunks(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [chunk for chunk in chunks if chunk]
