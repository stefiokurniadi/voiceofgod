from __future__ import annotations

from dataclasses import dataclass
import logging
import re

import httpx

from app.glossary import Glossary

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You translate one sermon sentence at a time between English and Bahasa Indonesia.
The speaker is preaching in a church. Be faithful and respectful.
Do not add, remove, explain, or soften what was said.
Do not invent verses, citations, quotations, or closings.
Do not add theology the speaker did not say.
Keep every token that looks like [[T0]] exactly as written.
Output only the translation, with no label and no quotes.
When the target language is Bahasa Indonesia, use formal worship Indonesian and address people as Anda, never kamu or slang.
When a phrase is already in the target language, or is in a third language, leave it as spoken.
"""


@dataclass
class TranslationRequest:
    text: str
    source_language: str
    glossary: Glossary
    context: list[tuple[str, str]]


def target_language(source_language: str) -> str:
    return "id" if source_language == "en" else "en"


def language_name(language: str) -> str:
    return "English" if language == "en" else "Bahasa Indonesia"


def build_user_message(request: TranslationRequest, protected_text: str) -> str:
    target = target_language(request.source_language)
    if request.context:
        rows = []
        for original, translation in request.context[-5:]:
            rows.append(f"{language_name(request.source_language)}: {original}")
            rows.append(f"{language_name(target)}: {translation}")
        context = "\n".join(rows)
    else:
        context = "(none)"
    return (
        f"Source language: {language_name(request.source_language)}\n"
        f"Target language: {language_name(target)}\n"
        f"Previous sentences:\n{context}\n\n"
        f"Sentence:\n{protected_text}"
    )


def clean_model_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(translation|terjemahan)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) >= 2 and cleaned[0] in {'"', "“"} and cleaned[-1] in {'"', "”"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


class GlossaryTranslator:
    """Applies the glossary and leaves the rest of the sentence unchanged.

    This is a development stand-in so the console can be exercised without a GPU.
    It is not a translation and must not be used for a service.
    """

    development = True

    async def healthy(self) -> bool:
        return True

    async def translate(self, request: TranslationRequest) -> tuple[str, list[str]]:
        if not request.text.strip():
            return "", []
        protection = request.glossary.protect(request.text, request.source_language)
        return request.glossary.restore(protection.text, protection)


class UnavailableTranslator:
    development = False

    async def healthy(self) -> bool:
        return False

    async def translate(self, request: TranslationRequest) -> tuple[str, list[str]]:
        del request
        raise RuntimeError("Translator is not configured")


class VllmTranslator:
    development = False

    def __init__(self, base_url: str, model: str, timeout: float = 3.5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    def _root(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._root()}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()

    async def translate(self, request: TranslationRequest) -> tuple[str, list[str]]:
        if not request.text.strip():
            return "", []
        protection = request.glossary.protect(request.text, request.source_language)
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(request, protection.text)},
            ],
        }
        response = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        restored, missing = request.glossary.restore(clean_model_output(content), protection)
        if missing:
            logger.info("glossary tokens missing from model output: %s", len(missing))
        return restored, missing
