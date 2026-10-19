"""Glossary protection.

Church terms are replaced with stable tokens before the model sees a sentence,
then written back afterwards. Do-not-translate spans win over shorter terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class Term:
    en: str
    id: str
    id_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProtectedSpan:
    token: str
    surface: str
    replacement: str


@dataclass(frozen=True)
class Protection:
    text: str
    spans: tuple[ProtectedSpan, ...]

    @property
    def mapping(self) -> dict[str, str]:
        return {span.token: span.replacement for span in self.spans}


class Glossary:
    def __init__(self, terms: list[Term], do_not_translate: list[str]):
        self.terms = list(terms)
        self.do_not_translate = [phrase.strip() for phrase in do_not_translate if phrase.strip()]

    @classmethod
    def load(cls, path: Path) -> Glossary:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        terms = [
            Term(
                en=item["en"].strip(),
                id=item["id"].strip(),
                id_aliases=tuple(alias.strip() for alias in item.get("id_aliases") or [] if alias.strip()),
            )
            for item in raw.get("terms") or []
        ]
        return cls(terms, list(raw.get("do_not_translate") or []))

    def merged(self, do_not_translate: list[str] | None = None, terms: list[dict] | None = None) -> Glossary:
        by_en = {term.en.casefold(): term for term in self.terms}
        for item in terms or []:
            term = Term(
                en=str(item["en"]).strip(),
                id=str(item["id"]).strip(),
                id_aliases=tuple(str(alias).strip() for alias in item.get("id_aliases") or [] if str(alias).strip()),
            )
            if term.en and term.id:
                by_en[term.en.casefold()] = term
        phrases = list(dict.fromkeys([*self.do_not_translate, *(do_not_translate or [])]))
        return Glossary(list(by_en.values()), phrases)

    def to_dict(self) -> dict:
        return {
            "do_not_translate": list(self.do_not_translate),
            "terms": [
                {"en": term.en, "id": term.id, "id_aliases": list(term.id_aliases)}
                for term in self.terms
            ],
        }

    @classmethod
    def from_public(cls, terms: list[dict] | None, do_not_translate: list[str] | None) -> Glossary:
        raw_terms = terms or []
        raw_phrases = do_not_translate or []
        if len(raw_terms) > 100 or len(raw_phrases) > 100:
            raise ValueError("The glossary is too large.")
        cleaned: list[Term] = []
        for item in raw_terms:
            if not isinstance(item, dict):
                raise ValueError("Each glossary term needs an English and Indonesian side.")
            english = str(item.get("en", "")).strip()
            indonesian = str(item.get("id", "")).strip()
            if not english and not indonesian:
                continue
            if not english or not indonesian:
                raise ValueError("Each glossary term needs an English and Indonesian side.")
            if len(english) > 80 or len(indonesian) > 80:
                raise ValueError("A glossary phrase is too long.")
            aliases = tuple(
                str(alias).strip()
                for alias in item.get("id_aliases") or []
                if str(alias).strip()
            )
            cleaned.append(Term(english, indonesian, aliases))
        phrases: list[str] = []
        for phrase in raw_phrases:
            text = str(phrase).strip()
            if not text:
                continue
            if len(text) > 80:
                raise ValueError("A do-not-translate phrase is too long.")
            phrases.append(text)
        return cls(cleaned, list(dict.fromkeys(phrases)))

    def source_pairs(self, source_language: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for term in self.terms:
            if source_language == "en":
                pairs.append((term.en, term.id))
            else:
                pairs.append((term.id, term.en))
                for alias in term.id_aliases:
                    pairs.append((alias, term.en))
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return pairs

    def hotwords(self, source_language: str) -> str:
        phrases = [source for source, _target in self.source_pairs(source_language)]
        phrases.extend(self.do_not_translate)
        return ", ".join(phrases[:40])

    def preferred_casing(self, text: str) -> str:
        phrases = sorted(
            {phrase for phrase in ([term.en for term in self.terms] + list(self.do_not_translate)) if phrase},
            key=len,
            reverse=True,
        )
        if not phrases:
            return text
        pattern = "|".join(rf"(?<![\w]){re.escape(phrase)}(?![\w])" for phrase in phrases)
        preferred = {phrase.casefold(): phrase for phrase in phrases}

        def replace(match: re.Match) -> str:
            return preferred[match.group(0).casefold()]

        return re.sub(pattern, replace, text, flags=re.IGNORECASE)

    def protect(self, text: str, source_language: str) -> Protection:
        candidates: list[tuple[int, int, str, str]] = []
        for phrase in sorted(self.do_not_translate, key=len, reverse=True):
            for match in _find(phrase, text):
                candidates.append((match.start(), match.end(), match.group(0), match.group(0)))
        for source, target in self.source_pairs(source_language):
            for match in _find(source, text):
                candidates.append((match.start(), match.end(), match.group(0), target))
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
        occupied = [False] * len(text)
        chosen: list[tuple[int, int, str]] = []
        for start, end, _surface, replacement in candidates:
            if start < 0 or end > len(text) or any(occupied[start:end]):
                continue
            for index in range(start, end):
                occupied[index] = True
            chosen.append((start, end, replacement))
        chosen.sort()
        spans: list[ProtectedSpan] = []
        pieces: list[str] = []
        cursor = 0
        for index, (start, end, replacement) in enumerate(chosen):
            token = f"[[T{index}]]"
            spans.append(ProtectedSpan(token=token, surface=text[start:end], replacement=replacement))
            pieces.append(text[cursor:start])
            pieces.append(token)
            cursor = end
        pieces.append(text[cursor:])
        return Protection("".join(pieces), tuple(spans))

    def restore(self, text: str, protection: Protection) -> tuple[str, list[str]]:
        missing: list[str] = []
        for span in sorted(protection.spans, key=lambda item: len(item.token), reverse=True):
            pattern = re.compile(re.escape(span.token), re.IGNORECASE)
            text, count = pattern.subn(lambda _match, value=span.replacement: value, text)
            if count:
                continue
            leaked = next(_find(span.surface, text), None)
            if leaked and span.surface.casefold() != span.replacement.casefold():
                text = text[: leaked.start()] + span.replacement + text[leaked.end() :]
                continue
            if span.surface.casefold() != span.replacement.casefold():
                missing.append(span.replacement)
        return text, missing


def _find(phrase: str, text: str):
    pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
    return pattern.finditer(text)
