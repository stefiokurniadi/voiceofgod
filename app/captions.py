from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words if word.text).strip()


@dataclass
class CaptionLine:
    id: str
    state: str
    original: str
    translation: str
    t0: float
    t1: float
    hidden: bool = False
    suppressed: bool = False
    released: bool = False
    reference_en: str | None = None
    reference_id: str | None = None
    listener_original: str = ""
    listener_translation: str = ""
    stt_ms: float | None = None
    translate_ms: float | None = None
    latency_ms: float | None = None
    glossary_miss: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "original": self.original,
            "translation": self.translation,
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "hidden": self.hidden,
            "suppressed": self.suppressed,
            "released": self.released,
            "reference_en": self.reference_en,
            "reference_id": self.reference_id,
            "listener_original": self.listener_original or self.original,
            "listener_translation": self.listener_translation or self.translation,
            "stt_ms": self.stt_ms,
            "translate_ms": self.translate_ms,
            "latency_ms": self.latency_ms,
            "glossary_miss": list(self.glossary_miss),
        }


def project_line(data: dict | None, role: str, *, tombstone: bool = False) -> dict | None:
    if not data:
        return None
    if role == "listener" and data.get("hidden"):
        if not tombstone:
            return None
        return {
            "id": data["id"],
            "state": data.get("state") or "final",
            "original": "",
            "translation": "",
            "t0": data.get("t0") or 0,
            "t1": data.get("t1") or 0,
            "hidden": True,
            "suppressed": False,
            "reference_en": None,
            "reference_id": None,
        }
    suppressed = bool(data.get("suppressed"))
    if role == "listener" and suppressed:
        original = data.get("listener_original") or ""
        translation = data.get("listener_translation") or ""
    else:
        original = data.get("original") or ""
        translation = data.get("translation") or ""
    projected = {
        "id": data["id"],
        "state": data["state"],
        "original": original,
        "translation": translation,
        "t0": data.get("t0") or 0,
        "t1": data.get("t1") or 0,
        "hidden": bool(data.get("hidden")),
        "suppressed": suppressed,
        "reference_en": data.get("reference_en"),
        "reference_id": data.get("reference_id"),
    }
    if role == "operator":
        projected["latency_ms"] = data.get("latency_ms")
        projected["stt_ms"] = data.get("stt_ms")
        projected["translate_ms"] = data.get("translate_ms")
        projected["glossary_miss"] = data.get("glossary_miss") or []
        projected["full_original"] = data.get("original") or ""
        projected["full_translation"] = data.get("translation") or ""
    return projected


def project_session(document: dict, role: str) -> dict:
    lines = []
    for line in document.get("lines") or []:
        projected = project_line(line, role)
        if projected:
            lines.append(projected)
    projected = {
        "id": document["id"],
        "speaker_language": document["speaker_language"],
        "status": document["status"],
        "banner": document.get("banner") or "",
        "seq": document.get("seq") or 0,
        "lines": lines,
        "provisional": project_line(document.get("provisional"), role),
    }
    if role == "operator":
        projected["metrics"] = document.get("metrics") or {}
        projected["development_translator"] = bool(document.get("development_translator"))
        projected["translator_ready"] = bool(document.get("translator_ready"))
        projected["glossary"] = document.get("glossary") or {"terms": [], "do_not_translate": []}
    return projected


def lines_after(lines: list[dict], after_id: str | None) -> list[dict] | None:
    """Lines strictly after an acknowledged caption. None means the id is unknown."""
    if not after_id:
        return lines
    ids = [line.get("id") for line in lines]
    if after_id not in ids:
        return None
    return lines[ids.index(after_id) + 1 :]
