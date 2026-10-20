"""Transcript downloads. Hidden lines are omitted. Suppressed lines use the text listeners saw."""

from __future__ import annotations


def render_txt(lines: list[dict], speaker_language: str, translation_language: str | None = None) -> str:
    from app.translator import caption_label, target_language

    source = caption_label(speaker_language)
    target = caption_label(target_language(speaker_language, translation_language))
    blocks: list[str] = []
    for line in _finals(lines):
        blocks.append(
            f"[{_clock(line.get('t0') or 0)}]\n"
            f"{source}: {line.get('original') or ''}\n"
            f"{target}: {line.get('translation') or ''}"
        )
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def render_srt(lines: list[dict]) -> str:
    cues: list[str] = []
    for index, line in enumerate(_finals(lines), start=1):
        start = float(line.get("t0") or 0)
        end = float(line.get("t1") or 0)
        if end <= start:
            end = start + 1.0
        text = "\n".join(part for part in (line.get("original") or "", line.get("translation") or "") if part)
        cues.append(f"{index}\n{_srt(start)} --> {_srt(end)}\n{text}\n")
    return "\n".join(cues)


def _finals(lines: list[dict]) -> list[dict]:
    return [line for line in lines if line.get("state") != "provisional" and not line.get("hidden")]


def _clock(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _srt(seconds: float) -> str:
    return _clock(seconds).replace(".", ",")
