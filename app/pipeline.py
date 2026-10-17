"""One live caption pipeline.

Finalised lines are append-only. A later hypothesis can replace the grey
provisional line, and it can finalise that same line, but it does not rewrite
an earlier final line.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from dataclasses import dataclass
import logging
import time

import numpy as np

from app.agreement import LocalAgreement, ends_sentence
from app.audio import SAMPLE_RATE, is_voiced, pcm16_to_float, rms
from app.captions import CaptionLine, Transcript
from app.glossary import Glossary
from app.scripture import is_long_reading, looks_like_commentary, primary_citation, reading_status
from app.translator import TranslationRequest

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    sample_rate: int = SAMPLE_RATE
    energy_threshold: float = 0.015
    hangover_seconds: float = 0.4
    decode_interval_seconds: float = 0.6
    max_hold_seconds: float = 8.0
    keep_tail_seconds: float = 1.0
    min_sentence_words: int = 4
    worship_seconds: float = 2.0
    reading_gap_seconds: float = 2.5
    provisional_debounce_seconds: float = 0.45
    translate_timeout_seconds: float = 3.5
    min_decode_seconds: float = 0.2
    capture_stall_seconds: float = 2.0


class SessionPipeline:
    def __init__(self, document: dict, repository, recognizer, translator, glossary: Glossary, config: PipelineConfig | None = None):
        self.doc = document
        self.repo = repository
        self.recognizer = recognizer
        self.translator = translator
        self.glossary = glossary
        self.config = config or PipelineConfig()
        self.speaker_language = document["speaker_language"]
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_origin = 0.0
        self.session_time = 0.0
        self.speech_active = False
        self.silence_run = 0.0
        self.agreement = LocalAgreement()
        self.generation = 0
        self.started = False
        self.decoding = False
        self.dirty = False
        self.force_cut = False
        self.tasks: set[asyncio.Task] = set()
        self.lock = asyncio.Lock()
        self.last_audio_at = 0.0
        self.last_final_t1 = 0.0
        self.last_decode_session_time = -999.0
        self.provisional_gen = 0
        self.reading = False
        self.reading_citation = None
        self._stopped = False
        self.watch_task: asyncio.Task | None = None

    async def start(self) -> None:
        self.started = True
        self._stopped = False
        self.last_audio_at = time.monotonic()
        await self._set_status("live", self._standing_banner())
        if self.watch_task is None or self.watch_task.done():
            self.watch_task = asyncio.create_task(self._watch_capture())

    async def stop(self) -> None:
        self.started = False
        self._stopped = True
        if self.watch_task and not self.watch_task.done():
            self.watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.watch_task
        action = None
        async with self.lock:
            if len(self.buffer):
                action = self._detach_buffer()
        if action:
            await self._finalize_detached(action, reason="stop")
        await self._clear_provisional()
        await self._set_status("ended", "")

    async def set_language(self, language: str) -> None:
        async with self.lock:
            self.speaker_language = language
            self.doc["speaker_language"] = language
            self.generation += 1
        await self._emit("status")

    async def set_listener_count(self, count: int) -> None:
        async with self.lock:
            self.doc.setdefault("metrics", {})["listeners"] = count
        await self._emit("status")

    async def hide(self, line_id: str) -> bool:
        async with self.lock:
            found = self._mutate(line_id, hidden=True)
        if not found:
            return False
        await self._emit("line", found)
        return True

    async def restore(self, line_id: str) -> bool:
        async with self.lock:
            found = None
            for line in self.doc["lines"]:
                if line["id"] == line_id:
                    line["released"] = True
                    line["hidden"] = False
                    line["suppressed"] = False
                    line["listener_original"] = line["original"]
                    line["listener_translation"] = line["translation"]
                    found = line
                    break
        if not found:
            return False
        await self._emit("line", found)
        return True

    async def accept_pcm(self, pcm: bytes) -> None:
        if not self.started or self.doc.get("status") == "ended":
            return
        if len(pcm) > 256 * 1024:
            return
        samples = pcm16_to_float(pcm)
        if samples.size == 0:
            return
        self.last_audio_at = time.monotonic()
        if self.doc.get("status") == "paused":
            await self._set_status("live", self._standing_banner())
        action = None
        async with self.lock:
            action = self._ingest(samples)
        if action is None:
            return
        if action["type"] == "decode":
            self._spawn(self._decode())
        elif action["type"] == "endpoint":
            await self._finalize_detached(action, reason="endpoint")

    async def idle(self) -> None:
        for _ in range(300):
            pending = [task for task in self.tasks if not task.done()]
            if not self.decoding and not pending:
                return
            if pending:
                await asyncio.wait(pending, timeout=0.05)
            else:
                await asyncio.sleep(0.01)

    def _ingest(self, samples: np.ndarray) -> dict | None:
        duration_in = len(samples) / self.config.sample_rate
        self.session_time += duration_in
        voiced = is_voiced(samples, self.config.energy_threshold)
        action = None
        if voiced:
            self.silence_run = 0.0
            self.speech_active = True
            self.buffer = np.concatenate([self.buffer, samples]) if len(self.buffer) else samples
        elif self.speech_active:
            self.silence_run += duration_in
            self.buffer = np.concatenate([self.buffer, samples]) if len(self.buffer) else samples
            if self.silence_run >= self.config.hangover_seconds:
                action = self._detach_buffer()
        if action is not None:
            return action
        buffered = len(self.buffer) / self.config.sample_rate
        if self.speech_active and buffered >= self.config.max_hold_seconds:
            self.force_cut = True
            self.last_decode_session_time = self.session_time
            return {"type": "decode"}
        due = self.session_time - self.last_decode_session_time >= self.config.decode_interval_seconds
        if self.speech_active and due and buffered >= self.config.min_decode_seconds:
            self.last_decode_session_time = self.session_time
            return {"type": "decode"}
        return None

    def _detach_buffer(self) -> dict:
        audio = self.buffer.copy()
        origin = self.buffer_origin
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_origin = origin + (len(audio) / self.config.sample_rate)
        self.speech_active = False
        self.silence_run = 0.0
        self.agreement.reset()
        self.generation += 1
        self.force_cut = False
        return {"type": "endpoint", "audio": audio, "origin": origin, "generation": self.generation}

    async def _decode(self) -> None:
        if self.decoding:
            self.dirty = True
            return
        self.decoding = True
        try:
            while True:
                async with self.lock:
                    if len(self.buffer) < int(self.config.sample_rate * self.config.min_decode_seconds):
                        self.dirty = False
                        break
                    audio = self.buffer.copy()
                    origin = self.buffer_origin
                    language = self.speaker_language
                    hotwords = self.glossary.hotwords(language)
                    generation = self.generation
                    force_cut = self.force_cut
                    self.dirty = False
                started = time.monotonic()
                try:
                    transcript = await asyncio.to_thread(self.recognizer.transcribe, audio, language, hotwords)
                except Exception:
                    logger.exception("speech recognition failed for session %s", self.doc["id"])
                    break
                stt_ms = (time.monotonic() - started) * 1000
                commit = None
                async with self.lock:
                    if generation != self.generation or self.buffer_origin != origin:
                        continue
                    commit = self._apply_hypothesis(transcript, origin, stt_ms, force_cut, len(audio) / self.config.sample_rate)
                    if commit is not None:
                        self.force_cut = False
                if commit is not None:
                    await self._finish_final(commit, endpoint_at=time.monotonic(), stt_ms=stt_ms)
                elif transcript.words:
                    await self._emit("state")
                    self._schedule_provisional_translation()
                else:
                    await self._maybe_worship(audio)
                if not self.dirty:
                    break
        finally:
            self.decoding = False

    def _apply_hypothesis(self, transcript: Transcript, origin: float, stt_ms: float, force_cut: bool, duration: float):
        words = _absolute(transcript.words, origin)
        if not words:
            return None
        if self.doc.get("banner") == "Worship":
            self.doc["banner"] = self._standing_banner()
        confirmed = self.agreement.update([word.text for word in words])
        cut = self._cut_index(words, confirmed, duration, force_cut, origin)
        if cut:
            chosen = words[:cut]
            self._trim_to(chosen[-1].end)
            self.generation += 1
            self.agreement.reset()
            return self._begin_final(chosen, stt_ms)
        self._write_provisional(words, stt_ms)
        return None

    def _cut_index(self, words, confirmed: int, duration: float, force_cut: bool, origin: float) -> int:
        if force_cut or duration >= self.config.max_hold_seconds:
            cut = 0
            for index, word in enumerate(words):
                if (word.end - origin) <= duration - self.config.keep_tail_seconds:
                    cut = index + 1
            if cut == 0 and len(words) > 1:
                cut = len(words) - 1
            return cut
        if confirmed >= self.config.min_sentence_words and confirmed < len(words) and ends_sentence(words[confirmed - 1].text):
            return confirmed
        return 0

    def _trim_to(self, absolute_end: float) -> None:
        cut = int((absolute_end - self.buffer_origin) * self.config.sample_rate)
        cut = max(0, min(cut, len(self.buffer)))
        self.buffer = self.buffer[cut:]
        self.buffer_origin += cut / self.config.sample_rate

    def _write_provisional(self, words, stt_ms: float) -> None:
        text = _text(words)
        current = self.doc.get("provisional")
        if current and current.get("state") == "provisional" and not current.get("hidden"):
            line_id = current["id"]
            translation = current.get("translation") or ""
        else:
            line_id = self._new_id()
            translation = ""
        line = CaptionLine(
            id=line_id,
            state="provisional",
            original=text,
            translation=translation,
            t0=words[0].start,
            t1=words[-1].end,
            stt_ms=round(stt_ms, 1),
        )
        self._apply_audience(line, stick=False)
        self.doc["provisional"] = line.as_dict()
        self.provisional_gen += 1
        self.doc.setdefault("metrics", {})["stt_ms"] = round(stt_ms, 1)

    def _begin_final(self, words, stt_ms: float) -> CaptionLine:
        text = _text(words)
        current = self.doc.get("provisional")
        if current and not current.get("hidden"):
            line_id = current["id"]
        else:
            line_id = self._new_id()
        if words[0].start - self.last_final_t1 > self.config.reading_gap_seconds:
            self.reading = False
            self.reading_citation = None
        line = CaptionLine(
            id=line_id,
            state="final",
            original=text,
            translation="",
            t0=words[0].start,
            t1=words[-1].end,
            stt_ms=round(stt_ms, 1),
        )
        self.doc["provisional"] = None
        self.provisional_gen += 1
        return line

    async def _finalize_detached(self, action: dict, reason: str) -> None:
        del reason
        audio = action["audio"]
        origin = action["origin"]
        if audio is None or len(audio) == 0:
            return
        endpoint_at = time.monotonic()
        language = self.speaker_language
        hotwords = self.glossary.hotwords(language)
        started = time.monotonic()
        try:
            transcript = await asyncio.to_thread(self.recognizer.transcribe, audio, language, hotwords)
        except Exception:
            logger.exception("speech recognition failed for session %s", self.doc["id"])
            transcript = Transcript([])
        stt_ms = (time.monotonic() - started) * 1000
        words = _absolute(transcript.words, origin)
        if not words:
            await self._maybe_worship(audio)
            await self._clear_provisional()
            return
        async with self.lock:
            if self.doc.get("banner") == "Worship":
                self.doc["banner"] = ""
            line = self._begin_final(words, stt_ms)
        await self._finish_final(line, endpoint_at=endpoint_at, stt_ms=stt_ms)

    async def _finish_final(self, line: CaptionLine, endpoint_at: float, stt_ms: float) -> None:
        context = self._context()
        request = TranslationRequest(
            text=line.original,
            source_language=self.speaker_language,
            glossary=self.glossary,
            context=context,
        )
        started = time.monotonic()
        banner = None
        try:
            translation, missing = await asyncio.wait_for(
                self.translator.translate(request),
                timeout=self.config.translate_timeout_seconds,
            )
            line.translation = translation
            line.glossary_miss = missing
            line.translate_ms = round((time.monotonic() - started) * 1000, 1)
        except Exception:
            logger.exception("translation failed for session %s line %s", self.doc["id"], line.id)
            line.translation = ""
            line.translate_ms = round((time.monotonic() - started) * 1000, 1)
            banner = "Translation failed for the last line"
        line.latency_ms = round((time.monotonic() - endpoint_at) * 1000, 1)
        line.stt_ms = round(stt_ms, 1)
        async with self.lock:
            self._apply_audience(line, stick=True)
            if line.t1 > self.last_final_t1:
                self.last_final_t1 = line.t1
            existing = [item for item in self.doc["lines"] if item["id"] == line.id]
            payload = line.as_dict()
            if existing:
                existing[0].update(payload)
            else:
                self.doc["lines"].append(payload)
            metrics = self.doc.setdefault("metrics", {})
            metrics["stt_ms"] = line.stt_ms
            metrics["translate_ms"] = line.translate_ms
            metrics["lag_ms"] = line.latency_ms
            if banner:
                self.doc["banner"] = banner
            elif self.doc.get("banner") == "Translation failed for the last line":
                self.doc["banner"] = ""
            stored = payload
        await self._emit("line", stored)

    def _schedule_provisional_translation(self) -> None:
        provisional = self.doc.get("provisional")
        if not provisional or provisional.get("hidden"):
            return
        generation = self.provisional_gen
        text = provisional.get("original") or ""
        line_id = provisional["id"]
        language = self.speaker_language
        context = self._context()
        self._spawn(self._translate_provisional(generation, line_id, text, language, context))

    async def _translate_provisional(self, generation: int, line_id: str, text: str, language: str, context) -> None:
        await asyncio.sleep(self.config.provisional_debounce_seconds)
        if generation != self.provisional_gen or not text.strip():
            return
        request = TranslationRequest(text=text, source_language=language, glossary=self.glossary, context=context)
        try:
            translation, missing = await asyncio.wait_for(
                self.translator.translate(request),
                timeout=self.config.translate_timeout_seconds,
            )
        except Exception:
            logger.info("provisional translation skipped for session %s", self.doc["id"])
            return
        async with self.lock:
            if generation != self.provisional_gen:
                return
            provisional = self.doc.get("provisional")
            if not provisional or provisional["id"] != line_id:
                return
            provisional["translation"] = translation
            provisional["glossary_miss"] = missing
            if not provisional.get("suppressed"):
                provisional["listener_translation"] = translation
        await self._emit("state")

    async def _maybe_worship(self, audio: np.ndarray) -> None:
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.worship_seconds or rms(audio) < self.config.energy_threshold:
            return
        await self._set_status(self.doc.get("status") or "live", "Worship")

    async def _clear_provisional(self) -> None:
        async with self.lock:
            if not self.doc.get("provisional"):
                return
            self.doc["provisional"] = None
            self.provisional_gen += 1
        await self._emit("state")

    async def _watch_capture(self) -> None:
        while self.started and not self._stopped:
            await asyncio.sleep(0.4)
            if not self.started or self.last_audio_at <= 0:
                continue
            stalled = time.monotonic() - self.last_audio_at > self.config.capture_stall_seconds
            if stalled and self.doc.get("status") == "live":
                await self._set_status("paused", "No audio from the operator microphone")

    def _standing_banner(self) -> str:
        if self.doc.get("development_translator"):
            return "Development translator is on. These are not real translations."
        return ""

    def _apply_audience(self, line: CaptionLine, stick: bool) -> None:
        if line.released:
            line.suppressed = False
            line.listener_original = line.original
            line.listener_translation = line.translation
            return
        citation = primary_citation(line.original)
        long_reading = bool(citation and is_long_reading(line.original, citation))
        if stick and looks_like_commentary(line.original) and not long_reading:
            self.reading = False
            self.reading_citation = None
        if stick and long_reading and citation is not None:
            self.reading = True
            self.reading_citation = citation
        if self.reading or long_reading:
            reference = citation or self.reading_citation
            listener_original, listener_translation, reference_en, reference_id = reading_status(
                reference, self.speaker_language
            )
            line.suppressed = True
            line.listener_original = listener_original
            line.listener_translation = listener_translation
            line.reference_en = reference_en
            line.reference_id = reference_id
            return
        line.suppressed = False
        line.listener_original = line.original
        line.listener_translation = line.translation
        if citation is not None:
            line.reference_en = citation.display_en
            line.reference_id = citation.display_id

    def _context(self) -> list[tuple[str, str]]:
        rows = []
        for line in self.doc.get("lines") or []:
            if line.get("hidden"):
                continue
            rows.append((line.get("original") or "", line.get("translation") or ""))
        return rows[-5:]

    def _new_id(self) -> str:
        current = int(self.doc.get("line_counter") or 0) + 1
        self.doc["line_counter"] = current
        return f"L{current:05d}"

    def _mutate(self, line_id: str, hidden: bool):
        for line in self.doc["lines"]:
            if line["id"] == line_id:
                line["hidden"] = hidden
                return line
        provisional = self.doc.get("provisional")
        if provisional and provisional["id"] == line_id:
            provisional["hidden"] = hidden
            return provisional
        return None

    async def _set_status(self, status: str, banner: str) -> None:
        async with self.lock:
            if self.doc.get("status") == status and (self.doc.get("banner") or "") == banner:
                return
            self.doc["status"] = status
            self.doc["banner"] = banner
        await self._emit("status")

    async def _emit(self, event_type: str, line: dict | None = None) -> None:
        async with self.lock:
            self.doc["seq"] = int(self.doc.get("seq") or 0) + 1
            snapshot = copy.deepcopy(self.doc)
            event = {
                "type": event_type,
                "seq": snapshot["seq"],
                "status": snapshot.get("status"),
                "banner": snapshot.get("banner") or "",
                "speaker_language": self.speaker_language,
                "provisional": snapshot.get("provisional"),
                "line": copy.deepcopy(line) if line else None,
                "metrics": snapshot.get("metrics") or {},
            }
        await self.repo.save(snapshot["id"], snapshot)
        await self.repo.publish(snapshot["id"], event)

    def _spawn(self, coroutine, track: bool = True) -> None:
        task = asyncio.create_task(coroutine)
        if track:
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)


def _text(words) -> str:
    return " ".join(word.text for word in words if word.text).strip()


def _absolute(words, origin: float):
    from app.captions import Word

    converted = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        converted.append(Word(text, start=origin + float(word.start), end=origin + float(word.end)))
    return converted
