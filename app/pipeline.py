"""One live caption pipeline.

A later hypothesis can replace the grey provisional line, and it can finalise
that same line. A scrap that continues the latest line is folded into that line.
An earlier line stays as it was.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from dataclasses import dataclass
import difflib
import logging
import re
import time

import numpy as np

from app.agreement import LocalAgreement
from app.audio import SAMPLE_RATE, apply_gain, clamp_input_boost, float_to_pcm16, is_voiced, pcm16_to_float, rms
from app.captions import CaptionLine, Transcript, Word
from app.glossary import Glossary
from app.scripture import is_long_reading, looks_like_commentary, primary_citation, reading_status
from app.translator import TranslationRequest, interpreter_background, phrase_language, target_language

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    sample_rate: int = SAMPLE_RATE
    energy_threshold: float = 0.015
    hangover_seconds: float = 0.9
    join_gap_seconds: float = 1.6
    decode_interval_seconds: float = 0.6
    max_hold_seconds: float = 8.0
    keep_tail_seconds: float = 1.0
    min_sentence_words: int = 4
    worship_seconds: float = 2.0
    reading_gap_seconds: float = 2.5
    provisional_debounce_seconds: float = 0.45
    translate_timeout_seconds: float = 3.5
    tts_timeout_seconds: float = 8.0
    min_decode_seconds: float = 0.2
    capture_stall_seconds: float = 2.0
    lag_ceiling_ms: float = 5000.0
    sentence_wait_seconds: float = 12.0


class SessionPipeline:
    def __init__(self, document: dict, repository, recognizer, translator, glossary: Glossary, config: PipelineConfig | None = None, speaker=None):
        self.doc = document
        self.repo = repository
        self.recognizer = recognizer
        self.translator = translator
        self.speaker = speaker
        self.voice_enabled = bool(document.get("voice"))
        self.durable = None
        self.glossary = glossary
        self.config = config or PipelineConfig()
        self.clips: dict[str, bytes] = {}
        self._billed_pcm = b""
        self._cached_words: list[Word] = []
        self._pending_pcm = b""
        self._pending_words: list[Word] = []
        self.speaker_language = document["speaker_language"]
        self.doc["translation_language"] = target_language(self.speaker_language, document.get("translation_language"))
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
        self.failure_hold = False
        self.rtmp_seen = False
        self.input_level = 0.0
        self.input_boost = clamp_input_boost(document.get("input_boost", 2))
        self.youtube_played = 0.0
        self.youtube_loading = False
        self.monitors: list[asyncio.Queue] = []
        self.audio_epoch = 0
        self._sealed_provisional_ids: set[str] = set()
        self._final_next = 0
        self._final_turn = 0
        self._final_done: set[int] = set()
        self._final_gate = asyncio.Condition()
        self._carry: dict | None = None
        self._carry_flush_pending = False
        self._spoken_text: dict[str, str] = {}
        self._spoken_revision: dict[str, int] = {}
        self._speech_chain: asyncio.Task | None = None

    async def start(self) -> None:
        self.started = True
        self._stopped = False
        self.failure_hold = False
        self.last_audio_at = time.monotonic()
        await self._set_status("live", self._standing_banner())
        if self.watch_task is None or self.watch_task.done():
            self.watch_task = asyncio.create_task(self._watch_capture())

    async def _halt(self) -> None:
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
            ticket, epoch = self._take_ticket()
            await self._finalize_detached(action, "stop", ticket, epoch)
        await self.idle()
        await self._flush_held_sentence()
        await self._clear_provisional()

    async def pause(self) -> None:
        await self._halt()
        await self._set_status("paused", "")

    async def stop(self) -> None:
        await self._halt()
        await self._set_status("ended", "")

    def _caption_language(self) -> str:
        return target_language(self.speaker_language, self.doc.get("translation_language"))

    async def set_language(self, language: str, translation: str | None = None) -> None:
        async with self.lock:
            self.speaker_language = language
            self.doc["speaker_language"] = language
            self.doc["translation_language"] = target_language(language, translation)
            self.generation += 1
            self.provisional_gen += 1
            self.audio_epoch += 1
            self._sealed_provisional_ids.clear()
            self._carry = None
            self._carry_flush_pending = False
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

    def note_level(self, level: float) -> None:
        self.input_level = level

    async def accept_pcm(self, pcm: bytes) -> None:
        if not self.started or self.failure_hold or self.doc.get("status") == "ended":
            return
        if self.doc.get("youtube_paused"):
            return
        if len(pcm) > 256 * 1024:
            return
        samples = pcm16_to_float(pcm)
        if samples.size == 0:
            return
        if self.doc.get("input") == "rtmp":
            samples = apply_gain(samples, self.input_boost)
            self.input_level = rms(samples)
        self.last_audio_at = time.monotonic()
        if self.doc.get("input") == "rtmp":
            from app.rtmp import DROPPED, WAITING

            if (self.doc.get("banner") or "") in {DROPPED, WAITING} or self.doc.get("status") == "paused":
                await self._set_status("live", self._standing_banner())
        elif self.doc.get("status") == "paused":
            await self._set_status("live", self._standing_banner())
        if self.youtube_loading:
            self.youtube_loading = False
            from app.youtube import LOADING

            if (self.doc.get("banner") or "") == LOADING:
                await self._set_status(self.doc.get("status") or "live", self._standing_banner())
        if self.doc.get("input") == "youtube":
            self.youtube_played += len(samples) / self.config.sample_rate
            self._offer_monitor(pcm)
        action = None
        flush = None
        async with self.lock:
            action = self._ingest(samples)
            if action is not None and action["type"] == "endpoint":
                action["ticket"], action["epoch"] = self._take_ticket()
            elif self._quiet_flush_due(action):
                self._carry_flush_pending = True
                flush = self._take_ticket()
        if flush is not None:
            self._spawn(self._flush_carry(flush[0], flush[1]))
        if action is None:
            return
        if action["type"] == "decode":
            self._spawn(self._decode())
        elif action["type"] == "endpoint":
            self._spawn(self._finalize_detached(action, "endpoint", action["ticket"], action["epoch"]))

    async def drop_partial(self) -> None:
        """Forget the audio that was still being heard. Used when a test seeks."""
        async with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)
            self.speech_active = False
            self.silence_run = 0.0
            self.agreement.reset()
            self.generation += 1
            self.audio_epoch += 1
            self._sealed_provisional_ids.clear()
            self.youtube_played = 0.0
            self.doc["provisional"] = None
            self.provisional_gen += 1
            self._carry = None
            self._carry_flush_pending = False
            self._clear_monitors()
        await self._emit("state")

    def attach_monitor(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.monitors.append(queue)
        return queue

    def detach_monitor(self, queue: asyncio.Queue) -> None:
        if queue in self.monitors:
            self.monitors.remove(queue)

    def _offer_monitor(self, pcm: bytes) -> None:
        for queue in list(self.monitors):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass

    def _clear_monitors(self) -> None:
        for queue in self.monitors:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

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
        provisional = self.doc.get("provisional")
        provisional_id = None
        if provisional and not provisional.get("hidden"):
            provisional_id = provisional["id"]
            self._sealed_provisional_ids.add(provisional_id)
        return {
            "type": "endpoint",
            "audio": audio,
            "origin": origin,
            "generation": self.generation,
            "provisional_id": provisional_id,
        }

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
                    transcript = await self._recognize(audio, language, hotwords)
                except Exception:
                    logger.exception("speech recognition failed for session %s", self.doc["id"])
                    await self._skip_failed_recognition()
                    break
                stt_ms = (time.monotonic() - started) * 1000
                commit = None
                ticket_epoch = None
                async with self.lock:
                    if generation != self.generation or self.buffer_origin != origin:
                        continue
                    commit = self._apply_hypothesis(transcript, origin, stt_ms, force_cut, len(audio) / self.config.sample_rate)
                    if commit is not None:
                        self.force_cut = False
                        ticket_epoch = self._take_ticket()
                if commit is not None and ticket_epoch is not None:
                    ticket, epoch = ticket_epoch
                    self._spawn(self._finish_when_ready(commit, time.monotonic(), stt_ms, ticket, epoch))
                elif transcript.words:
                    await self._emit("state")
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
        self._write_provisional(words, stt_ms, confirmed)
        return None

    def _cut_index(self, words, confirmed: int, duration: float, force_cut: bool, origin: float) -> int:
        del confirmed
        if force_cut or duration >= self.config.max_hold_seconds:
            cut = 0
            for index, word in enumerate(words):
                if (word.end - origin) <= duration - self.config.keep_tail_seconds:
                    cut = index + 1
            if cut == 0 and len(words) > 1:
                cut = len(words) - 1
            return cut
        return 0

    def _trim_to(self, absolute_end: float) -> None:
        cut = int((absolute_end - self.buffer_origin) * self.config.sample_rate)
        cut = max(0, min(cut, len(self.buffer)))
        self.buffer = self.buffer[cut:]
        self.buffer_origin += cut / self.config.sample_rate

    def _write_provisional(self, words, stt_ms: float, confirmed: int = 0) -> None:
        live = self.glossary.preferred_casing(_text(words))
        carry = self._carry if self._carry and str(self._carry.get("text") or "").strip() else None
        if carry:
            text = self.glossary.preferred_casing(_join_caption(str(carry["text"]), live))
            line_id = str(carry["id"])
            translation = ""
            t0 = float(carry["t0"])
            carry_words = len(str(carry["text"]).split())
        else:
            text = live
            t0 = words[0].start
            carry_words = 0
            current = self.doc.get("provisional")
            if self._reusable_provisional(current):
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
            t0=t0,
            t1=words[-1].end,
            stt_ms=round(stt_ms, 1),
        )
        self._apply_audience(line, stick=False)
        spoken = line.as_dict()
        spoken["settled"] = carry_words + max(0, min(int(confirmed), len(words)))
        self.doc["provisional"] = spoken
        self.provisional_gen += 1
        self.doc.setdefault("metrics", {})["stt_ms"] = round(stt_ms, 1)

    def _reusable_provisional(self, current) -> bool:
        return bool(
            current
            and current.get("state") == "provisional"
            and not current.get("hidden")
            and current.get("id") not in self._sealed_provisional_ids
        )

    def _begin_final(self, words, stt_ms: float, line_id: str | None = None) -> CaptionLine:
        text = self.glossary.preferred_casing(_text(words))
        current = self.doc.get("provisional")
        if line_id:
            chosen = line_id
        elif self._reusable_provisional(current):
            chosen = current["id"]
        else:
            chosen = self._new_id()
        if words[0].start - self.last_final_t1 > self.config.reading_gap_seconds:
            self.reading = False
            self.reading_citation = None
        line = CaptionLine(
            id=chosen,
            state="final",
            original=text,
            translation="",
            t0=words[0].start,
            t1=words[-1].end,
            stt_ms=round(stt_ms, 1),
        )
        if current and current.get("id") == chosen:
            self.doc["provisional"] = None
            self.provisional_gen += 1
        return line

    def _attach_or_begin(self, words, stt_ms: float, line_id: str | None = None) -> CaptionLine:
        fresh = self._begin_final(words, stt_ms, line_id)
        previous = self._latest_final()
        if previous is None or previous.get("hidden") or previous.get("suppressed"):
            return fresh
        gap = fresh.t0 - float(previous.get("t1") or 0)
        if not _should_attach(previous.get("original") or "", fresh.original, gap, self.config.join_gap_seconds):
            return fresh
        fresh.id = previous["id"]
        fresh.t0 = float(previous.get("t0") or fresh.t0)
        fresh.original = self.glossary.preferred_casing(_join_caption(previous.get("original") or "", fresh.original))
        return fresh

    def _latest_final(self) -> dict | None:
        for line in reversed(self.doc.get("lines") or []):
            if line.get("state") == "final":
                return line
        return None

    def _take_ticket(self) -> tuple[int, int]:
        ticket = self._final_next
        self._final_next += 1
        return ticket, self.audio_epoch

    async def _wait_turn(self, ticket: int) -> None:
        async with self._final_gate:
            while ticket != self._final_turn:
                await self._final_gate.wait()

    async def _advance_turn(self, ticket: int) -> None:
        async with self._final_gate:
            self._final_done.add(ticket)
            while self._final_turn in self._final_done:
                self._final_done.discard(self._final_turn)
                self._final_turn += 1
            self._final_gate.notify_all()

    def _stale(self, epoch: int) -> bool:
        return epoch != self.audio_epoch or self.failure_hold

    async def _finish_when_ready(self, line: CaptionLine, endpoint_at: float, stt_ms: float, ticket: int, epoch: int) -> None:
        try:
            await self._wait_turn(ticket)
            if self._stale(epoch):
                return
            async with self.lock:
                ready = [] if self._stale(epoch) else self._sentence_gate(line)
            if self._stale(epoch):
                return
            await self._emit("state")
            for item in ready:
                await self._finish_final(item, endpoint_at=endpoint_at, stt_ms=stt_ms, epoch=epoch)
        finally:
            await self._advance_turn(ticket)

    async def _finalize_detached(self, action: dict, reason: str, ticket: int, epoch: int) -> None:
        del reason
        try:
            audio = action["audio"]
            origin = action["origin"]
            if audio is None or len(audio) == 0 or self._stale(epoch):
                return
            endpoint_at = time.monotonic()
            language = self.speaker_language
            hotwords = self.glossary.hotwords(language)
            started = time.monotonic()
            try:
                transcript = await self._recognize(audio, language, hotwords, whole=True)
            except Exception:
                logger.exception("speech recognition failed for session %s", self.doc["id"])
                await self._skip_failed_recognition()
                await self._retire_provisional(action.get("provisional_id"))
                return
            stt_ms = (time.monotonic() - started) * 1000
            words = _absolute(transcript.words, origin)
            await self._wait_turn(ticket)
            if self._stale(epoch):
                return
            if not words:
                await self._maybe_worship(audio)
                await self._retire_provisional(action.get("provisional_id"))
                return
            async with self.lock:
                if self._stale(epoch):
                    return
                if self.doc.get("banner") == "Worship":
                    self.doc["banner"] = ""
                if self._carry:
                    line = self._begin_final(words, stt_ms, action.get("provisional_id"))
                else:
                    line = self._attach_or_begin(words, stt_ms, action.get("provisional_id"))
                ready = self._sentence_gate(line)
            await self._emit("state")
            for item in ready:
                await self._finish_final(item, endpoint_at=endpoint_at, stt_ms=stt_ms, epoch=epoch)
        finally:
            provisional_id = action.get("provisional_id")
            if provisional_id:
                self._sealed_provisional_ids.discard(provisional_id)
            await self._advance_turn(ticket)

    async def _retire_provisional(self, line_id: str | None) -> None:
        if not line_id:
            return
        async with self.lock:
            provisional = self.doc.get("provisional")
            if not provisional or provisional.get("id") != line_id:
                return
            self.doc["provisional"] = None
            self.provisional_gen += 1
        await self._emit("state")

    async def _finish_final(self, line: CaptionLine, endpoint_at: float, stt_ms: float, epoch: int) -> None:
        context = self._context()
        caption = self._caption_language()
        request = TranslationRequest(
            text=line.original,
            source_language=self.speaker_language,
            glossary=self.glossary,
            context=context,
            session_context=interpreter_background(self.doc),
            chosen_language=caption,
        )
        started = time.monotonic()
        banner = None
        try:
            revise = getattr(self.translator, "revise", None)
            if revise is not None:
                corrected, translation, missing = await asyncio.wait_for(
                    revise(request),
                    timeout=self.config.translate_timeout_seconds,
                )
                accepted = accept_revision(line.original, corrected)
                if accepted is not None and accepted.casefold() != " ".join(line.original.split()).casefold():
                    line.heard = line.original
                    line.original = self.glossary.preferred_casing(accepted)
            elif phrase_language(line.original) == caption:
                translation, missing = line.original, []
            else:
                translation, missing = await asyncio.wait_for(
                    self.translator.translate(request),
                    timeout=self.config.translate_timeout_seconds,
                )
            if phrase_language(line.original) == caption:
                translation, missing = line.original, []
            line.translation = finish_translation(line.original, translation, _keep_words(self.glossary), target=caption)
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
            if self._stale(epoch):
                return
            self._apply_audience(line, stick=True)
            spoken = "" if line.hidden else (line.listener_translation or "").strip()
            if line.t1 > self.last_final_t1:
                self.last_final_t1 = line.t1
            existing = [item for item in self.doc["lines"] if item["id"] == line.id]
            payload = line.as_dict()
            if existing and existing[0].get("audio"):
                payload["audio"] = True
                payload["audio_id"] = existing[0].get("audio_id") or line.id
            else:
                payload["audio"] = False
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
            elif (line.latency_ms or 0) > self.config.lag_ceiling_ms:
                self.doc["banner"] = "Captions are running behind."
            elif self.doc.get("banner") in {"Translation failed for the last line", "Captions are running behind."}:
                self.doc["banner"] = self._standing_banner()
            stored = payload
        await self._emit("line", stored)
        self._enqueue_speech(line.id, spoken, epoch)

    async def _recognize(self, audio: np.ndarray, language: str, hotwords: str | None, whole: bool = False) -> Transcript:
        if whole:
            self._billed_pcm = b""
            self._cached_words = []
            return await asyncio.to_thread(self.recognizer.transcribe, audio, language, hotwords)
        piece, offset, cached = self._unbilled(audio)
        if piece is None:
            return Transcript(cached)
        try:
            transcript = await asyncio.to_thread(self.recognizer.transcribe, piece, language, hotwords)
        except Exception:
            self._billed_pcm = self._pending_pcm
            self._cached_words = self._pending_words
            raise
        if cached is None:
            return transcript
        shifted = [Word(word.text, word.start + offset, word.end + offset) for word in transcript.words]
        self._cached_words = list(cached) + shifted
        return Transcript(self._cached_words)

    def _unbilled(self, audio: np.ndarray) -> tuple[np.ndarray | None, float, list[Word] | None]:
        if not getattr(self.recognizer, "billed_once", False):
            return audio, 0.0, None
        pcm = float_to_pcm16(audio)
        if self._billed_pcm and pcm.startswith(self._billed_pcm):
            offset = (len(self._billed_pcm) // 2) / SAMPLE_RATE
            fresh = pcm[len(self._billed_pcm) :]
            cached = list(self._cached_words)
            if len(fresh) < int(SAMPLE_RATE * 0.15):
                return None, offset, cached
            self._pending_pcm = self._billed_pcm
            self._pending_words = cached
            self._billed_pcm = pcm
            return pcm16_to_float(fresh), offset, cached
        self._pending_pcm = self._billed_pcm
        self._pending_words = list(self._cached_words)
        self._billed_pcm = pcm
        self._cached_words = []
        return audio, 0.0, []

    async def set_voice(self, enabled: bool) -> None:
        async with self.lock:
            self.voice_enabled = bool(enabled)
            self.doc["voice"] = self.voice_enabled
        await self._emit("status")

    async def _store_clip(self, line_id: str, text: str) -> bool:
        if not self.voice_enabled or not self.speaker or not text:
            return False
        try:
            clip = await asyncio.wait_for(
                self.speaker.synthesize(text, self._caption_language()),
                timeout=self.config.tts_timeout_seconds,
            )
        except Exception:
            logger.info("speech synthesis failed for session %s line %s", self.doc["id"], line_id)
            return False
        if not clip:
            return False
        self.clips[line_id] = clip
        return True

    def _enqueue_speech(self, line_id: str, text: str, epoch: int) -> None:
        """Speak lines in order. An update waits, then speaks only the words that were added."""
        if not text.strip():
            return
        prior = self._speech_chain

        async def run() -> None:
            if prior is not None:
                with contextlib.suppress(Exception):
                    await prior
            await self._speak_in_turn(line_id, text, epoch)

        task = asyncio.create_task(run())
        self._speech_chain = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _speak_in_turn(self, line_id: str, text: str, epoch: int) -> None:
        if self._stale(epoch):
            return
        text = " ".join(text.split())
        async with self.lock:
            if self._stale(epoch):
                return
            previous = self._spoken_text.get(line_id, "")
            addition = speech_addition(previous, text)
            if addition is None:
                return
            revision = self._spoken_revision.get(line_id, 0) + 1
            self._spoken_revision[line_id] = revision
            self._spoken_text[line_id] = text
            clip_id = line_id if revision == 1 else f"{line_id}#{revision}"
        if not await self._store_clip(clip_id, addition):
            async with self.lock:
                if self._spoken_text.get(line_id) == text:
                    self._spoken_text[line_id] = previous
                    self._spoken_revision[line_id] = revision - 1
            return
        if self._stale(epoch):
            self.clips.pop(clip_id, None)
            return
        async with self.lock:
            if self._stale(epoch):
                self.clips.pop(clip_id, None)
                return
            found = next((item for item in self.doc["lines"] if item["id"] == line_id), None)
            if found is None or found.get("hidden"):
                self.clips.pop(clip_id, None)
                return
            found["audio"] = True
            found["audio_id"] = clip_id
            stored = copy.deepcopy(found)
        await self._emit("line", stored)

    async def _skip_failed_recognition(self) -> None:
        """Drop the clip that the recognizer rejected and keep listening."""
        async with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)
            self.speech_active = False
            self.silence_run = 0.0
            self.generation += 1
            self.audio_epoch += 1
            self.provisional_gen += 1
            self.doc["provisional"] = None
            self.dirty = False
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
                if self.doc.get("input") == "rtmp":
                    from app.rtmp import UNAVAILABLE, WAITING

                    if (self.doc.get("banner") or "") == UNAVAILABLE or self.rtmp_seen:
                        continue
                    if (self.doc.get("banner") or "") != WAITING:
                        await self._set_status("live", WAITING)
                    continue
                elif self.doc.get("input") == "youtube":
                    from app.youtube import ENDED, UNAVAILABLE, WAITING

                    if self.youtube_loading or self.doc.get("youtube_paused") or (self.doc.get("banner") or "") in {UNAVAILABLE, ENDED}:
                        continue
                    banner = WAITING
                else:
                    banner = "No audio from the operator microphone"
                await self._set_status("paused", banner)

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
                reference, self.speaker_language, self._caption_language()
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
                "translation_language": self._caption_language(),
                "voice": bool(snapshot.get("voice")),
                "service_name": snapshot.get("service_name") or "",
                "service_note": snapshot.get("service_note") or "",
                "service_date": snapshot.get("service_date") or "",
                "provisional": snapshot.get("provisional"),
                "line": copy.deepcopy(line) if line else None,
                "metrics": snapshot.get("metrics") or {},
            }
        await self.repo.save(snapshot["id"], snapshot)
        await self.repo.publish(snapshot["id"], event)
        durable = event_type in {"line", "deleted"} or (
            event_type == "status" and snapshot.get("status") in {"ended", "deleted"}
        )
        if self.durable and durable:
            try:
                await self.durable(snapshot)
            except Exception:
                logger.exception("could not save service %s", snapshot["id"])

    def _spawn(self, coroutine, track: bool = True) -> None:
        task = asyncio.create_task(coroutine)
        if track:
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    def _quiet_flush_due(self, action: dict | None) -> bool:
        if action is not None or self.speech_active or self.decoding or self._carry_flush_pending:
            return False
        if not self._carry or self._final_next != self._final_turn:
            return False
        return self.session_time - float(self._carry["t1"]) >= self.config.reading_gap_seconds

    def _sentence_gate(self, line: CaptionLine) -> list[CaptionLine]:
        """Send only text up to the last full stop. Hold the rest for the next chunk."""
        carry = self._carry if self._carry and str(self._carry.get("text") or "").strip() else None
        if carry:
            text = _join_caption(str(carry["text"]), line.original)
            t0 = float(carry["t0"])
            line_id = str(carry["id"])
        else:
            text = line.original.strip()
            t0 = line.t0
            line_id = line.id
        text = self.glossary.preferred_casing(text).strip()
        t1 = line.t1 if line.t1 >= t0 else t0
        if text and (t1 - t0) >= self.config.sentence_wait_seconds:
            self._carry = None
            line.original = text
            line.t0 = t0
            line.t1 = t1
            line.id = line_id
            self._drop_matching_provisional(line_id)
            return [line]
        finished, rest = _split_sentence(text)
        if not finished:
            self._carry = {"id": line_id, "text": text, "t0": t0, "t1": t1}
            self._show_carry(line.stt_ms)
            return []
        cut = _portion_time(text, finished, t0, t1)
        line.original = finished
        line.t0 = t0
        line.t1 = cut
        line.id = line_id
        if rest:
            self._carry = {"id": self._new_id(), "text": rest, "t0": cut, "t1": t1}
            self._show_carry(line.stt_ms)
        else:
            self._carry = None
            self._drop_matching_provisional(line_id)
        return [line]

    def _show_carry(self, stt_ms: float | None) -> None:
        carry = self._carry
        if not carry:
            return
        text = self.glossary.preferred_casing(str(carry["text"]))
        caption = CaptionLine(
            id=str(carry["id"]),
            state="provisional",
            original=text,
            translation="",
            t0=float(carry["t0"]),
            t1=float(carry["t1"]),
            stt_ms=stt_ms,
        )
        self._apply_audience(caption, stick=False)
        spoken = caption.as_dict()
        spoken["settled"] = 0
        self.doc["provisional"] = spoken
        self.provisional_gen += 1

    def _drop_matching_provisional(self, line_id: str) -> None:
        provisional = self.doc.get("provisional")
        if provisional and provisional.get("id") == line_id:
            self.doc["provisional"] = None
            self.provisional_gen += 1

    def _take_carry_line(self) -> CaptionLine | None:
        carry = self._carry
        if not carry or not str(carry.get("text") or "").strip():
            self._carry = None
            return None
        self._carry = None
        text = self.glossary.preferred_casing(str(carry["text"]).strip())
        self._drop_matching_provisional(str(carry["id"]))
        return CaptionLine(
            id=str(carry["id"]),
            state="final",
            original=text,
            translation="",
            t0=float(carry["t0"]),
            t1=float(carry["t1"]),
        )

    async def _flush_carry(self, ticket: int, epoch: int, *, force: bool = False) -> None:
        line = None
        try:
            await self._wait_turn(ticket)
            if not self._stale(epoch):
                async with self.lock:
                    if not self._stale(epoch) and self._carry:
                        quiet_for = self.session_time - float(self._carry["t1"])
                        ready = force or (
                            not self.speech_active
                            and not self.decoding
                            and quiet_for >= self.config.reading_gap_seconds
                        )
                        if ready:
                            line = self._take_carry_line()
            if line is not None:
                await self._finish_final(line, endpoint_at=time.monotonic(), stt_ms=line.stt_ms or 0.0, epoch=epoch)
        finally:
            self._carry_flush_pending = False
            await self._advance_turn(ticket)

    async def _flush_held_sentence(self) -> None:
        async with self.lock:
            if not self._carry:
                return
            ticket, epoch = self._take_ticket()
        await self._flush_carry(ticket, epoch, force=True)


_SENTENCE_END = re.compile(r"(?:\.\.\.|…|[.?!])[\"'”’)\]]*(?=\s|$)")


def _split_sentence(text: str) -> tuple[str, str]:
    """Text up to the last full stop, and the unfinished tail after it."""
    last = None
    for match in _SENTENCE_END.finditer(text):
        last = match
    if last is None:
        return "", text.strip()
    return text[: last.end()].strip(), text[last.end() :].strip()


def _portion_time(text: str, finished: str, t0: float, t1: float) -> float:
    total = len(text.split()) or 1
    done = len(finished.split())
    if done >= total:
        return t1
    span = max(0.0, t1 - t0)
    return t0 + span * (done / total)


def accept_revision(raw: str, corrected: str) -> str | None:
    """Keep a cleanup that still says the same line. Reject a rewrite."""
    raw_text = " ".join(raw.split())
    revised = " ".join((corrected or "").split())
    if not revised:
        return None
    if raw_text.casefold() == revised.casefold():
        return revised
    raw_words = raw_text.split()
    revised_words = revised.split()
    if not raw_words:
        return None
    if abs(len(revised_words) - len(raw_words)) / len(raw_words) > 0.30:
        return None
    if difflib.SequenceMatcher(a=raw_text.casefold(), b=revised.casefold()).ratio() < 0.6:
        return None
    return revised


def _spoken_token(word: str) -> str:
    return word.strip(".,!?;:\"'“”").casefold()


def speech_addition(previous: str, current: str) -> str | None:
    """Words still to be spoken after a line grows. None when nothing new was added."""
    current = " ".join(current.split())
    previous = " ".join(previous.split())
    if not current or current == previous:
        return None
    if not previous:
        return current
    previous_words = previous.split()
    current_words = current.split()
    shared = 0
    while shared < len(previous_words) and shared < len(current_words):
        if _spoken_token(previous_words[shared]) != _spoken_token(current_words[shared]):
            break
        shared += 1
    if shared == 0:
        return current
    extra = " ".join(current_words[shared:]).strip()
    return extra or None


def _should_attach(previous: str, new: str, gap: float, limit: float) -> bool:
    """A scrap after a breath belongs on the line already on screen."""
    if gap < -0.05 or gap > limit:
        return False
    left = previous.strip()
    right = new.strip()
    if not left or not right:
        return False
    first = re.match(r"[A-Za-z']+", right)
    first_word = first.group(0).casefold() if first else ""
    if first_word in _CONTINUATIONS:
        return True
    if right[0].islower():
        return True
    if left.endswith((",", ";", ":")):
        return True
    words = re.findall(r"[A-Za-z0-9']+", right)
    short = right.casefold().strip(" .!?,")
    return len(words) <= 1 and short not in _COMPLETE_SHORT


def _join_caption(previous: str, new: str) -> str:
    left = previous.rstrip()
    right = new.strip()
    if not left:
        return right
    if not right:
        return left
    if not re.search(r"[.?!]$", left) and right[0].isupper():
        right = right[0].lower() + right[1:]
    return f"{left} {right}"


_CONTINUATIONS = {
    "and", "for", "with", "even", "are", "who", "that", "not", "but", "or",
    "which", "when", "because", "so", "of", "from",
}
_COMPLETE_SHORT = {
    "amen", "hallelujah", "thanks", "thank you", "please", "yes", "no",
    "hello", "hi", "okay", "ok", "shalom",
}


def _keep_words(glossary: Glossary) -> set[str]:
    words: set[str] = set()
    for term in glossary.terms:
        words.update(term.en.split())
        words.update(term.id.split())
    for phrase in glossary.do_not_translate:
        words.update(phrase.split())
    words.update({"Allah", "Anda", "Roh", "Kudus", "Tuhan", "Yesus", "Yohanes", "Injil"})
    return {word.strip(".,;:") for word in words if word.strip(".,;:")}


_NOT_A_NAME = {
    "sunday", "lord", "god", "amen", "i",
    "of", "the", "a", "an", "and", "to", "for", "in", "on", "at", "by", "or",
}
_JESUS = re.compile(r"\b(?:jesus|yesus|christ|kristus)\b", re.IGNORECASE)
_ANIMAL = re.compile(r"\b(?:animal|animals|goat|lamb|sheep|ox|cattle|cow|bird|dog|cat|lion|fish)\b", re.IGNORECASE)


def spoken_names(source: str) -> list[str]:
    """Names the recognizer capitalized in the middle of a sentence."""
    found: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", source):
        found.append(match.group(1))
    masked = re.sub(r"(^|[.!?]\s+)([A-Z])", lambda item: item.group(1) + item.group(2).lower(), source)
    covered = {part for phrase in found for part in phrase.split()}
    for match in re.finditer(r"\b([A-Z][a-z]+)\b", masked):
        word = match.group(1)
        if word.casefold() in _NOT_A_NAME or word in covered:
            continue
        found.append(word)
    return found


def repair_translation(source: str, text: str, keep: set[str] | None = None, *, target: str | None = None) -> str:
    """Correct translation mistakes that changed what the preacher said."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return cleaned
    kept = {word.casefold() for word in (keep or set())}
    kept.update({"amen", "hallelujah", "shalom"})
    if not _JESUS.search(source):
        cleaned = re.sub(r"\b(?:Yesus|yesus|Jesus|jesus)\b", "", cleaned)
    cleaned = re.sub(r"\bsaya kita\b", "kita", cleaned, flags=re.IGNORECASE)
    if re.search(r"\bSunday\b", source):
        cleaned = re.sub(r"\bhari\s+minggu\b", "hari Minggu", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bminggu(?=\s+(?:pagi|depan)\b)", "Minggu", cleaned)
    if re.search(r"\bmale\b", source, re.IGNORECASE) and not _ANIMAL.search(source):
        cleaned = re.sub(r"\bseekor jantan\b", "seorang laki-laki", cleaned, flags=re.IGNORECASE)
    if target != "en":
        cleaned = _drop_copied_english(source, cleaned, kept)
    for name in sorted(spoken_names(source), key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(name)}\b", name, cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"(?:\.\s*){2,}", ". ", cleaned)
    return cleaned.strip()


def _drop_copied_english(source: str, text: str, kept: set[str]) -> str:
    source_words = {word.casefold() for word in re.findall(r"[A-Za-z']+", source)}
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    words = [token for token in tokens if re.fullmatch(r"[A-Za-z']+", token)]
    copied = [
        word for word in words
        if word.casefold() in source_words and word.casefold() not in kept and word.isascii()
    ]
    other = [word for word in words if word not in copied]
    if len(other) < 2 or len(copied) > len(other):
        return text
    drop = {word.casefold() for word in copied}
    return "".join(token for token in tokens if not (re.fullmatch(r"[A-Za-z']+", token) and token.casefold() in drop))


def finish_translation(source: str, text: str, keep: set[str] | None = None, *, target: str | None = None) -> str:
    kept = set(keep or ())
    kept.update({"Tuhan", "Yesus", "Allah"})
    for name in spoken_names(source):
        kept.update(name.split())
    if re.search(r"\bSunday\b", source):
        kept.add("Minggu")
    # The listener language wins. A caption already heard in that language stays there.
    if target in {"en", "id"} and phrase_language(source) == target and phrase_language(text) not in {None, target}:
        text = source
    return polish_translation(repair_translation(source, text, kept, target=target), kept)


def polish_translation(text: str, keep: set[str] | None = None) -> str:
    """Capitalize the line, and do not capitalize a word only because a comma preceded it."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return cleaned
    cleaned = cleaned[0].upper() + cleaned[1:]
    kept = {word.casefold() for word in (keep or set())}

    def fix(match: re.Match) -> str:
        word = match.group(1)
        if word.casefold() in kept:
            return match.group(0)
        return " " + word[0].lower() + word[1:]

    return re.sub(r"(?<=[^\s.?!])\s+([A-Z][\w]+)", fix, cleaned)


def _text(words) -> str:
    raw = " ".join(word.text for word in words if word.text).strip()
    return _join_false_periods(raw)


def _join_false_periods(text: str) -> str:
    """A short pause inside one breath should not become a new sentence."""
    continuations = {
        "comforts", "from", "is", "and", "let", "we", "are", "was", "were",
        "opens", "open", "has", "have", "will", "can",
    }

    def join(match: re.Match) -> str:
        word = match.group(1)
        if word.casefold() not in continuations:
            return match.group(0)
        return " " + word[0].lower() + word[1:]

    text = re.sub(r"\.\s+([A-Z][A-Za-z]+)", join, text)
    return re.sub(r",\s+([A-Z][A-Za-z]+)", join, text)


def _absolute(words, origin: float):
    from app.captions import Word

    converted = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        converted.append(Word(text, start=origin + float(word.start), end=origin + float(word.end)))
    return converted
