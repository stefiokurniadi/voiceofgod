from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000


def float_to_pcm16(audio: np.ndarray) -> bytes:
    if audio is None or len(audio) == 0:
        return b""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    audio /= 32768.0
    return np.ascontiguousarray(audio)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def is_voiced(audio: np.ndarray, threshold: float) -> bool:
    return rms(audio) >= threshold


def clamp_input_boost(value: object, default: float = 2.0) -> float:
    """Keep the operator boost on the 1×–4× slider, in steps of 0.5×."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    stepped = math.floor(number * 2 + 0.5) / 2
    return float(min(4.0, max(1.0, stepped)))


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    if gain == 1.0 or audio.size == 0:
        return audio
    return np.clip(np.asarray(audio, dtype=np.float32) * np.float32(gain), -1.0, 1.0)
