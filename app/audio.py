from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


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
