"""GPU utilisation for the operator metrics. Missing nvidia-smi is reported as unavailable."""

from __future__ import annotations

import shutil
import subprocess
import time

_cached: dict | None = None
_cached_at = 0.0


def read_gpu() -> dict | None:
    global _cached, _cached_at
    now = time.monotonic()
    if now - _cached_at < 2.0:
        return _cached
    _cached_at = now
    if shutil.which("nvidia-smi") is None:
        _cached = None
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        _cached = None
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        _cached = None
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 3:
        _cached = None
        return None
    try:
        _cached = {
            "utilization_percent": float(parts[0]),
            "memory_used_mib": float(parts[1]),
            "memory_total_mib": float(parts[2]),
        }
    except ValueError:
        _cached = None
    return _cached
