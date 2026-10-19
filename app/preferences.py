from __future__ import annotations

import json
from pathlib import Path


class Preferences:
    """Operator choices saved on this machine. Spoken translation starts off."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.tts_enabled = False
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.tts_enabled = bool(data.get("tts_enabled", False))

    def set_tts(self, enabled: bool) -> None:
        self.tts_enabled = bool(enabled)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"tts_enabled": self.tts_enabled}, indent=2) + "\n"
        self.path.write_text(payload, encoding="utf-8")
