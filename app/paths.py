from pathlib import Path
import os

ROOT = Path(os.environ.get("VOICEOFGOD_ROOT", Path(__file__).resolve().parents[1]))
STATIC_DIR = ROOT / "app" / "static"
GLOSSARY_PATH = ROOT / "glossary" / "default.yml"
FIXTURE_DIR = ROOT / "eval" / "fixtures"
