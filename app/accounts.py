from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import hmac
import logging
import re
import secrets

logger = logging.getLogger(__name__)

COOKIE = "voiceofgod_session"
PASSWORD_ROUNDS = 600_000
LOGIN_DAYS = 30
LOCK_MINUTES = 15
LOCK_AFTER = 5
USERNAME = re.compile(r"^[A-Za-z0-9._-]{3,40}$")
CHURCH_NOTE = """City:
Pastor names:
Service time:
Anything else the interpreter should know:
"""
SERVICE_NOTE = """Preacher:
Passage:
Anything else about this service:
"""


class AuthError(Exception):
    pass


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ROUNDS)
    return f"pbkdf2_sha256${PASSWORD_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(rounds),
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(digest, bytes.fromhex(digest_hex))


def check_username(username: str) -> str:
    cleaned = username.strip().lower()
    if not USERNAME.fullmatch(cleaned):
        raise ValueError("Use 3 to 40 letters, numbers, dots, or hyphens for the username.")
    return cleaned


def check_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8 or len(password) > 200:
        raise ValueError("Use a password of at least 8 characters.")
    return password


def check_keep_names(names: list) -> list[str]:
    if not isinstance(names, list):
        raise ValueError("Names to keep must be a list.")
    cleaned = []
    for item in names:
        text = str(item).strip()
        if not text:
            continue
        if len(text) > 80:
            raise ValueError("A name to keep is too long.")
        cleaned.append(text)
    if len(cleaned) > 100:
        raise ValueError("The list of names to keep is too large.")
    return list(dict.fromkeys(cleaned))


def _public_user(row) -> dict:
    if row is None:
        return {}
    church_id = row["church_id"]
    keep = row["keep_names"] if "keep_names" in row and row["keep_names"] is not None else []
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "disabled": bool(row["disabled"]),
        "church_id": church_id,
        "church_name": row["church_name"] or "" if "church_name" in row else "",
        "church_note": row["church_note"] or "" if "church_note" in row else "",
        "keep_names": list(keep),
        "speaker_language": row["speaker_language"],
        "translation_language": row["translation_language"] if "translation_language" in row else "id",
        "tts_enabled": bool(row["tts_enabled"]),
        "stt_provider": row["stt_provider"] if "stt_provider" in row and row["stt_provider"] else "default",
        "tts_provider": row["tts_provider"] if "tts_provider" in row and row["tts_provider"] else "google",
    }


_USER_SQL = """
    SELECT u.id, u.username, u.role, u.disabled, u.church_id, u.speaker_language, u.translation_language, u.tts_enabled,
           u.stt_provider, u.tts_provider,
           c.name AS church_name, c.note AS church_note, c.keep_names
    FROM users u
    LEFT JOIN churches c ON c.id = u.church_id
"""


class Accounts:
    def __init__(self, database, settings):
        self.db = database
        self.settings = settings

    async def ensure_admin(self) -> None:
        existing = await self.db.fetchrow("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
        if existing:
            return
        username = (self.settings.admin_username or "").strip()
        password = self.settings.admin_password or ""
        if not username or not password:
            raise RuntimeError("Set ADMIN_USERNAME and ADMIN_PASSWORD before the first start.")
        await self.create_user(username, password, role="admin")
        logger.info("created the admin account %s", username.strip().lower())

    async def reset_for_tests(self) -> None:
        name = await self.db.database_name()
        if name != "voiceofgod_test":
            raise RuntimeError("Refusing to empty a database that is not voiceofgod_test.")
        await self.db.execute(
            "TRUNCATE login_failures, login_sessions, services, users, churches RESTART IDENTITY CASCADE"
        )
        await self.ensure_admin()

    async def create_user(self, username: str, password: str, role: str = "operator") -> dict:
        if role not in {"admin", "operator"}:
            raise ValueError("Role must be admin or operator.")
        cleaned = check_username(username)
        check_password(password)
        taken = await self.db.fetchrow("SELECT id FROM users WHERE username = $1", cleaned)
        if taken:
            raise ValueError("That username is already in use.")
        user_id = secrets.token_urlsafe(12)
        await self.db.execute(
            """
            INSERT INTO users (id, username, password_hash, role)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            cleaned,
            hash_password(password),
            role,
        )
        return await self.user_by_id(user_id)

    async def user_by_id(self, user_id: str) -> dict | None:
        row = await self.db.fetchrow(_USER_SQL + " WHERE u.id = $1", user_id)
        return _public_user(row) if row else None

    async def user_from_token(self, token: str) -> dict | None:
        if not token:
            return None
        row = await self.db.fetchrow(
            _USER_SQL + " JOIN login_sessions s ON s.user_id = u.id WHERE s.token_hash = $1 AND s.expires_at > now()",
            hash_token(token),
        )
        if row is None or row["disabled"]:
            return None
        return _public_user(row)

    async def login(self, username: str, password: str) -> str:
        cleaned = username.strip().lower()
        if await self._locked(cleaned):
            raise AuthError("Too many sign-in attempts. Try again in 15 minutes.")
        row = await self.db.fetchrow(
            "SELECT id, password_hash, disabled FROM users WHERE username = $1",
            cleaned,
        )
        if row is None or row["disabled"] or not verify_password(password, row["password_hash"]):
            if not await self._locked(cleaned):
                await self.db.execute("INSERT INTO login_failures (username) VALUES ($1)", cleaned)
            raise AuthError("The username or password is not valid.")
        await self.db.execute("DELETE FROM login_failures WHERE username = $1", cleaned)
        token = secrets.token_urlsafe(32)
        await self.db.execute(
            "INSERT INTO login_sessions (token_hash, user_id, expires_at) VALUES ($1, $2, $3)",
            hash_token(token),
            row["id"],
            datetime.now(timezone.utc) + timedelta(days=LOGIN_DAYS),
        )
        return token

    async def logout(self, token: str) -> None:
        if token:
            await self.db.execute("DELETE FROM login_sessions WHERE token_hash = $1", hash_token(token))

    async def _locked(self, username: str) -> bool:
        row = await self.db.fetchrow(
            """
            SELECT count(*) AS n FROM login_failures
            WHERE username = $1 AND failed_at > now() - interval '15 minutes'
            """,
            username,
        )
        return int(row["n"]) >= LOCK_AFTER

    async def list_users(self) -> list[dict]:
        rows = await self.db.fetch(
            """
            SELECT u.id, u.username, u.role, u.disabled, u.church_id, c.name AS church_name, u.created_at
            FROM users u
            LEFT JOIN churches c ON c.id = u.church_id
            ORDER BY u.created_at
            """
        )
        return [
            {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "disabled": bool(row["disabled"]),
                "church_id": row["church_id"],
                "church_name": row["church_name"] or "",
            }
            for row in rows
        ]

    async def set_disabled(self, user_id: str, disabled: bool) -> None:
        user = await self.db.fetchrow("SELECT id, role FROM users WHERE id = $1", user_id)
        if user is None:
            raise KeyError(user_id)
        if disabled and user["role"] == "admin":
            others = await self.db.fetchrow(
                "SELECT count(*) AS n FROM users WHERE role = 'admin' AND disabled = false AND id <> $1",
                user_id,
            )
            if int(others["n"]) == 0:
                raise ValueError("The last admin cannot be disabled.")
        await self.db.execute("UPDATE users SET disabled = $2 WHERE id = $1", user_id, disabled)
        if disabled:
            await self.db.execute("DELETE FROM login_sessions WHERE user_id = $1", user_id)

    async def set_password(self, user_id: str, password: str) -> None:
        check_password(password)
        found = await self.db.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
        if found is None:
            raise KeyError(user_id)
        await self.db.execute(
            "UPDATE users SET password_hash = $2 WHERE id = $1",
            user_id,
            hash_password(password),
        )
        await self.db.execute("DELETE FROM login_sessions WHERE user_id = $1", user_id)

    async def save_church(self, user_id: str, name: str, note: str, keep_names: list) -> dict:
        cleaned_name = name.strip()
        if not cleaned_name or len(cleaned_name) > 120:
            raise ValueError("Enter a church name of up to 120 characters.")
        if len(note) > 4000:
            raise ValueError("The church note is too long.")
        names = check_keep_names(keep_names)
        user = await self.db.fetchrow("SELECT church_id FROM users WHERE id = $1", user_id)
        if user is None:
            raise KeyError(user_id)
        if user["church_id"]:
            await self.db.execute(
                "UPDATE churches SET name = $2, note = $3, keep_names = $4 WHERE id = $1",
                user["church_id"],
                cleaned_name,
                note.strip(),
                names,
            )
        else:
            church_id = secrets.token_urlsafe(12)
            await self.db.execute(
                "INSERT INTO churches (id, name, note, keep_names) VALUES ($1, $2, $3, $4)",
                church_id,
                cleaned_name,
                note.strip(),
                names,
            )
            await self.db.execute("UPDATE users SET church_id = $2 WHERE id = $1", user_id, church_id)
        return await self.user_by_id(user_id)

    async def save_preferences(
        self,
        user_id: str,
        speaker_language: str,
        tts_enabled: bool,
        translation_language: str = "id",
        stt_provider: str = "default",
        tts_provider: str = "google",
    ) -> dict:
        from app.translator import SPEAKER_LANGUAGES, SPEAKER_LANGUAGE_ERROR

        if speaker_language not in SPEAKER_LANGUAGES:
            raise ValueError(SPEAKER_LANGUAGE_ERROR)
        if translation_language not in {"en", "id"}:
            raise ValueError("Listeners read English or Bahasa Indonesia.")
        if stt_provider not in {"default", "soniox"}:
            raise ValueError("Speech recognition must be Standard or Soniox.")
        if tts_provider not in {"google", "soniox"}:
            raise ValueError("The spoken voice must be Google WaveNet or Soniox.")
        await self.db.execute(
            """
            UPDATE users
            SET speaker_language = $2, tts_enabled = $3, translation_language = $4,
                stt_provider = $5, tts_provider = $6
            WHERE id = $1
            """,
            user_id,
            speaker_language,
            bool(tts_enabled),
            translation_language,
            stt_provider,
            tts_provider,
        )
        return await self.user_by_id(user_id)

    async def insert_service(
        self,
        service_id: str,
        church_id: str,
        name: str,
        note: str,
        service_date: date | None,
        speaker_language: str,
        voice: bool,
        token_hash: str,
        document: dict,
        translation_language: str = "id",
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO services (
                id, church_id, name, note, service_date, speaker_language, translation_language,
                voice, token_hash, document, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            service_id,
            church_id,
            name,
            note,
            service_date,
            speaker_language,
            translation_language if translation_language in {"en", "id"} else "id",
            voice,
            token_hash,
            document,
            document.get("status") or "ready",
        )

    async def save_document(self, service_id: str, document: dict) -> None:
        from app.translator import target_language

        speaker = document.get("speaker_language") or "en"
        raw_date = str(document.get("service_date") or "").strip()
        service_date = None
        if raw_date:
            try:
                service_date = date.fromisoformat(raw_date)
            except ValueError:
                service_date = None
        await self.db.execute(
            """
            UPDATE services
            SET document = $2, status = $3, speaker_language = $4, voice = $5, translation_language = $6,
                name = $7, note = $8, service_date = $9
            WHERE id = $1
            """,
            service_id,
            document,
            document.get("status") or "ready",
            speaker,
            bool(document.get("voice")),
            target_language(speaker, document.get("translation_language")),
            str(document.get("service_name") or "").strip() or "Service",
            str(document.get("service_note") or ""),
            service_date,
        )

    async def load_service(self, service_id: str):
        return await self.db.fetchrow(
            """
            SELECT id, church_id, name, note, token_hash, document, status
            FROM services
            WHERE id = $1
            """,
            service_id,
        )

    async def list_services(self, church_id: str) -> list[dict]:
        rows = await self.db.fetch(
            """
            SELECT id, name, note, service_date, speaker_language, translation_language, voice, status, created_at
            FROM services
            WHERE church_id = $1 AND status <> 'deleted'
            ORDER BY created_at DESC
            """,
            church_id,
        )
        from app.translator import target_language

        listed = []
        for row in rows:
            service_date = row["service_date"]
            speaker = row["speaker_language"]
            listed.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "note": row["note"],
                    "service_date": service_date.isoformat() if service_date else "",
                    "speaker_language": speaker,
                    "translation_language": target_language(speaker, row["translation_language"]),
                    "tts_enabled": bool(row["voice"]),
                    "status": row["status"],
                    "created_at": row["created_at"].isoformat(),
                }
            )
        return listed


def cookie_token(source) -> str:
    cookies = getattr(source, "cookies", None) or {}
    return cookies.get(COOKIE, "")
