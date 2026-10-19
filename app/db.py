from __future__ import annotations

import json
import logging

import asyncpg

from app.paths import MIGRATIONS_DIR

logger = logging.getLogger(__name__)


async def _prepare(connection: asyncpg.Connection) -> None:
        await connection.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )


class Database:
    def __init__(self, url: str):
        self.url = (url or "").strip()
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is not None:
            return
        if not self.url:
            raise RuntimeError("Set DATABASE_URL before starting.")
        self.pool = await asyncpg.create_pool(self.url, min_size=1, max_size=5, init=_prepare)
        logger.info("connected to postgres")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def migrate(self) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            rows = await connection.fetch("SELECT name FROM schema_migrations")
            applied = {row["name"] for row in rows}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.name in applied:
                    continue
                script = path.read_text(encoding="utf-8")
                async with connection.transaction():
                    await connection.execute(script)
                    await connection.execute("INSERT INTO schema_migrations (name) VALUES ($1)", path.name)
                logger.info("applied migration %s", path.name)

    async def execute(self, sql: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.execute(sql, *args)

    async def fetch(self, sql: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(sql, *args)

    async def database_name(self) -> str:
        row = await self.fetchrow("SELECT current_database() AS name")
        return row["name"]
