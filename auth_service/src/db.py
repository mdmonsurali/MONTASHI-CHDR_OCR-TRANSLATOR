"""asyncpg pool + user / session CRUD for auth_service.

Reads the same schema.sql that ocr_service uses; this is safe because the
DDL is idempotent (CREATE IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
"""
from __future__ import annotations

import logging
import os
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import asyncpg

log = logging.getLogger("auth_service.db")

POSTGRES_DSN = (
    f"postgres://{os.getenv('POSTGRES_USER', 'dotsocr')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'changeme')}"
    f"@{os.getenv('POSTGRES_HOST', 'postgres')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'dotsocr')}"
)
SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "/workspace/database/schema.sql"))

pool: Optional[asyncpg.Pool] = None


# ── Lifecycle ────────────────────────────────────────────────────────────
async def init_pool() -> None:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(dsn=POSTGRES_DSN, min_size=1, max_size=5)
        log.info("postgres pool ready: %s", POSTGRES_DSN.rsplit("@", 1)[-1])


async def close_pool() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None


async def apply_schema() -> None:
    if pool is None:
        raise RuntimeError("init_pool() must be called first")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(sql)
    log.info("schema applied from %s", SCHEMA_FILE)


async def health() -> None:
    if pool is None:
        raise RuntimeError("postgres pool not initialised")
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")


# ── Users ────────────────────────────────────────────────────────────────
USER_COLS = "id, username, role, must_change_password, disabled, created_at, updated_at"


async def get_user_by_username(username: str) -> Optional[dict]:
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {USER_COLS}, password_hash FROM users WHERE lower(username) = lower($1)",
            username,
        )
    return dict(row) if row else None


async def get_user_by_id(user_id: _uuid.UUID) -> Optional[dict]:
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT {USER_COLS} FROM users WHERE id = $1", user_id)
    return dict(row) if row else None


async def create_user(username: str, password_hash: str, role: str,
                      must_change_password: bool = True) -> dict:
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            INSERT INTO users (username, password_hash, role, must_change_password)
            VALUES ($1, $2, $3, $4)
            RETURNING {USER_COLS}
            """,
            username, password_hash, role, must_change_password,
        )
    return dict(row)


async def list_users() -> list[dict]:
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT {USER_COLS} FROM users ORDER BY created_at ASC"
        )
    return [dict(r) for r in rows]


async def count_admins() -> int:
    assert pool is not None
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*)::int FROM users WHERE role = 'admin' AND disabled = FALSE"
        )


_USER_UPDATE_COLS = {"username", "password_hash", "role", "must_change_password", "disabled"}


async def update_user(user_id: _uuid.UUID, **fields) -> Optional[dict]:
    assert pool is not None
    cols = [(k, v) for k, v in fields.items() if k in _USER_UPDATE_COLS]
    if not cols:
        return await get_user_by_id(user_id)
    set_clauses = []
    values: list = []
    for i, (col, val) in enumerate(cols, start=1):
        set_clauses.append(f"{col} = ${i}")
        values.append(val)
    set_clauses.append("updated_at = now()")
    sql = (f"UPDATE users SET {', '.join(set_clauses)} "
           f"WHERE id = ${len(values) + 1} RETURNING {USER_COLS}")
    values.append(user_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *values)
    return dict(row) if row else None


async def delete_user(user_id: _uuid.UUID) -> bool:
    assert pool is not None
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
    return result.endswith(" 1")


# ── Sessions ─────────────────────────────────────────────────────────────
async def create_session(user_id: _uuid.UUID, ttl: timedelta) -> dict:
    assert pool is not None
    expires_at = datetime.now(timezone.utc) + ttl
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, expires_at)
            VALUES ($1, $2)
            RETURNING session_id, user_id, expires_at, created_at, last_seen_at
            """,
            user_id, expires_at,
        )
    return dict(row)


async def fetch_session_with_user(session_id: _uuid.UUID) -> Optional[dict]:
    """Return {session_id, user_id, expires_at, username, role, must_change_password,
    disabled} or None if expired / missing / user disabled."""
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT s.session_id, s.user_id, s.expires_at,
                   u.username, u.role, u.must_change_password, u.disabled
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.session_id = $1
              AND s.expires_at > now()
              AND u.disabled = FALSE
            """,
            session_id,
        )
    return dict(row) if row else None


async def touch_session(session_id: _uuid.UUID) -> None:
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET last_seen_at = now() WHERE session_id = $1",
            session_id,
        )


async def delete_session(session_id: _uuid.UUID) -> None:
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM sessions WHERE session_id = $1", session_id)


async def delete_user_sessions(user_id: _uuid.UUID,
                               except_session: Optional[_uuid.UUID] = None) -> None:
    assert pool is not None
    async with pool.acquire() as conn:
        if except_session is None:
            await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)
        else:
            await conn.execute(
                "DELETE FROM sessions WHERE user_id = $1 AND session_id <> $2",
                user_id, except_session,
            )


async def cleanup_expired_sessions() -> int:
    assert pool is not None
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM sessions WHERE expires_at < now()")
    try:
        return int(result.rsplit(" ", 1)[-1])
    except Exception:
        return 0


# ── Documents backfill (bootstrap) ───────────────────────────────────────
async def backfill_documents_owner(admin_id: _uuid.UUID) -> int:
    assert pool is not None
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE documents SET owner_id = $1 WHERE owner_id IS NULL",
            admin_id,
        )
    try:
        return int(result.rsplit(" ", 1)[-1])
    except Exception:
        return 0
