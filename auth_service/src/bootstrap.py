"""Idempotent master bootstrap.

Reads MASTER_USERNAME / MASTER_PASSWORD (with ADMIN_USERNAME / ADMIN_PASSWORD
accepted as fallback so old .env files keep working) and ensures that:

1. A user row with that username exists.
2. That row has role='master'.
3. That row is not disabled.

If the row already exists with the right state, nothing changes (idempotent).
If somebody manually demoted or disabled the master, the next start
auto-heals it back to a working master — that's the lockout-prevention
contract for this account.

Pre-existing installs from earlier versions will have one of:
  - an 'admin' bootstrap row → it gets promoted to 'master' and renamed to
    MASTER_USERNAME (typically 'master');
  - no users at all → fresh insert.
"""
from __future__ import annotations

import logging
import os

import db
from security import hash_password

log = logging.getLogger("auth_service.bootstrap")


def _read_credentials() -> tuple[str, str]:
    """Prefer the MASTER_* names; fall back to ADMIN_* for older .env files."""
    username = (os.getenv("MASTER_USERNAME")
                or os.getenv("ADMIN_USERNAME")
                or "master").strip()
    password = (os.getenv("MASTER_PASSWORD")
                or os.getenv("ADMIN_PASSWORD")
                or "").strip()
    if not username or not password:
        raise RuntimeError(
            "MASTER_USERNAME and MASTER_PASSWORD must be set "
            "(ADMIN_USERNAME / ADMIN_PASSWORD are accepted as fallback)."
        )
    return username, password


async def bootstrap_admin() -> None:
    username, password = _read_credentials()

    assert db.pool is not None
    async with db.pool.acquire() as conn:
        # 1. Find an existing master, if any.
        existing_master = await conn.fetchrow(
            "SELECT id, username, disabled FROM users WHERE role = 'master' LIMIT 1"
        )

        if existing_master:
            master_id = existing_master["id"]
            if existing_master["disabled"]:
                await conn.execute(
                    "UPDATE users SET disabled = FALSE, updated_at = now() "
                    "WHERE id = $1",
                    master_id,
                )
                log.info("bootstrap: re-enabled disabled master '%s'",
                         existing_master["username"])
            else:
                log.info("bootstrap: master already exists (id=%s, username=%s)",
                         master_id, existing_master["username"])
        else:
            # 2. No master. Look for the seeded username — could be an admin
            #    from an older install, or a demoted master that we should
            #    promote back. Use a case-insensitive match on the chosen
            #    username so the same .env value always wins.
            seeded = await conn.fetchrow(
                "SELECT id, role FROM users "
                "WHERE lower(username) = lower($1) LIMIT 1",
                username,
            )
            if seeded:
                master_id = seeded["id"]
                await conn.execute(
                    "UPDATE users SET role = 'master', disabled = FALSE, "
                    "updated_at = now() WHERE id = $1",
                    master_id,
                )
                log.info("bootstrap: promoted existing '%s' (was role=%s) "
                         "to master (id=%s)", username, seeded["role"],
                         master_id)
            else:
                # 3. Fall back to any pre-existing admin row from earlier
                #    versions so the operator's previous credentials still
                #    work, just under the new username.
                old_admin = await conn.fetchrow(
                    "SELECT id, username FROM users WHERE role = 'admin' "
                    "ORDER BY created_at ASC LIMIT 1"
                )
                if old_admin:
                    master_id = old_admin["id"]
                    await conn.execute(
                        "UPDATE users SET role = 'master', username = $1, "
                        "disabled = FALSE, updated_at = now() WHERE id = $2",
                        username, master_id,
                    )
                    log.info("bootstrap: promoted legacy admin '%s' to "
                             "master and renamed to '%s' (id=%s)",
                             old_admin["username"], username, master_id)
                else:
                    # 4. Fresh install — create the master from env.
                    row = await conn.fetchrow(
                        """
                        INSERT INTO users
                            (username, password_hash, role, must_change_password)
                        VALUES ($1, $2, 'master', FALSE)
                        RETURNING id
                        """,
                        username, hash_password(password),
                    )
                    master_id = row["id"]
                    log.info("bootstrap: created master user '%s' (id=%s)",
                             username, master_id)

    backfilled = await db.backfill_documents_owner(master_id)
    if backfilled:
        log.info("bootstrap: assigned %d orphan documents to master",
                 backfilled)
