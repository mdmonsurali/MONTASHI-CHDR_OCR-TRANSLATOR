"""Identity dependency — trusts X-User-Id / X-User-Role headers set by
ui_service over the internal docker network. Same contract as ocr_service
and translator_service."""

from __future__ import annotations

import uuid as _uuid

from fastapi import Header, HTTPException


class Identity:
    __slots__ = ("id", "role")

    def __init__(self, id: _uuid.UUID, role: str):
        self.id = id
        self.role = role


def require_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> Identity:
    if not x_user_id or not x_user_role:
        raise HTTPException(401, "Missing identity headers")
    try:
        uid = _uuid.UUID(x_user_id)
    except (ValueError, TypeError):
        raise HTTPException(401, "Invalid X-User-Id")
    if x_user_role not in {"user", "admin", "master"}:
        raise HTTPException(401, "Invalid X-User-Role")
    return Identity(id=uid, role=x_user_role)


def identity_headers(identity: Identity) -> dict:
    return {"X-User-Id": str(identity.id), "X-User-Role": identity.role}
