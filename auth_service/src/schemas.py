"""Pydantic request/response models for auth_service."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _check_username(v: str) -> str:
    if not _USERNAME_RE.match(v):
        raise ValueError("username must be 3-32 chars: letters, digits, . _ -")
    return v


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


class CreateUserRequest(BaseModel):
    username: str
    password: str = Field(min_length=8, max_length=128)
    role: str = "user"

    @field_validator("username")
    @classmethod
    def _u(cls, v):
        return _check_username(v)

    @field_validator("role")
    @classmethod
    def _r(cls, v):
        if v not in {"user", "admin"}:
            raise ValueError("role must be 'user' or 'admin'")
        return v


class UpdateUserRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)
    role: Optional[str] = None
    disabled: Optional[bool] = None

    @field_validator("username")
    @classmethod
    def _u(cls, v):
        return _check_username(v) if v is not None else v

    @field_validator("role")
    @classmethod
    def _r(cls, v):
        if v is None:
            return v
        if v not in {"user", "admin"}:
            raise ValueError("role must be 'user' or 'admin'")
        return v


class UserOut(BaseModel):
    id: str
    username: str
    role: str
    must_change_password: bool
    disabled: bool
    created_at: datetime
    updated_at: datetime


class ValidateResponse(BaseModel):
    user_id: str
    username: str
    role: str
    must_change_password: bool
