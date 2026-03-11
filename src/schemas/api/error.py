"""Structured error response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    type: str
    message: str
    request_id: str | None = None
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
