"""Health check response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class PingResponse(BaseModel):
    status: str = "ok"
    version: str
