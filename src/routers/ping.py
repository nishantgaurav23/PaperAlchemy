"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.config import get_settings
from src.schemas.api.health import PingResponse

router = APIRouter()


@router.get("/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    settings = get_settings()
    return PingResponse(version=settings.app.version)
