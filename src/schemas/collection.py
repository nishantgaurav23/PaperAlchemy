"""Pydantic schemas for Collection model."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.schemas.paper import PaperResponse


class CollectionCreate(BaseModel):
    """Schema for creating a new collection."""

    name: str
    description: str | None = None


class CollectionUpdate(BaseModel):
    """Schema for partial collection updates."""

    name: str | None = None
    description: str | None = None


class CollectionResponse(BaseModel):
    """Schema for collection API responses (list view)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    user_id: uuid.UUID | None = None
    paper_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CollectionDetailResponse(BaseModel):
    """Schema for collection detail API response (includes papers)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    user_id: uuid.UUID | None = None
    papers: list[PaperResponse] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CollectionPaperAction(BaseModel):
    """Schema for add/remove paper operations."""

    paper_id: uuid.UUID
