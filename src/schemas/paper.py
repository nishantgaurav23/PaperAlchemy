"""Pydantic schemas for Paper model."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaperCreate(BaseModel):
    """Schema for creating a new paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: datetime
    updated_date: datetime | None = None
    pdf_url: str
    pdf_content: str | None = None
    sections: list[dict] | None = None
    summary: dict | None = None
    highlights: dict | None = None
    methodology: dict | None = None
    parsing_status: str = "pending"
    parsing_error: str | None = None


class PaperUpdate(BaseModel):
    """Schema for partial paper updates. All fields optional."""

    title: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    categories: list[str] | None = None
    published_date: datetime | None = None
    updated_date: datetime | None = None
    pdf_url: str | None = None
    pdf_content: str | None = None
    sections: list[dict] | None = None
    summary: dict | None = None
    highlights: dict | None = None
    methodology: dict | None = None
    parsing_status: str | None = None
    parsing_error: str | None = None


class PaperResponse(BaseModel):
    """Schema for paper API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: datetime
    updated_date: datetime | None = None
    pdf_url: str
    pdf_content: str | None = None
    sections: list[dict] | None = None
    summary: dict | None = None
    highlights: dict | None = None
    methodology: dict | None = None
    parsing_status: str
    parsing_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
