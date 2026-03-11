"""Request/response schemas for the ingestion endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for POST /api/v1/ingest/fetch."""

    target_date: str | None = Field(
        default=None,
        description="Target date in YYYYMMDD format. Defaults to yesterday.",
        pattern=r"^\d{8}$",
    )


class IngestResponse(BaseModel):
    """Response body for POST /api/v1/ingest/fetch."""

    papers_fetched: int = 0
    pdfs_downloaded: int = 0
    pdfs_parsed: int = 0
    papers_stored: int = 0
    arxiv_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time: float = 0.0
