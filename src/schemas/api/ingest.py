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
    max_results: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Max papers to fetch. Useful when fetching latest without date filter.",
    )


class IngestResponse(BaseModel):
    """Response body for POST /api/v1/ingest/fetch."""

    papers_fetched: int = 0
    pdfs_downloaded: int = 0
    pdfs_parsed: int = 0
    papers_stored: int = 0
    chunks_indexed: int = 0
    arxiv_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time: float = 0.0


class ReparseRequest(BaseModel):
    """Request body for POST /api/v1/ingest/reparse."""

    status_filter: str = Field(
        default="pending",
        description="Re-parse papers with this parsing_status. Use 'pending', 'failed', or 'all'.",
    )
    limit: int = Field(default=50, ge=1, le=500, description="Max papers to re-parse in one batch.")


class ReparseResponse(BaseModel):
    """Response body for POST /api/v1/ingest/reparse."""

    total_found: int = 0
    pdfs_downloaded: int = 0
    pdfs_parsed: int = 0
    chunks_indexed: int = 0
    errors: list[str] = Field(default_factory=list)
    processing_time: float = 0.0
