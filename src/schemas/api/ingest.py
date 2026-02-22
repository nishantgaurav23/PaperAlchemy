"""
Request and response schemas for the ingestion API endpoints.

These are used by:
    - POST /api/v1/ingest/fetch  (Airflow Task 2 calls this)
    - POST /api/v1/ingest/index  (Airflow Task 3 calls this)

Keeping them in a dedicated file (not mixed with search/ask schemas)
makes it clear these endpoints are for pipeline orchestration, not
end-user queries.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    """Request body for POST /api/v1/ingest/fetch."""
    date: Optional[str] = Field(
        default=None,
        description="Target date in YYYYMMDD format. Defaults to yesterday.",
        examples=["20260221"],
    )
    max_results: int = Field(default=10, ge=1, le=100)
    process_pdfs: bool = Field(default=True, description="Download and parse PDFs via Docling.")


class FetchResponse(BaseModel):
    """Response from POST /api/v1/ingest/fetch."""
    target_date: str
    papers_fetched: int
    pdfs_downloaded: int
    pdfs_parsed: int
    papers_stored: int
    arxiv_ids: List[str]
    errors: List[str]
    processing_time: float


class IndexRequest(BaseModel):
    """Request body for POST /api/v1/ingest/index."""
    arxiv_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific paper IDs to index. If None, indexes papers from last since_hours.",
    )
    since_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Look back window when arxiv_ids is not provided.",
    )


class IndexResponse(BaseModel):
    """Response from POST /api/v1/ingest/index."""
    papers_processed: int
    chunks_created: int
    chunks_indexed: int
    errors: List[str]
