"""Request/response schemas for the hybrid search endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HybridSearchRequest(BaseModel):
    """Request body for POST /api/v1/search."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    size: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    from_: int = Field(default=0, ge=0, description="Pagination offset", alias="from")
    categories: list[str] | None = Field(default=None, description="Filter by arXiv categories")
    use_hybrid: bool = Field(default=True, description="Use hybrid BM25+KNN search (True) or BM25-only (False)")
    latest_papers: bool = Field(default=False, description="Sort by latest papers first")
    min_score: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum score threshold")

    model_config = {"populate_by_name": True}


class SearchHit(BaseModel):
    """A single search result hit."""

    arxiv_id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    score: float = 0.0
    highlights: dict = Field(default_factory=dict)
    chunk_text: str = ""
    chunk_id: str = ""
    section_title: str | None = None


class SearchResponse(BaseModel):
    """Response body for POST /api/v1/search."""

    query: str
    total: int
    hits: list[SearchHit]
    size: int
    from_: int = Field(alias="from", default=0)
    search_mode: str  # "hybrid" or "bm25"

    model_config = {"populate_by_name": True}
