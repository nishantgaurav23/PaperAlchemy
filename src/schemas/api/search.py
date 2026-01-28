"""Search request model."""

from typing import List, Optional

from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query across title, abstract, and authors"
    )
    size: int = Field(
        default=10,
        ge=1, le=50,
        description="Number of results to return"
    )
    from_: int = Field(
        default=0,
        ge=0,
        alias="from",
        description="Offset for pagination"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by arXiv categories (e.g., ['cs.AI', 'cs.LG'])"
    )
    latest_papers: bool = Field(
        default=False,
        description="Sort by publication date (newest first) instead of relevance"
    )

    class Config:
        populate_by_name = True

class SearchHit(BaseModel):
    """Individual search result."""

    arxiv_id: str
    title: str
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    categories: Optional[List[str]] = None
    published_date: Optional[str] = None
    pdf_url: Optional[str] = None
    score: float
    highlights: Optional[dict] = None

    # Chunk-specific fields (for hybrid search in Week 4+)
    chunk_text: Optional[str] = Field(
        None,
        description="Text content of the matching chunk"
    )
    chunk_id: Optional[str] = Field(
        None,
        description="Unique identifier for the chunk"
    )
    section_title: Optional[str] = Field(
        None,
        description="Setion name where the chunk was found"
    )

class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    total: int
    hits: List[SearchHit]
    size: int= Field(description="Number of results requested")
    from_: int = Field(alias="from", description="Offset used for pagination")
    search_mode: Optional[str] = Field(
        None,
        description="Search mode used: bm25, vector, or hybrid"
    )
    error: Optional[str] = None

    class Config:
        populate_by_name = True