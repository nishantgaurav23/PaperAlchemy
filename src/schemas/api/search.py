"""
Search request/response schemas for BM25 and hybrid search endpoints.

Why it's needed:
    Pydantic models validate incoming search requests (query length, page size
    bounds) and structure outgoing responses. Without these, invalid queries
    like empty strings or size=99999 would reach OpenSearch and either error
    or overload the cluster.

What it does:
    - SearchRequest: Validates POST body with query, size (1-50), pagination,
      category filters, and sort preference. Uses alias "from" for JSON
      compatibility (from_ in Python since 'from' is a keyword).
    - SearchHit: One search result with paper metadata + optional chunk fields
      (chunk_text, chunk_id, section_title) for Week 4+ hybrid search.
    - SearchResponse: Wraps hits with total count, pagination info, search_mode
      (bm25/vector/hybrid), and optional error message.

How it helps:
    - Input validation: rejects bad queries before hitting OpenSearch
    - API documentation: FastAPI generates Swagger UI from these schemas
    - Type safety: routers return typed objects, not raw dicts
    - Forward-compatible: chunk fields are Optional, ready for Week 4
"""

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

class HybridSearchRequest(BaseModel):
    """Request model for hybrid search supporting BM25, vector, and hybrid modes.

    Why it's needed:
        The hybrid search endpoint needs additional controls beyond basic
        SearchRequest: toggling hybrid model on/off (fallback to pure BM25),
        and filtering low-confidence results via min_score. This keeps the 
        original SearchRequest unchanged for backward compatibility.

    What it does:
        - use_hybrid: When True, the router embeds the uery via Jina and
          runs both B25 + KNN search with RRF fusion. When False, falls
          back to pure BM25 KEYWORD SEARCH (Useful when Jina is down).
        - min_score: Filters out results below this threshold. hYBRID rrf
          scores are typical 0.0-0.03, so even 0.001 can remove noise.
        - size allows up to 100 (vs 50 for basic search) because hybrid
          search returns chunk-level results which may need deduplication.

    How it helps:
        - Graceful degradation: set use_hybrid=False if embeddings API fails
        - Quality control: min_score removes low-relevance noise
        - Swagger docs: json_schema_extra provides a working example
    """

    query: str = Field(
        ...,
        description="Search query text",
        min_length=1,
        max_length=500
    )
    size:int = Field(
        10,
        description="Number of results to return",
        ge=1, le=100
    )
    from_: int = Field(
        0,
        description="Offset for pagination",
        ge=0,
        alias="from"
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Filter by arXiv categories (e.g., ['cs.AI','cs.LG'])"
    )
    latest_papers: bool = Field(
        False,
        description="Sort by publication date instead of relevance"
    )
    use_hybrid: bool = Field(
        False,
        description="Enable hybrid search (BM25 + vector) with automatic "
                    "embedding generation. Set False to use BM25 only"               
    )
    min_score: float = Field(
        0.0,
        description="Minimum score threshold for results",
        ge=0.0
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "query": "machine learning neural networks",
                "size": 10,
                "categories": ["cs.AI", "cs.LG"],
                "latest_papers": False,
                "use_hybrid": True,
            }
        }

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