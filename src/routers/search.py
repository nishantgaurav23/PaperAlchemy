"""Hybrid search endpoint: BM25 + KNN vector search with RRF fusion.

Orchestrates the OpenSearch client (S4.1) and Jina embedding service (S4.3)
to deliver hybrid search results. Gracefully degrades to BM25-only when
embeddings fail.

Also provides a live arXiv search endpoint for finding papers online.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.dependency import EmbeddingsDep, OpenSearchDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse
from src.services.arxiv.factory import make_arxiv_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------- arXiv web search schemas ----------


class ArxivSearchRequest(BaseModel):
    """Request body for POST /api/v1/search/arxiv."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    category: str | None = Field(default=None, description="arXiv category filter (e.g. cs.AI)")
    max_results: int = Field(default=20, ge=1, le=100, description="Max results to return")
    sort_by: str = Field(default="relevance", description="Sort by: relevance or submittedDate")


class ArxivSearchHit(BaseModel):
    """A single arXiv search result."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: str
    pdf_url: str
    arxiv_url: str


class ArxivSearchResponse(BaseModel):
    """Response body for POST /api/v1/search/arxiv."""

    query: str
    total: int
    hits: list[ArxivSearchHit]


def _map_hits(raw_hits: list[dict]) -> list[SearchHit]:
    """Transform raw OpenSearch hits into typed SearchHit models."""
    return [SearchHit(**hit) for hit in raw_hits]


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
) -> SearchResponse:
    """Hybrid search combining BM25 keyword search with KNN vector search.

    Falls back to BM25-only if embedding generation fails.
    Returns 503 if OpenSearch is unreachable.
    """
    # FR-6: Health check guard
    if not opensearch_client.health_check():
        raise HTTPException(status_code=503, detail="OpenSearch is unreachable")

    # FR-3: Query embedding with graceful fallback
    query_embedding: list[float] | None = None
    search_mode = "bm25"

    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            search_mode = "hybrid"
        except Exception as e:
            logger.warning("Embedding failed, falling back to BM25: %s", e)
            query_embedding = None
            search_mode = "bm25"

    # FR-4: Unified search execution
    results = await opensearch_client.asearch_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.size,
        from_=request.from_,
        categories=request.categories,
        latest=request.latest_papers,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=request.min_score or 0.0,
    )

    # FR-5: Result mapping
    hits = _map_hits(results.get("hits", []))

    return SearchResponse(
        query=request.query,
        total=results.get("total", 0),
        hits=hits,
        size=request.size,
        from_=request.from_,
        search_mode=search_mode,
    )


@router.post("/search/arxiv", response_model=ArxivSearchResponse)
async def arxiv_web_search(request: ArxivSearchRequest) -> ArxivSearchResponse:
    """Search arXiv API directly for papers online.

    This is a live web search — not limited to the local knowledge base.
    """
    arxiv_client = make_arxiv_client()

    sort_by = "submittedDate" if request.sort_by == "date" else "relevance"

    try:
        papers = await arxiv_client.fetch_papers(
            search_query=request.query,
            category=request.category,
            max_results=request.max_results,
            sort_by=sort_by,
            sort_order="descending",
            skip_default_category=request.category is None,
        )
    except Exception as e:
        logger.error("arXiv search failed: %s", e)
        raise HTTPException(status_code=502, detail=f"arXiv API error: {e}")

    hits = [
        ArxivSearchHit(
            arxiv_id=p.arxiv_id,
            title=p.title,
            authors=p.authors,
            abstract=p.abstract,
            categories=p.categories,
            published_date=p.published_date,
            pdf_url=p.pdf_url,
            arxiv_url=p.arxiv_url,
        )
        for p in papers
    ]

    return ArxivSearchResponse(
        query=request.query,
        total=len(hits),
        hits=hits,
    )


@router.get("/search/health")
async def search_health(opensearch_client: OpenSearchDep) -> dict:
    """Quick health check for the search subsystem."""
    healthy = opensearch_client.health_check()
    return {
        "status": "ok" if healthy else "degraded",
        "opensearch": healthy,
    }
