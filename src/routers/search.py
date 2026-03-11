"""Hybrid search endpoint: BM25 + KNN vector search with RRF fusion.

Orchestrates the OpenSearch client (S4.1) and Jina embedding service (S4.3)
to deliver hybrid search results. Gracefully degrades to BM25-only when
embeddings fail.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.dependency import EmbeddingsDep, OpenSearchDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()


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
    results = opensearch_client.search_unified(
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


@router.get("/search/health")
async def search_health(opensearch_client: OpenSearchDep) -> dict:
    """Quick health check for the search subsystem."""
    healthy = opensearch_client.health_check()
    return {
        "status": "ok" if healthy else "degraded",
        "opensearch": healthy,
    }
