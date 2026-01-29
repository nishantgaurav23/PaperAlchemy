"""Search router for BM25 and hybrid search."""

import logging

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from src.dependency import OpenSearchDep
from src.schemas.api.search import SearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])

@router.get("/search")
async def search_papers_get(
    opensearch_client: OpenSearchDep,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    size: int = Query(default=10, ge=1, le=50, description="Number of results"),
    from_: int = Query(default=0, ge=0, alias="from", description="Pagination offset"),
    categories: Optional[List[str]] = Query(default=None, description="Filter by categories"),
    latest: bool = Query(default=False, description="Sort by date instead of relevance"),

) -> SearchResponse:
    """
    Simple search endpoint using GET method.

    Use this for quick keyword searches from browser or curl.
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Search servie is currently unavailable"
            )
        
        results = opensearch_client.search_papers(
            query=q,
            size=size,
            from_=from_,
            categories=categories,
            latest=latest,
        )

        hits = _format_hits(results)

        return SearchResponse(
            query=q,
            total=results.get("total", 0),
            hits=hits,
            size=size,
            **{"from": from_},
            search_mode="bm25",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
@router.post("/search", response_model=SearchResponse)
async def search_paper_post(
    request: SearchRequest,
    opensearch_client: OpenSearchDep,
) -> SearchResponse:
    """
    Advanced search endpoint using POST method.

    Use this for complex queries with filters and pagination.
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Search service is currently unavailable"
            )
        logger.info(
            f"Search request: query='{request.query}', "
            f"size={request.size}, categories={request.categories}"
        )

        results = opensearch_client.search_papers(
            query=request.query,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest= request.latest_papers,
        )

        hits = _format_hits(results)

        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode="bm25",
        )
        logger.info(f"Search completed: {search_response.total} results returned")
        return search_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    

def _format_hits(results: dict) -> List[SearchHit]:
    """
    Format raw OpenSearch hits into SearchHit models.

    Args:
        results: Raw search results from OpenSearch client

    Returns:
        List of formatted SearhHit objects
    """
    hits = []
    for hit in results.get("hits", []):
        hits.append(
            SearchHit(
                arxiv_id=hit.get("arxiv_id", ""),
                title=hit.get("title", ""),
                authors=hit.get("authors"),
                abstract=hit.get("abstract"),
                categories=hit.get("categories"),
                published_date=hit.get("published_date"),
                pdf_url=hit.get("pdf_url"),
                score=hit.get("score", 0.0),
                highlights=hit.get("highlights"),
                chunk_text=hit.get("chunk_text"),
                chunk_id=hit.get("chunk_id"),
                section_title=hit.get("section_title")
            )
        )

    return hits