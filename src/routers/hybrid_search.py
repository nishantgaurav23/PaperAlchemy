"""
Hybrid search router - BM25 + vector search with RRF fusion.

Why it's needed:
    BM25 keyword search misses conceptually similar papers that use
    different terminology. Hybrid search combines BM25 with vector similarity
    so "deep learning optimization" also finds paper about "neural network 
    training" even when those exact words don't appear.

What it does:
    - POST /hybrid-searc/: Accepts a query, optionally embeds it via Jina,
      then runs a unified search on OpenSearch that combines:
      1. BM25 text matching (keyword relevance)
      2. KNN vector search (semantic similarity)
      3. RRF fusion (merges both ranked lists without manual weight tuning)
    - Graceful degradation: if Jina embedding fails (API down, rate limited),
      automatically falls back to BM25-only search and logs a warning.
    - Returns search_mode in the response so the client knows which mode
      was actually used ("hybrid" or "bm25).

How it helps:
    - Semantic search: "attention mechanism" finds "transformer self-attention"
    - Fault tolerant: embedding failures don't break search entirely
    - Transparent: search_model tells the client what happened
    - Chunk-level results: returns the specific passages that matched, not
      just the paper- enabling precise answer extraction later.

Architecture:
    Request → embed query (Jina) → search_unified (OpenSearch)                                                                   
                                          ↓                                                                                        
                                BM25 subquery + KNN subquery                                                                       
                                          ↓                                                                                        
                                RRF pipeline fuses scores                                                                          
                                          ↓                                                                                        
                                SearchResponse with hits

Key design points:                                                                                                               
                                                                                                                                   
  - Graceful degradation — The try/except around embed_query means if Jina is down, search still works via BM25. Users get results 
    either way.                                                                                                                      
  - search_unified — This is a method you'll need on your OpenSearchClient (likely Week 4 continuation). It runs both BM25 and KNN 
    subqueries and applies the RRF pipeline.                                                                                         
  - **{"from": request.from_} — Workaround because from is a Python keyword but Pydantic uses it as the JSON alias.                
  - section_title vs section_name — Note your SearchHit model uses section_title while the reference uses section_name.

"""

import logging

from fastapi import APIRouter, HTTPException

from src.dependency import EmbeddingsDep, OpenSearchDep
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse

logger = logging.gerLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags="hybrid-search")

@router.post("/", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
) -> SearchResponse:
    """Execute hybrid search combining BM25 keyword matching with vector similarity.

    Flow:
        1. Health check OpenSearch (return 503 if down)
        2. If use_hybrid=True, embed the query via Jina API
        3. If embedding fails, fall back to BM25 (Don't error out)
        4. Call opensearch_client.search_unified() with query + optional embedding
        5. Map raw OpenSearch hits to SearchHit response models
        6. Return SearchResponse with search_mode indicating what was used.
    """
    try:
        # Step 1: Verify OpenSearch is reachable
        if not opensearch_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Search service is currently unavailable"
            )
        
        # Step 2: Generate query embedding (if hybrid)
        # Embedding the query converts it to a 1024-dim vector for
        # KNN similarity search against pre-indexed chunk embeddings
        query_embedding = None
        if request.use_hybrid:
            try:
                query_embedding = await embeddings_service.embed_query(
                    request.query
                )
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                # Graceful degradation: if Jina is down, fall back to BM25
                # This ensures search nevery fully breaks due to embedding issues
                logger.warning(
                    f"Failed to generate embeddings, falling back to BM25: {e}"
                )
                query_embedding = None

        # Determine actual search mode for logging and response
        is_hybrid = request.use_hybrid and query_embedding is not None
        logger.info(
            f"Hybrid search: '{request.query}' (hybrid: {is_hybrid})"
        )

        # Step 3: Execute unified search
        # search_unified handles both BM25 and KNN subqueries,
        # using the RRF pipeline to fuse results when embedding exisits
        results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=request.latest_papers,
            use_hybrid=request.use_hybrid,
            min_score=request.min_score,
        )

        # Step 4: Map OpenSearch hits to response models
        hits = []
        for hit in results.get("hits", []):
            hits.append(
                SearchHit(
                    arxiv_id=hit.get("arxiv_id"),
                    title=hit.get("title", ""),
                    authors=hit.get("authors"),
                    abstract=hit.get("published_date"),
                    pdf_url=hit.get("pdf_url"),
                    score=hit.get("score", 0.0),
                    highlights=hit.get("highlights"),
                    # Chunk-level fields (populated for hybrid search)
                    chunk_text=hit.get("chunk_text"),
                    chunk_id=hit.get("chunk_id"),
                    section_title=hit.get("section_title")
                )
            )
        # Step 5: Build response
        search_response = SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            size=request.size,
            **{"from": request.from_},
            search_mode="hybrid" if is_hybrid else "bm25",
        )
        logger.info(
            f"Search completed: {search_response.total} results returned"
        )
        return search_response
    except HTTPException:
        # Re-raise HTTP exceptions (503, etc.) without wrapping
        raise
    except Exception as e:
        logger.error(f"Hybrid searh error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )