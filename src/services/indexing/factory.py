"""
Factory function for creating the HybridIndexingServie with all dependencies.

Why it's needed:
    HybridIndexingService requires three dependencies: TextChunker,
    JinaEmbeddingsClient, and OpenSearchClient. Each of these needs 
    configuration from Settings. Without a factory, every call site
    (notebooks, routers, scripts) would repeat the same wiring logic - 
    reading settings, creating each dependency, passing them in.

What it does:
    - make hybrid_indexing_servie(): Creates a fully-wired
      HybridIndexService in one call. Reads chunking config from
      settings, creates the embedding client, creates the OpenSearch
      client, and assembles them into the ndexing service.

How it helps:
    - Single call to get a ready-to-use indexing service
    - opensearch_host override for notebooks (localhost: 9201 vs Docker)
    - All configuration flows from Settings - change once, applied everywhere
    - Easy to mock in tests: patch this factory instead of three constructors
"""

from typing import Optional

from src.config import Settings, get_settings
from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client_fresh

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker

def make_hybrid_indexing_service(
        settings: Optional[Settings] = None,
        opensearch_host: Optional[str] = None
) -> HybridIndexingService:
    """Created a fully-wired HybridIndexingService instance.
    
    Assembles all three dependencies from settings and returns a
    ready-to-use service.Creates fresh instances each time to avoid
    stale connections in async contexts.

    Args:
        settings: Optional Settings instance. If None, loads from
                  environment via get_settings().
        opensearch_host: Optional host override for OpenSearch.
                         Use "http://localhost:9201" in notebooks
                         (Docker exposes port 9201 on the host).
                         If None, uses settings.opensearch
                        
    Returns:
        HybridIndexingService wired with:
        - TextChunker configured from settings.chunking
        - JinaEmbeddingsClient configured from settings.jina_api_key
        - OpenSearchClient configured from settings.opensearch

    Example:
        # In a notebook
        service = make_hybrid_indexing_service(
                opensearch_host="http://locahost:9201
        )
        stats = await service.index_paper(paper_data)

        # In FastAPI (uses Docker service names from settings)
        servie = make_hybrid_indexing_service()
    """
    if settings is None:
        settings = get_settings()

    # Create TextChunker from chunking settings
    # (chunk_size=600, overlap_size=100, min_chunk=100)
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,   
    )

    # Create Jina embeddings client (read jina_api_key from settings)
    embeddings_client = make_embeddings_client(settings)

    # Create OpenSearch client(fresh instance, not cached singleton)
    # opensearch_host override is for notebook use (localhost vs Dc=ocker)
    """Why make opensearch_client_fresh instead of opensearch_client?
    The cached singleton (make_opensearch_client) use @lru_cache - it returns the same instance every time.
    That's fine for the FastAPI app (one process, one client). But in notebooks or batch scripts, you may need different
    hosts of fresh connections, so the facvtory uses the non-cached version.
    """
    opensearch_client = make_opensearch_client_fresh(
        settings, host= opensearch_host
    )

    # Assemble and return the fully-wired service
    return HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client
    )