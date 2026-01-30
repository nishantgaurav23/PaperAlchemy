"""
Factory functions for creating Jina embeddings clients.

Why it's needed:
    The JinaEmbeddingsClient requires an API key from settings. Rather than
    having every caller read settings and construct the client themselves,
    the factory centralizes this logic. This ensures consistent initialization
    and makes it easy to swap implementations (e.g., mock client for testing).

What it does:
    - make_embeddings_service(): Creates a new JinaEmbeddingsClient instance
      using the API key from application settings. Creates a fresh instance
      each time to avoid issues with closed HTTP connections in async contexts.

How it helps:
    - Single place to change if we switch from Jina to OpenAI embeddings
    - Dependency injection: FastAPI dependencies call the factory
    - Testing: can be patched to return a mock client
    - No stale connections: fresh client per request avoids async lifecycle bugs
"""

from typing import Optional

from src.config import Settings, get_settings
from .jina_client import JinaEmbeddingsClient


def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Create a new Jina embeddings client instance.

    Creates a fresh client each time to avoid closed-client issues in
    async contexts. The httpx.AsyncClient inside JinaEmbeddingsClient
    manages its own connection pool.

    Args:
        settings: Optional Settings instance. If None, loads from
                  environment variables via get_settings(). Pass
                  explicitly in tests or when settings are already
                  available to avoid redundant loading.

    Returns:
        JinaEmbeddingsClient configured with the API key from settings.

    Example:
        # In a FastAPI dependency
        client = make_embeddings_service()
        vectors = await client.embed_query("search term")
        await client.close()
    """
    if settings is None:
        settings = get_settings()

    # Read the Jina API key from settings (JINA_API_KEY env var)
    api_key = settings.jina_api_key

    return JinaEmbeddingsClient(api_key=api_key)


def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Alias for make_embeddings_service().

    Provided for consistency with other factories that use the
    'make_*_client' naming convention (e.g., make_opensearch_client).

    Args:
        settings: Optional Settings instance.

    Returns:
        JinaEmbeddingsClient instance.
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key

    return JinaEmbeddingsClient(api_key=api_key)
