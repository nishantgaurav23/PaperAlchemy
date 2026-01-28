"""Unified factory for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings
from .client import OpenSearchClient

@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """
    Factory function to create a cached OpenSearch client.

    Use lru_cache to maintain a singleton instance for efficiency.

    Args:
        settings: Optional settings instance

    Returns:
    Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings)

def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """
    Factory function to create a fresh OpenSearch client (not cached).
    
    Use this when you need a new client instance (e.g., for testing or 
    when connection issue occur).

    Args:
        settings: Optional settings instance
        host: Optional host override

    Returns:
        New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided host or setting host
    opensearch_host = host or settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings)
