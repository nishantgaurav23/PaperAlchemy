"""Factory functions for creating OpenSearch clients."""

from __future__ import annotations

from functools import lru_cache

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Settings | None = None) -> OpenSearchClient:
    """Cached singleton OpenSearch client for app use."""
    if settings is None:
        settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Settings | None = None, host: str | None = None) -> OpenSearchClient:
    """Fresh (non-cached) OpenSearch client for notebooks/tests."""
    if settings is None:
        settings = get_settings()
    opensearch_host = host or settings.opensearch.host
    return OpenSearchClient(host=opensearch_host, settings=settings)
