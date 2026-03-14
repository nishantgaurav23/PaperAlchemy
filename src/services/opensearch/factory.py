"""Factory functions for creating OpenSearch clients."""

from __future__ import annotations

from src.config import Settings, get_settings

from .client import OpenSearchClient

_cached_client: OpenSearchClient | None = None


def make_opensearch_client(settings: Settings | None = None) -> OpenSearchClient:
    """Cached singleton OpenSearch client for app use."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    if settings is None:
        settings = get_settings()
    _cached_client = OpenSearchClient(host=settings.opensearch.host, settings=settings)
    return _cached_client


def make_opensearch_client_fresh(settings: Settings | None = None, host: str | None = None) -> OpenSearchClient:
    """Fresh (non-cached) OpenSearch client for notebooks/tests."""
    if settings is None:
        settings = get_settings()
    opensearch_host = host or settings.opensearch.host
    return OpenSearchClient(host=opensearch_host, settings=settings)
