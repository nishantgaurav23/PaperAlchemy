"""Factory for creating Jina embeddings clients from settings."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.exceptions import ConfigurationError
from src.services.embeddings.client import JinaEmbeddingsClient


def make_embeddings_client(settings: Settings | None = None) -> JinaEmbeddingsClient:
    """Create a JinaEmbeddingsClient from application settings.

    Raises ConfigurationError if the Jina API key is not configured.
    """
    if settings is None:
        settings = get_settings()

    jina = settings.jina
    if not jina.api_key:
        raise ConfigurationError(detail="Jina API key is not configured — set JINA__API_KEY")

    return JinaEmbeddingsClient(
        api_key=jina.api_key,
        model=jina.model,
        dimensions=jina.dimensions,
        timeout=jina.timeout,
    )
