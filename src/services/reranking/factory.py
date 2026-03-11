"""Factory function for creating RerankerService instances (S4b.1)."""

from __future__ import annotations

from sentence_transformers import CrossEncoder
from src.config import RerankerSettings, get_settings
from src.exceptions import RerankerError
from src.services.reranking.service import RerankerService


def create_reranker_service(settings: RerankerSettings | None = None) -> RerankerService:
    """Create a RerankerService based on the provider configuration.

    Args:
        settings: RerankerSettings instance. If None, uses get_settings().

    Returns:
        Configured RerankerService.

    Raises:
        RerankerError: If the provider is unknown or model loading fails.
    """
    if settings is None:
        settings = get_settings().reranker

    if settings.provider == "local":
        try:
            model = CrossEncoder(settings.model, device=settings.device)
        except Exception as e:
            raise RerankerError(f"Failed to load cross-encoder model: {e}") from e
        return RerankerService(settings=settings, model=model, provider="local")

    if settings.provider == "cohere":
        return RerankerService(settings=settings, model=None, provider="cohere")

    raise RerankerError(f"Unknown reranker provider: {settings.provider}")
