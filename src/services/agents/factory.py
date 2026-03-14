"""Factory for creating AgenticRAGService instances (S6.7).

Provides a DI-friendly constructor for use with FastAPI Depends().
"""

from __future__ import annotations

from typing import Any

from src.services.agents.agentic_rag import AgenticRAGService


def create_agentic_rag_service(
    llm_provider: Any,
    retrieval_pipeline: Any | None = None,
    cache_service: Any | None = None,
    web_search_service: Any | None = None,
    arxiv_client: Any | None = None,
    **kwargs: Any,
) -> AgenticRAGService:
    """Create an AgenticRAGService with validated dependencies.

    Args:
        llm_provider: Required LLM provider instance.
        retrieval_pipeline: Optional retrieval pipeline.
        cache_service: Optional cache service.
        web_search_service: Optional web search service for fallback.
        arxiv_client: Optional arXiv client for live paper search.
        **kwargs: Additional keyword arguments passed to AgenticRAGService.

    Returns:
        Configured AgenticRAGService.

    Raises:
        ValueError: If llm_provider is None.
    """
    if llm_provider is None:
        raise ValueError("llm_provider is required and cannot be None")

    return AgenticRAGService(
        llm_provider=llm_provider,
        retrieval_pipeline=retrieval_pipeline,
        cache_service=cache_service,
        web_search_service=web_search_service,
        arxiv_client=arxiv_client,
        **kwargs,
    )
