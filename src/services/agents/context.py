"""Per-request runtime context for agent nodes.

Holds live service clients and request-scoped configuration,
injected into every LangGraph node via the context parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentContext:
    """Runtime context injected into every LangGraph node.

    Created per-request with live service instances and configuration overrides.
    Nodes access services via ``context.llm_provider``, ``context.retrieval_pipeline``, etc.
    """

    llm_provider: Any  # LLMProvider protocol — Any to avoid import issues at runtime
    retrieval_pipeline: Any | None = None  # RetrievalPipeline | None
    cache_service: Any | None = None  # CacheClient | None
    web_search_service: Any | None = None  # WebSearchService | None
    arxiv_client: Any | None = None  # ArxivClient | None
    model_name: str = ""
    temperature: float = 0.7
    top_k: int = 5
    max_retrieval_attempts: int = 3
    guardrail_threshold: int = 40
    trace_id: str | None = None
    user_id: str = "api_user"


def create_agent_context(*, llm_provider: Any, **overrides: Any) -> AgentContext:
    """Factory for creating an AgentContext with optional overrides.

    Args:
        llm_provider: Required LLMProvider instance.
        **overrides: Optional keyword arguments matching AgentContext fields.

    Returns:
        Configured AgentContext instance.
    """
    return AgentContext(llm_provider=llm_provider, **overrides)
