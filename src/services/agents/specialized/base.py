"""Base class and result models for specialized agents (S6.8).

Defines the protocol that all specialized agents must implement,
plus common result models for structured output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem

logger = logging.getLogger(__name__)


class SpecializedAgentResult(BaseModel):
    """Base result returned by any specialized agent."""

    agent_name: str
    analysis: str
    sources: list[SourceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpecializedAgentBase(ABC):
    """Abstract base class for all specialized agents.

    Each specialized agent must implement:
    - ``name``: A human-readable agent identifier.
    - ``run()``: Execute the agent's analysis task.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this agent."""

    @abstractmethod
    async def run(
        self,
        query: str,
        context: AgentContext,
        papers: list[SourceItem] | None = None,
        **kwargs: Any,
    ) -> SpecializedAgentResult:
        """Execute the specialized agent's analysis.

        Args:
            query: The user's research question or analysis request.
            context: Runtime context with LLM provider and retrieval pipeline.
            papers: Optional pre-retrieved papers. If None, agent retrieves its own.
            **kwargs: Agent-specific keyword arguments.

        Returns:
            A SpecializedAgentResult (or subclass) with analysis and sources.
        """

    async def _retrieve_papers(self, query: str, context: AgentContext) -> list[SourceItem]:
        """Retrieve papers from the knowledge base via the retrieval pipeline.

        Used when no papers are provided to ``run()``.
        """
        if context.retrieval_pipeline is None:
            logger.warning("%s: No retrieval pipeline available", self.name)
            return []

        try:
            result = await context.retrieval_pipeline.retrieve(query, top_k=context.top_k)
            return [
                SourceItem(
                    arxiv_id=getattr(hit, "arxiv_id", ""),
                    title=getattr(hit, "title", ""),
                    authors=getattr(hit, "authors", []) or [],
                    url=getattr(hit, "pdf_url", "") or f"https://arxiv.org/abs/{getattr(hit, 'arxiv_id', '')}",
                    relevance_score=getattr(hit, "score", 0.0),
                    chunk_text=getattr(hit, "chunk_text", ""),
                )
                for hit in result.results
                if getattr(hit, "arxiv_id", "")
            ]
        except Exception as e:
            logger.error("%s: Retrieval failed: %s", self.name, e)
            return []
