"""Trend analyzer specialized agent (S6.8).

Analyzes research trends across papers in the knowledge base,
identifying emerging topics, methodological shifts, and key papers.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.specialized.base import SpecializedAgentBase

logger = logging.getLogger(__name__)

TREND_ANALYSIS_PROMPT = """You are a research trend analyst. Analyze the provided papers to identify research trends.

## Papers
{papers_text}

## User Query
{query}

Analyze these papers to identify:
1. **Trends**: Major research directions. For each, specify:
   - Topic name
   - Direction: "rising" (gaining traction), "stable" (consistent), or "declining" (fading)
   - Key papers driving this trend
   - Brief description
2. **Timeline**: A brief chronological narrative of how the field has evolved.
3. **Emerging Topics**: New areas that are just starting to appear.

Base your analysis ONLY on the provided papers. Cite specific papers by title and arXiv ID."""


class TrendItem(BaseModel):
    """A single identified research trend."""

    topic: str
    direction: Literal["rising", "stable", "declining"]
    key_papers: list[SourceItem] = Field(default_factory=list)
    description: str = ""


class TrendAnalysisResult(BaseModel):
    """Complete trend analysis output."""

    trends: list[TrendItem] = Field(default_factory=list)
    timeline: str = ""
    emerging_topics: list[str] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)


class TrendAnalyzerAgent(SpecializedAgentBase):
    """Analyzes research trends across papers in the knowledge base."""

    @property
    def name(self) -> str:
        return "trend_analyzer"

    async def run(
        self,
        query: str,
        context: AgentContext,
        papers: list[SourceItem] | None = None,
        **kwargs: Any,
    ) -> TrendAnalysisResult:
        """Analyze research trends from available papers.

        If no papers are provided, retrieves them from the knowledge base first.
        """
        if papers is None:
            papers = await self._retrieve_papers(query, context)

        if not papers:
            return TrendAnalysisResult(
                timeline="No relevant papers found in the knowledge base for trend analysis.",
            )

        papers_text = self._format_papers(papers)
        prompt = TREND_ANALYSIS_PROMPT.format(papers_text=papers_text, query=query)

        try:
            model = context.llm_provider.get_langchain_model(
                model=context.model_name,
                temperature=context.temperature,
            )
            structured_model = model.with_structured_output(TrendAnalysisResult)
            result: TrendAnalysisResult = await structured_model.ainvoke(prompt)
            result.sources = papers
            return result
        except Exception as e:
            logger.error("TrendAnalyzerAgent LLM call failed: %s", e)
            return TrendAnalysisResult(
                timeline=f"Error analyzing trends: {e}",
                sources=papers,
            )

    @staticmethod
    def _format_papers(papers: list[SourceItem]) -> str:
        """Format papers into numbered text for the prompt."""
        parts = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors) if p.authors else "Unknown"
            parts.append(f"[{i}] {p.title} — {authors_str}\n    arXiv: {p.url}\n    Content: {p.chunk_text[:500]}")
        return "\n\n".join(parts)
