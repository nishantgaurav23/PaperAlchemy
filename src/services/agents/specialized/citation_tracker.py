"""Citation tracker specialized agent (S6.8).

Tracks citation relationships between papers, finding papers that cite
a given paper, papers cited by it, and mapping influence within the knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.specialized.base import SpecializedAgentBase

logger = logging.getLogger(__name__)

CITATION_TRACK_PROMPT = """You are a citation tracker for academic papers. Analyze citation relationships.

## Papers
{papers_text}

## User Query
{query}

Analyze the provided papers to identify:
1. **Target Paper**: The paper the user is asking about.
2. **Cited By**: Papers in the collection that reference/cite the target paper.
3. **References**: Papers in the collection that the target paper cites.
4. **Citation Count**: How many papers in this collection cite the target.
5. **Influence Summary**: A brief assessment of the target paper's influence.

Look for explicit references, mentions by name, and methodological dependencies.
Base your analysis ONLY on the provided papers."""


class CitationTrackResult(BaseModel):
    """Citation relationship analysis output."""

    target_paper: SourceItem
    cited_by: list[SourceItem] = Field(default_factory=list)
    references: list[SourceItem] = Field(default_factory=list)
    citation_count: int = 0
    influence_summary: str = ""
    sources: list[SourceItem] = Field(default_factory=list)


class CitationTrackerAgent(SpecializedAgentBase):
    """Maps citation relationships between papers in the knowledge base."""

    @property
    def name(self) -> str:
        return "citation_tracker"

    async def run(
        self,
        query: str,
        context: AgentContext,
        papers: list[SourceItem] | None = None,
        **kwargs: Any,
    ) -> CitationTrackResult:
        """Track citation relationships for papers relevant to the query.

        If no papers are provided, retrieves them from the knowledge base first.
        """
        if papers is None:
            papers = await self._retrieve_papers(query, context)

        if not papers:
            return CitationTrackResult(
                target_paper=SourceItem(
                    arxiv_id="unknown",
                    title="Unknown Paper",
                    authors=[],
                    url="",
                ),
                influence_summary="No relevant papers found in the knowledge base.",
            )

        papers_text = self._format_papers(papers)
        prompt = CITATION_TRACK_PROMPT.format(papers_text=papers_text, query=query)

        try:
            model = context.llm_provider.get_langchain_model(
                model=context.model_name,
                temperature=0.3,
            )
            structured_model = model.with_structured_output(CitationTrackResult)
            result: CitationTrackResult = await structured_model.ainvoke(prompt)
            result.sources = papers
            return result
        except Exception as e:
            logger.error("CitationTrackerAgent LLM call failed: %s", e)
            return CitationTrackResult(
                target_paper=papers[0]
                if papers
                else SourceItem(
                    arxiv_id="unknown",
                    title="Unknown",
                    authors=[],
                    url="",
                ),
                influence_summary=f"Error tracking citations: {e}",
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
