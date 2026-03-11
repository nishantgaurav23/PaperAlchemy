"""Summarizer specialized agent (S6.8).

Generates structured summaries of academic papers covering:
objective, methodology, key findings, contributions, and limitations.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.specialized.base import SpecializedAgentBase

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """You are a research paper summarizer. Analyze the provided papers and produce a structured summary.

## Papers
{papers_text}

## User Query
{query}

Produce a structured summary with these sections:
1. **Objective**: What problem does this research address?
2. **Methodology**: What approach/methods are used?
3. **Key Findings**: What are the main results? (list 2-5 items)
4. **Contributions**: What are the novel contributions? (list 1-4 items)
5. **Limitations**: What are the limitations? (list 1-3 items)
6. **Summary**: A 2-4 sentence overall summary in markdown.

Always cite papers by their title and arXiv ID. Use only information from the provided papers."""


class SummarizerResult(BaseModel):
    """Structured paper summary output."""

    objective: str = ""
    methodology: str = ""
    key_findings: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary_text: str = ""
    sources: list[SourceItem] = Field(default_factory=list)


class SummarizerAgent(SpecializedAgentBase):
    """Generates structured summaries of academic papers."""

    @property
    def name(self) -> str:
        return "summarizer"

    async def run(
        self,
        query: str,
        context: AgentContext,
        papers: list[SourceItem] | None = None,
        **kwargs: Any,
    ) -> SummarizerResult:
        """Summarize papers relevant to the query.

        If no papers are provided, retrieves them from the knowledge base first.
        """
        if papers is None:
            papers = await self._retrieve_papers(query, context)

        if not papers:
            return SummarizerResult(
                summary_text="No relevant papers found in the knowledge base for this query.",
                sources=[],
            )

        papers_text = self._format_papers(papers)
        prompt = SUMMARIZER_PROMPT.format(papers_text=papers_text, query=query)

        try:
            model = context.llm_provider.get_langchain_model(
                model=context.model_name,
                temperature=context.temperature,
            )
            structured_model = model.with_structured_output(SummarizerResult)
            result: SummarizerResult = await structured_model.ainvoke(prompt)
            result.sources = papers
            return result
        except Exception as e:
            logger.error("SummarizerAgent LLM call failed: %s", e)
            return SummarizerResult(
                summary_text=f"Error generating summary: {e}",
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
