"""Fact-checker specialized agent (S6.8).

Cross-references claims against papers in the knowledge base,
verifying with supporting or contradicting evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.specialized.base import SpecializedAgentBase

logger = logging.getLogger(__name__)

FACT_CHECK_PROMPT = """You are a research fact-checker. Verify claims against the provided academic papers.

## Papers
{papers_text}

## User Query
{query}

For each claim in the query:
1. Identify the specific claim being made.
2. Search the provided papers for supporting or contradicting evidence.
3. Assign a verdict: "supported", "contradicted", or "insufficient_evidence".
4. Provide an explanation citing specific papers.

Only use evidence from the provided papers. If the papers don't address a claim, mark it as "insufficient_evidence"."""


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    evidence: list[SourceItem] = Field(default_factory=list)
    explanation: str = ""


class FactCheckResult(BaseModel):
    """Complete fact-check analysis output."""

    claims: list[ClaimVerification] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    overall_assessment: str = ""


class FactCheckerAgent(SpecializedAgentBase):
    """Verifies factual claims by cross-referencing with papers in the knowledge base."""

    @property
    def name(self) -> str:
        return "fact_checker"

    async def run(
        self,
        query: str,
        context: AgentContext,
        papers: list[SourceItem] | None = None,
        **kwargs: Any,
    ) -> FactCheckResult:
        """Verify claims in the query against available papers.

        If no papers are provided, retrieves them from the knowledge base first.
        """
        if papers is None:
            papers = await self._retrieve_papers(query, context)

        if not papers:
            return FactCheckResult(
                claims=[
                    ClaimVerification(
                        claim=query,
                        verdict="insufficient_evidence",
                        explanation="No relevant papers found in the knowledge base.",
                    )
                ],
                overall_assessment="No relevant papers found to verify this claim.",
            )

        papers_text = self._format_papers(papers)
        prompt = FACT_CHECK_PROMPT.format(papers_text=papers_text, query=query)

        try:
            model = context.llm_provider.get_langchain_model(
                model=context.model_name,
                temperature=0.3,
            )
            structured_model = model.with_structured_output(FactCheckResult)
            result: FactCheckResult = await structured_model.ainvoke(prompt)
            result.sources = papers
            return result
        except Exception as e:
            logger.error("FactCheckerAgent LLM call failed: %s", e)
            return FactCheckResult(
                claims=[
                    ClaimVerification(
                        claim=query,
                        verdict="insufficient_evidence",
                        explanation=f"Error during fact-checking: {e}",
                    )
                ],
                sources=papers,
                overall_assessment=f"Error during fact-checking: {e}",
            )

    @staticmethod
    def _format_papers(papers: list[SourceItem]) -> str:
        """Format papers into numbered text for the prompt."""
        parts = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors) if p.authors else "Unknown"
            parts.append(f"[{i}] {p.title} — {authors_str}\n    arXiv: {p.url}\n    Content: {p.chunk_text[:500]}")
        return "\n\n".join(parts)
