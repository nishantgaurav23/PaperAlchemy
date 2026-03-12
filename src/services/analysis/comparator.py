"""Side-by-side paper comparison service: compare 2+ papers via LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.schemas.api.analysis import ComparedPaper, ComparisonResponse, PaperComparison

if TYPE_CHECKING:
    from src.repositories.paper import PaperRepository
    from src.services.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

MAX_TOTAL_WORDS = 6000

COMPARISON_PROMPT = """You are a research paper analyst. Compare the following academic papers side-by-side.

Return ONLY valid JSON with this exact structure (no markdown fences, no explanation):
{{
    "methods_comparison": "How the approaches/methodologies differ and overlap (2-4 sentences)",
    "results_comparison": "Comparative analysis of results and findings (2-4 sentences)",
    "contributions_comparison": "What each paper uniquely contributes (2-4 sentences)",
    "limitations_comparison": "Comparative limitations and gaps (2-4 sentences)",
    "common_themes": ["theme 1", "theme 2"],
    "key_differences": ["difference 1", "difference 2"],
    "verdict": "Brief overall synthesis — which excels at what, how they complement each other (2-3 sentences)"
}}

Rules:
- methods_comparison, results_comparison, contributions_comparison, limitations_comparison: 2-4 sentences each
- common_themes: 1-5 items — shared themes, topics, or approaches
- key_differences: 1-5 items — most notable differences
- verdict: 2-3 sentences synthesizing the comparison
- Reference specific papers by their titles
- Be specific — reference actual methods, datasets, and metrics when available

Papers to compare:
{content}
"""


class ComparatorService:
    """Compares 2+ academic papers side-by-side using an LLM."""

    def __init__(self, llm_provider: LLMProvider, paper_repo: PaperRepository) -> None:
        self._llm = llm_provider
        self._repo = paper_repo

    async def compare(
        self,
        paper_ids: list[uuid.UUID],
        *,
        force: bool = False,
    ) -> ComparisonResponse:
        """Generate a structured side-by-side comparison of 2+ papers.

        Args:
            paper_ids: UUIDs of papers to compare (2-5, deduplicated).
            force: If True, always regenerate (skip any cache).

        Returns:
            ComparisonResponse with structured comparison + metadata.

        Raises:
            ValueError: Fewer than 2 or more than 5 unique paper IDs.
            PaperNotFoundError: Any paper not in database.
            InsufficientContentError: Any paper has no usable content.
            LLMServiceError: LLM provider failure.
        """
        # Deduplicate while preserving order
        seen: set[uuid.UUID] = set()
        unique_ids: list[uuid.UUID] = []
        for pid in paper_ids:
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)

        if len(unique_ids) < 2:
            raise ValueError("Comparison requires at least 2 unique paper IDs")
        if len(unique_ids) > 5:
            raise ValueError("Comparison supports at most 5 papers")

        # Fetch all papers
        papers = []
        for pid in unique_ids:
            paper = await self._repo.get_by_id(pid)
            if paper is None:
                raise PaperNotFoundError(f"Paper {pid} not found")

            has_abstract = bool(paper.abstract and paper.abstract.strip())
            has_sections = bool(paper.sections)
            has_pdf = bool(paper.pdf_content and paper.pdf_content.strip()) if hasattr(paper, "pdf_content") else False

            if not has_abstract and not has_sections and not has_pdf:
                raise InsufficientContentError(f"Paper {pid} has no abstract, sections, or PDF content")

            papers.append(paper)

        # Build prompt
        content = self._extract_multi_content(papers)
        prompt = COMPARISON_PROMPT.format(content=content)

        # Call LLM
        try:
            response = await self._llm.generate(prompt, temperature=0.3, max_tokens=1500)
        except Exception as e:
            raise LLMServiceError(f"Comparison generation failed: {e}") from e

        # Parse output
        comparison_data, warning = self._parse_comparison(response.text)

        # Build compared paper metadata
        compared_papers = [
            ComparedPaper(
                id=p.id,
                title=p.title,
                authors=p.authors if isinstance(p.authors, list) else [str(p.authors)] if p.authors else [],
            )
            for p in papers
        ]

        comparison = PaperComparison(papers=compared_papers, **comparison_data)

        return ComparisonResponse(
            paper_ids=unique_ids,
            comparison=comparison,
            model=response.model,
            provider=response.provider,
            latency_ms=response.usage.latency_ms if response.usage else None,
            warning=warning,
        )

    def _extract_multi_content(self, papers: list) -> str:
        """Format content from multiple papers with per-paper labels.

        Truncates each paper proportionally to stay within MAX_TOTAL_WORDS total.
        """
        n = len(papers)
        per_paper_limit = MAX_TOTAL_WORDS // n

        parts: list[str] = []
        for i, paper in enumerate(papers, 1):
            paper_parts: list[str] = []
            paper_parts.append(f"--- Paper {i}: {paper.title} ---")

            if hasattr(paper, "authors") and paper.authors:
                authors_str = ", ".join(paper.authors) if isinstance(paper.authors, list) else str(paper.authors)
                paper_parts.append(f"Authors: {authors_str}")

            if paper.abstract and paper.abstract.strip():
                paper_parts.append(f"Abstract: {paper.abstract.strip()}")

            if paper.sections:
                for section in paper.sections:
                    title = section.get("title", "") if isinstance(section, dict) else getattr(section, "title", "")
                    content = section.get("content", "") if isinstance(section, dict) else getattr(section, "content", "")
                    if content and content.strip():
                        paper_parts.append(f"## {title}\n{content.strip()}")

            elif hasattr(paper, "pdf_content") and paper.pdf_content and paper.pdf_content.strip():
                paper_parts.append(f"Full Text:\n{paper.pdf_content.strip()}")

            paper_text = "\n".join(paper_parts)
            words = paper_text.split()
            if len(words) > per_paper_limit:
                paper_text = " ".join(words[:per_paper_limit])

            parts.append(paper_text)

        return "\n\n".join(parts)

    def _parse_comparison(self, text: str) -> tuple[dict, str | None]:
        """Parse LLM output into comparison dict. Falls back on malformed output."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            return {
                "methods_comparison": data.get("methods_comparison", ""),
                "results_comparison": data.get("results_comparison", ""),
                "contributions_comparison": data.get("contributions_comparison", ""),
                "limitations_comparison": data.get("limitations_comparison", ""),
                "common_themes": data.get("common_themes", []),
                "key_differences": data.get("key_differences", []),
                "verdict": data.get("verdict", ""),
            }, None
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse comparison JSON, using fallback: %s", text[:200])
            return {
                "methods_comparison": "Unable to parse structured comparison.",
                "results_comparison": "See raw analysis.",
                "contributions_comparison": "See raw analysis.",
                "limitations_comparison": "See raw analysis.",
                "common_themes": ["See raw analysis"],
                "key_differences": [text.strip()[:500] if text.strip() else "Unable to extract differences"],
                "verdict": "LLM output could not be parsed into structured comparison.",
            }, "LLM returned malformed output — fallback comparison generated"
