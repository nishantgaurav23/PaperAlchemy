"""Key highlights extraction service: extract structured insights from papers via LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.schemas.api.analysis import HighlightsResponse, PaperHighlights

if TYPE_CHECKING:
    from src.repositories.paper import PaperRepository
    from src.services.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

HIGHLIGHTS_PROMPT = """You are a research paper analyst. Extract key highlights from the following academic paper content.

Return ONLY valid JSON with this exact structure (no markdown fences, no explanation):
{{
    "novel_contributions": ["contribution 1", "contribution 2"],
    "important_findings": ["finding 1", "finding 2"],
    "practical_implications": ["implication 1", "implication 2"],
    "limitations": ["limitation 1"],
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Rules:
- novel_contributions: 1-5 items — what is genuinely new in this paper
- important_findings: 1-5 items — key results, discoveries, empirical evidence
- practical_implications: 1-5 items — real-world applications, impact, downstream uses
- limitations: 1-3 items — noted limitations, caveats, open questions
- keywords: 3-10 items — key terms, topics, methodologies

Paper content:
{content}
"""

MAX_CONTENT_WORDS = 4000


class HighlightsService:
    """Extracts structured key highlights from academic papers using an LLM."""

    def __init__(self, llm_provider: LLMProvider, paper_repo: PaperRepository) -> None:
        self._llm = llm_provider
        self._repo = paper_repo

    async def extract_highlights(
        self,
        paper_id: uuid.UUID,
        *,
        force: bool = False,
    ) -> HighlightsResponse:
        """Extract key highlights from a paper.

        Args:
            paper_id: UUID of the paper to analyze.
            force: If True, always regenerate (skip any cache).

        Returns:
            HighlightsResponse with structured highlights + metadata.

        Raises:
            PaperNotFoundError: Paper not in database.
            InsufficientContentError: Paper has no abstract and no sections.
            LLMServiceError: LLM provider failure.
        """
        paper = await self._repo.get_by_id(paper_id)
        if paper is None:
            raise PaperNotFoundError(f"Paper {paper_id} not found")

        # Check for sufficient content
        has_abstract = bool(paper.abstract and paper.abstract.strip())
        has_sections = bool(paper.sections)
        has_pdf = bool(paper.pdf_content and paper.pdf_content.strip())

        if not has_abstract and not has_sections and not has_pdf:
            raise InsufficientContentError(f"Paper {paper_id} has no abstract, sections, or PDF content")

        warning: str | None = None
        if not has_sections and not has_pdf:
            warning = "Only abstract available — highlights extracted from abstract alone"

        content = self._prepare_content(paper)
        prompt = HIGHLIGHTS_PROMPT.format(content=content)

        try:
            response = await self._llm.generate(prompt, temperature=0.3)
        except Exception as e:
            raise LLMServiceError(f"LLM failed to generate highlights: {e}") from e

        highlights, parse_warning = self._parse_highlights(response.text)
        if parse_warning:
            warning = parse_warning

        return HighlightsResponse(
            paper_id=paper_id,
            highlights=highlights,
            model=response.model,
            provider=response.provider,
            latency_ms=response.usage.latency_ms if response.usage else None,
            warning=warning,
        )

    def _prepare_content(self, paper) -> str:
        """Format paper content for the LLM prompt, prioritizing key sections.

        Truncates to ~4000 words to stay within context limits.
        """
        parts: list[str] = []

        # Always include abstract first
        if paper.abstract and paper.abstract.strip():
            parts.append(f"## Abstract\n{paper.abstract.strip()}")

        # Add sections, prioritizing results/conclusion/discussion
        priority_keywords = {"results", "conclusion", "discussion", "findings", "experiments"}
        other_sections: list[str] = []
        priority_sections: list[str] = []

        if paper.sections:
            for section in paper.sections:
                title = section.get("title", "") if isinstance(section, dict) else getattr(section, "title", "")
                content = section.get("content", "") if isinstance(section, dict) else getattr(section, "content", "")
                if not content or not content.strip():
                    continue
                if title.lower().strip() == "abstract":
                    continue  # Already added above
                formatted = f"## {title}\n{content.strip()}"
                if any(kw in title.lower() for kw in priority_keywords):
                    priority_sections.append(formatted)
                else:
                    other_sections.append(formatted)

        # Priority sections first, then others
        parts.extend(priority_sections)
        parts.extend(other_sections)

        # If no sections but we have pdf_content, use it
        if not paper.sections and paper.pdf_content and paper.pdf_content.strip():
            parts.append(f"## Full Text\n{paper.pdf_content.strip()}")

        combined = "\n\n".join(parts)

        # Truncate to MAX_CONTENT_WORDS
        words = combined.split()
        if len(words) > MAX_CONTENT_WORDS:
            combined = " ".join(words[:MAX_CONTENT_WORDS])

        return combined

    def _parse_highlights(self, text: str) -> tuple[PaperHighlights, str | None]:
        """Parse LLM output into PaperHighlights. Falls back on malformed output."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            return PaperHighlights(**data), None
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse highlights JSON, using fallback: %s", text[:200])
            # Fallback: treat entire text as a single finding
            return PaperHighlights(
                novel_contributions=["See raw analysis below"],
                important_findings=[text.strip()[:500] if text.strip() else "Unable to extract findings"],
                practical_implications=["Further analysis needed"],
                limitations=[],
                keywords=["analysis", "paper"],
            ), "LLM returned malformed output — fallback highlights generated"
