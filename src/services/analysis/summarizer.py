"""AI-generated paper summary service.

Generates structured summaries with: objective, method, key findings,
contribution, and limitations using the LLM provider.
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.repositories.paper import PaperRepository
from src.schemas.api.analysis import PaperSummary, SummaryResponse
from src.services.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

MAX_CONTENT_WORDS = 4000

SUMMARY_SYSTEM_PROMPT = """\
You are a scientific paper analysis assistant. Given the content of a research paper, \
produce a structured JSON summary with exactly these fields:

{
    "objective": "The main research question or objective (1-2 sentences)",
    "method": "The methodology or approach used (1-3 sentences)",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "contribution": "The main contribution to the field (1-2 sentences)",
    "limitations": "Key limitations or caveats (1-2 sentences)"
}

Rules:
- Return ONLY valid JSON, no markdown fences or extra text.
- key_findings must be a JSON array of 2-5 concise strings.
- Be specific — reference actual methods, datasets, and metrics when available.
- If information for a field is not available, provide your best inference and note it.
"""


class SummarizerService:
    """Generates structured AI summaries for academic papers."""

    def extract_content(self, paper) -> str:
        """Extract and format paper content for the LLM prompt.

        Prioritises abstract + key sections (Introduction, Methodology, Results, Conclusion).
        Truncates to ~MAX_CONTENT_WORDS words to fit context windows.
        """
        parts: list[str] = []

        # Title
        parts.append(f"Title: {paper.title}")

        # Authors
        if hasattr(paper, "authors") and paper.authors:
            authors_str = ", ".join(paper.authors) if isinstance(paper.authors, list) else str(paper.authors)
            parts.append(f"Authors: {authors_str}")

        # Abstract
        if paper.abstract:
            parts.append(f"\nAbstract:\n{paper.abstract}")

        # Sections (if available)
        if paper.sections:
            priority_keywords = [
                "introduction",
                "methodology",
                "method",
                "approach",
                "results",
                "experiment",
                "conclusion",
                "discussion",
            ]
            added_sections: list[str] = []

            # Add priority sections first
            for section in paper.sections:
                section_title = section.get("title", "") if isinstance(section, dict) else getattr(section, "title", "")
                section_content = section.get("content", "") if isinstance(section, dict) else getattr(section, "content", "")
                if not section_content:
                    continue

                title_lower = section_title.lower()
                if any(kw in title_lower for kw in priority_keywords):
                    added_sections.append(f"\n## {section_title}\n{section_content}")

            # Add remaining sections if space permits
            for section in paper.sections:
                section_title = section.get("title", "") if isinstance(section, dict) else getattr(section, "title", "")
                section_content = section.get("content", "") if isinstance(section, dict) else getattr(section, "content", "")
                if not section_content:
                    continue

                title_lower = section_title.lower()
                if not any(kw in title_lower for kw in priority_keywords):
                    added_sections.append(f"\n## {section_title}\n{section_content}")

            parts.extend(added_sections)

        # Truncate to MAX_CONTENT_WORDS
        full_text = "\n".join(parts)
        words = full_text.split()
        if len(words) > MAX_CONTENT_WORDS:
            full_text = " ".join(words[:MAX_CONTENT_WORDS])

        return full_text

    async def summarize(
        self,
        *,
        paper_id: uuid.UUID,
        paper_repo: PaperRepository,
        session: AsyncSession,
        llm_provider: LLMProvider,
        force: bool = False,
    ) -> SummaryResponse:
        """Generate a structured summary for a paper.

        Args:
            paper_id: UUID of the paper to summarize.
            paper_repo: Repository for paper lookups.
            session: Database session (unused directly, but needed for repo).
            llm_provider: LLM provider for text generation.
            force: If True, always generate fresh (no caching).

        Returns:
            SummaryResponse with structured summary and metadata.

        Raises:
            PaperNotFoundError: If paper_id doesn't exist.
            InsufficientContentError: If paper has no abstract and no sections.
            LLMServiceError: If LLM generation fails.
        """
        # 1. Fetch paper
        paper = await paper_repo.get_by_id(paper_id)
        if paper is None:
            raise PaperNotFoundError(f"Paper {paper_id} not found")

        # 2. Check content sufficiency
        has_abstract = bool(paper.abstract and paper.abstract.strip())
        has_sections = bool(paper.sections)
        has_pdf_content = bool(paper.pdf_content and paper.pdf_content.strip())

        if not has_abstract and not has_sections and not has_pdf_content:
            raise InsufficientContentError(f"Paper {paper_id} has no abstract, sections, or parsed content")

        warnings: list[str] = []
        if not has_sections and not has_pdf_content:
            warnings.append("Generated from abstract only — summary may lack detail")

        # 3. Extract content
        content = self.extract_content(paper)

        # 4. Build prompt
        prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n---\n\n{content}\n\n---\n\nProvide the JSON summary:"

        # 5. Call LLM
        try:
            llm_response = await llm_provider.generate(
                prompt,
                temperature=0.3,
                max_tokens=1000,
            )
        except LLMServiceError:
            raise
        except Exception as e:
            raise LLMServiceError(f"Summary generation failed: {e}") from e

        # 6. Parse structured output
        summary = self._parse_summary(llm_response.text, warnings)

        # 7. Build response
        latency_ms = None
        if llm_response.usage and hasattr(llm_response.usage, "latency_ms"):
            latency_ms = llm_response.usage.latency_ms

        return SummaryResponse(
            paper_id=paper_id,
            title=paper.title,
            summary=summary,
            model=llm_response.model,
            provider=llm_response.provider,
            latency_ms=latency_ms,
            warnings=warnings,
        )

    def _parse_summary(self, llm_text: str, warnings: list[str]) -> PaperSummary:
        """Parse LLM output into structured PaperSummary.

        Attempts JSON parsing first, falls back to raw text extraction.
        """
        # Strip markdown code fences if present
        text = llm_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            return PaperSummary(
                objective=data.get("objective", ""),
                method=data.get("method", ""),
                key_findings=data.get("key_findings", []),
                contribution=data.get("contribution", ""),
                limitations=data.get("limitations", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse structured summary, using raw text: %s", e)
            warnings.append("LLM output was not valid JSON — summary extracted from raw text")
            return PaperSummary(
                objective=text[:500] if text else "Unable to extract objective",
                method="See full text",
                key_findings=["See full text for details"],
                contribution="See full text",
                limitations="Unable to parse structured output",
            )
