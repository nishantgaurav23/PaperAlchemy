"""Methodology & findings deep-dive analysis service: extract structured methodology from papers via LLM."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.schemas.api.analysis import MethodologyAnalysis, MethodologyResponse

if TYPE_CHECKING:
    from src.repositories.paper import PaperRepository
    from src.services.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

METHODOLOGY_PROMPT = """You are a research paper analyst specializing in methodology and experimental design.
Analyze the following academic paper content and extract a structured methodology & findings analysis.

Return ONLY valid JSON with this exact structure (no markdown fences, no explanation):
{{
    "research_design": "Type of study and overall approach (1-3 sentences)",
    "datasets": [
        {{"name": "Dataset Name", "description": "Brief description", "size": "e.g. 10k samples"}}
    ],
    "baselines": ["Baseline method 1", "Baseline method 2"],
    "key_results": [
        {{"metric": "Metric name", "value": "Value", "context": "On which benchmark/task"}}
    ],
    "statistical_significance": "Notes on statistical tests, p-values, confidence intervals (or null if not reported)",
    "reproducibility_notes": "Code availability, hyperparameters, compute resources (or null if not mentioned)"
}}

Rules:
- research_design: 1-3 sentences describing the study type (experimental, theoretical, survey, etc.)
- datasets: List of datasets used. Empty list [] for theoretical papers with no datasets.
- baselines: List of baseline methods compared against. Empty list [] if none.
- key_results: List of quantitative results with metrics. Empty list [] for theoretical papers.
- statistical_significance: null if not reported in the paper.
- reproducibility_notes: null if not mentioned.
- Be specific — reference actual methods, datasets, metrics, and values.

Paper content:
{content}
"""

MAX_CONTENT_WORDS = 4000


class MethodologyService:
    """Extracts structured methodology and findings analysis from academic papers using an LLM."""

    def __init__(self, llm_provider: LLMProvider, paper_repo: PaperRepository) -> None:
        self._llm = llm_provider
        self._repo = paper_repo

    async def analyze_methodology(
        self,
        paper_id: uuid.UUID,
        *,
        force: bool = False,
    ) -> MethodologyResponse:
        """Analyze the methodology and findings of a paper.

        Args:
            paper_id: UUID of the paper to analyze.
            force: If True, always regenerate (skip any cache).

        Returns:
            MethodologyResponse with structured analysis + metadata.

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
            warning = "Only abstract available — methodology analysis extracted from abstract alone"

        content = self._prepare_content(paper)
        prompt = METHODOLOGY_PROMPT.format(content=content)

        try:
            response = await self._llm.generate(prompt, temperature=0.3)
        except Exception as e:
            raise LLMServiceError(f"LLM failed to generate methodology analysis: {e}") from e

        analysis, parse_warning = self._parse_analysis(response.text)
        if parse_warning:
            warning = parse_warning

        return MethodologyResponse(
            paper_id=paper_id,
            analysis=analysis,
            model=response.model,
            provider=response.provider,
            latency_ms=response.usage.latency_ms if response.usage else None,
            warning=warning,
        )

    def _prepare_content(self, paper) -> str:
        """Format paper content for the LLM prompt, prioritizing methodology-relevant sections.

        Truncates to ~4000 words to stay within context limits.
        """
        parts: list[str] = []

        # Always include abstract first
        if paper.abstract and paper.abstract.strip():
            parts.append(f"## Abstract\n{paper.abstract.strip()}")

        # Add sections, prioritizing methodology/experiments/results
        priority_keywords = {"methodology", "method", "approach", "experiments", "experimental", "results", "evaluation", "setup"}
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

    def _parse_analysis(self, text: str) -> tuple[MethodologyAnalysis, str | None]:
        """Parse LLM output into MethodologyAnalysis. Falls back on malformed output."""
        from src.schemas.api.analysis import DatasetInfo, ResultEntry

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            # Parse nested objects
            datasets = [DatasetInfo(**d) if isinstance(d, dict) else d for d in data.get("datasets", [])]
            key_results = [ResultEntry(**r) if isinstance(r, dict) else r for r in data.get("key_results", [])]

            return MethodologyAnalysis(
                research_design=data.get("research_design", ""),
                datasets=datasets,
                baselines=data.get("baselines", []),
                key_results=key_results,
                statistical_significance=data.get("statistical_significance"),
                reproducibility_notes=data.get("reproducibility_notes"),
            ), None
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse methodology analysis JSON, using fallback: %s", text[:200])
            return MethodologyAnalysis(
                research_design=text.strip()[:500] if text.strip() else "Unable to extract research design",
                datasets=[],
                baselines=[],
                key_results=[],
                statistical_significance=None,
                reproducibility_notes=None,
            ), "LLM returned malformed output — fallback analysis generated"
