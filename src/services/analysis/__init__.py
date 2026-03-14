"""Paper analysis services: summary, highlights, methodology, comparison."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from src.repositories.paper import PaperRepository

logger = logging.getLogger(__name__)


async def run_ai_analysis(
    paper_id: uuid.UUID,
    paper_repo: PaperRepository,
    session: AsyncSession,
    llm_provider: Any,
) -> dict[str, Any]:
    """Run AI analysis (summary, highlights, methodology) and store results in DB.

    Returns dict with analysis results and any warnings. Failures are non-fatal.
    """
    from src.services.analysis.highlights import HighlightsService
    from src.services.analysis.methodology import MethodologyService
    from src.services.analysis.summarizer import SummarizerService

    results: dict[str, Any] = {"summary": None, "highlights": None, "methodology": None, "warnings": []}

    # Summary
    try:
        summarizer = SummarizerService()
        summary_resp = await summarizer.summarize(
            paper_id=paper_id,
            paper_repo=paper_repo,
            session=session,
            llm_provider=llm_provider,
        )
        results["summary"] = summary_resp.summary.model_dump()
    except Exception as e:
        logger.warning("Summary generation failed for %s: %s", paper_id, e)
        results["warnings"].append(f"Summary generation failed: {e}")

    # Highlights
    try:
        highlights_svc = HighlightsService(llm_provider=llm_provider, paper_repo=paper_repo)
        highlights_resp = await highlights_svc.extract_highlights(paper_id)
        results["highlights"] = highlights_resp.highlights.model_dump()
    except Exception as e:
        logger.warning("Highlights extraction failed for %s: %s", paper_id, e)
        results["warnings"].append(f"Highlights extraction failed: {e}")

    # Methodology
    try:
        methodology_svc = MethodologyService(llm_provider=llm_provider, paper_repo=paper_repo)
        methodology_resp = await methodology_svc.analyze_methodology(paper_id)
        results["methodology"] = methodology_resp.analysis.model_dump()
    except Exception as e:
        logger.warning("Methodology analysis failed for %s: %s", paper_id, e)
        results["warnings"].append(f"Methodology analysis failed: {e}")

    # Persist analysis results to DB
    paper = await paper_repo.get_by_id(paper_id)
    if paper:
        if results["summary"]:
            paper.summary = results["summary"]
        if results["highlights"]:
            paper.highlights = results["highlights"]
        if results["methodology"]:
            paper.methodology = results["methodology"]
        await session.flush()

    return results
