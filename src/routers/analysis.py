"""Paper analysis endpoints: summary, highlights, methodology, comparison."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from src.dependency import LLMProviderDep, PaperRepoDep, SessionDep
from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.schemas.api.analysis import (
    ComparisonRequest,
    ComparisonResponse,
    HighlightsResponse,
    MethodologyResponse,
    SummaryResponse,
)
from src.services.analysis.comparator import ComparatorService
from src.services.analysis.highlights import HighlightsService
from src.services.analysis.methodology import MethodologyService
from src.services.analysis.summarizer import SummarizerService

logger = logging.getLogger(__name__)

router = APIRouter()

_summarizer = SummarizerService()


# Static route must come BEFORE {paper_id} parametric routes
@router.post("/papers/compare", response_model=ComparisonResponse)
async def compare_papers(
    body: ComparisonRequest,
    paper_repo: PaperRepoDep,
    llm_provider: LLMProviderDep,
    force: bool = Query(default=False, description="Force regeneration, bypassing any cache"),
) -> ComparisonResponse:
    """Compare 2-5 papers side-by-side."""
    try:
        service = ComparatorService(llm_provider=llm_provider, paper_repo=paper_repo)
        return await service.compare(body.paper_ids, force=force)
    except PaperNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail) from e
    except InsufficientContentError as e:
        raise HTTPException(status_code=422, detail=e.detail) from e
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error comparing papers: %s", e)
        raise HTTPException(status_code=500, detail=f"Paper comparison failed: {e}") from e


@router.post("/papers/{paper_id}/summary", response_model=SummaryResponse)
async def generate_summary(
    paper_id: uuid.UUID,
    paper_repo: PaperRepoDep,
    session: SessionDep,
    llm_provider: LLMProviderDep,
    force: bool = Query(default=False, description="Force regeneration, bypassing any cache"),
) -> SummaryResponse:
    """Generate an AI-powered structured summary for a paper."""
    try:
        result = await _summarizer.summarize(
            paper_id=paper_id,
            paper_repo=paper_repo,
            session=session,
            llm_provider=llm_provider,
            force=force,
        )
        # Persist to DB
        await paper_repo.update_analysis(paper_id, summary=result.summary.model_dump())
        await session.commit()
        return result
    except PaperNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail) from e
    except InsufficientContentError as e:
        raise HTTPException(status_code=422, detail=e.detail) from e
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except Exception as e:
        logger.exception("Unexpected error generating summary: %s", e)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}") from e


@router.post("/papers/{paper_id}/highlights", response_model=HighlightsResponse)
async def extract_highlights(
    paper_id: uuid.UUID,
    paper_repo: PaperRepoDep,
    session: SessionDep,
    llm_provider: LLMProviderDep,
    force: bool = Query(default=False, description="Force regeneration, bypassing any cache"),
) -> HighlightsResponse:
    """Extract key highlights and insights from a paper."""
    try:
        service = HighlightsService(llm_provider=llm_provider, paper_repo=paper_repo)
        result = await service.extract_highlights(paper_id, force=force)
        # Persist to DB
        await paper_repo.update_analysis(paper_id, highlights=result.highlights.model_dump())
        await session.commit()
        return result
    except PaperNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail) from e
    except InsufficientContentError as e:
        raise HTTPException(status_code=422, detail=e.detail) from e
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except Exception as e:
        logger.exception("Unexpected error extracting highlights: %s", e)
        raise HTTPException(status_code=500, detail=f"Highlights extraction failed: {e}") from e


@router.post("/papers/{paper_id}/methodology", response_model=MethodologyResponse)
async def analyze_methodology(
    paper_id: uuid.UUID,
    paper_repo: PaperRepoDep,
    session: SessionDep,
    llm_provider: LLMProviderDep,
    force: bool = Query(default=False, description="Force regeneration, bypassing any cache"),
) -> MethodologyResponse:
    """Analyze the methodology and findings of a paper."""
    try:
        service = MethodologyService(llm_provider=llm_provider, paper_repo=paper_repo)
        result = await service.analyze_methodology(paper_id, force=force)
        # Persist to DB
        await paper_repo.update_analysis(paper_id, methodology=result.analysis.model_dump())
        await session.commit()
        return result
    except PaperNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail) from e
    except InsufficientContentError as e:
        raise HTTPException(status_code=422, detail=e.detail) from e
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except Exception as e:
        logger.exception("Unexpected error analyzing methodology: %s", e)
        raise HTTPException(status_code=500, detail=f"Methodology analysis failed: {e}") from e
