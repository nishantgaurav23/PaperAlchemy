"""Paper listing and retrieval endpoints (GET /papers, GET /papers/{id})."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from src.dependency import PaperRepoDep, SessionDep
from src.schemas.paper import PaperResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=list[PaperResponse])
async def list_papers(
    paper_repo: PaperRepoDep,
    session: SessionDep,
    query: str | None = Query(default=None, description="Search query (title/abstract)"),
    category: str | None = Query(default=None, description="Filter by arXiv category"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of papers"),
    offset: int = Query(default=0, ge=0, description="Number of papers to skip"),
) -> list[PaperResponse]:
    """List papers with optional filtering by query and category."""
    papers = await paper_repo.search(
        query=query,
        category=category,
        limit=limit,
        offset=offset,
    )
    return [PaperResponse.model_validate(p) for p in papers]


@router.get("/by-arxiv/{arxiv_id:path}", response_model=PaperResponse)
async def get_paper_by_arxiv(
    arxiv_id: str,
    paper_repo: PaperRepoDep,
    session: SessionDep,
) -> PaperResponse:
    """Get a single paper by its arXiv ID (e.g., 2301.12345 or upload_abc123)."""
    paper = await paper_repo.get_by_arxiv_id(arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper with arxiv_id={arxiv_id} not found")
    return PaperResponse.model_validate(paper)


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: uuid.UUID,
    paper_repo: PaperRepoDep,
    session: SessionDep,
) -> PaperResponse:
    """Get a single paper by its UUID."""
    paper = await paper_repo.get_by_id(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
    return PaperResponse.model_validate(paper)
