"""
Ingestion API router — exposes paper fetch and index operations as REST endpoints.

Why it's needed:
    Airflow runs in a separate container with SQLAlchemy 1.4. Our src/ code
    uses SQLAlchemy 2.0 (DeclarativeBase, Mapped). These cannot coexist.
    The solution: Airflow calls these HTTP endpoints instead of importing src.*
    directly. The API container handles all business logic with the correct
    dependencies already installed.

What it does:
    POST /api/v1/ingest/fetch  — fetch yesterday's (or a given date's) papers
                                  from arXiv, parse PDFs, store in PostgreSQL
    POST /api/v1/ingest/index  — read stored papers from PostgreSQL, chunk +
                                  embed + bulk-index into OpenSearch

How Airflow uses it:
    Task 2 (fetch_daily_papers)  → POST /api/v1/ingest/fetch
    Task 3 (index_papers_hybrid) → POST /api/v1/ingest/index
    Both tasks are simple httpx.post() calls — zero src.* imports in Airflow.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException

from src.dependency import SessionDep, SettingsDep
from src.repositories.paper import PaperRepository
from src.schemas.api.ingest import (
    FetchRequest, FetchResponse,
    IndexRequest, IndexResponse,
)
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.metadata_fetcher import make_metadata_fetcher

logger = logging.getLogger(__name__)

ingest_router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


@ingest_router.post("/fetch", response_model=FetchResponse)
async def fetch_papers(
    request: FetchRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> FetchResponse:
    """
    Fetch arXiv papers for a given date, parse PDFs, store in PostgreSQL.

    Called by Airflow Task 2 (fetch_daily_papers). Defaults to yesterday
    when no date is provided so scheduled runs always process the previous day.
    """
    # Default to yesterday when no date given
    if request.date:
        target_date = request.date
    else:
        yesterday = datetime.utcnow() - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info(f"POST /ingest/fetch | date={target_date} | max_results={request.max_results}")

    try:
        metadata_fetcher = make_metadata_fetcher()
        result = await metadata_fetcher.fetch_and_process_papers(
            from_date=target_date,
            to_date=target_date,
            max_results=request.max_results,
            process_pdfs=request.process_pdfs,
            store_to_db=True,
            db_session=session,
        )

        return FetchResponse(
            target_date=target_date,
            papers_fetched=result.get("papers_fetched", 0),
            pdfs_downloaded=result.get("pdfs_downloaded", 0),
            pdfs_parsed=result.get("pdfs_parsed", 0),
            papers_stored=result.get("papers_stored", 0),
            arxiv_ids=result.get("papers", []),
            errors=result.get("errors", []),
            processing_time=result.get("processing_time", 0.0),
        )

    except Exception as e:
        logger.error(f"Fetch pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ingest_router.post("/index", response_model=IndexResponse)
async def index_papers(
    request: IndexRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> IndexResponse:
    """
    Chunk, embed, and index papers into OpenSearch.

    Called by Airflow Task 3 (index_papers_hybrid). Accepts specific
    arxiv_ids from Task 2's XCom result, or falls back to papers stored
    in the last since_hours.
    """
    logger.info(
        f"POST /ingest/index | arxiv_ids={request.arxiv_ids} | since_hours={request.since_hours}"
    )

    try:
        repo = PaperRepository(session)

        # Resolve which papers to index
        if request.arxiv_ids:
            papers = [repo.get_by_arxiv_id(aid) for aid in request.arxiv_ids]
            papers = [p for p in papers if p is not None]
        else:
            since = datetime.utcnow() - timedelta(hours=request.since_hours)
            papers = repo.get_recently_stored(since=since)

        if not papers:
            logger.warning("No papers found to index")
            return IndexResponse(
                papers_processed=0, chunks_created=0, chunks_indexed=0, errors=[]
            )

        # Convert ORM objects to plain dicts for the indexing service
        paper_dicts = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "categories": p.categories,
                "published_date": p.published_date.strftime("%Y-%m-%dT%H:%M:%SZ") if p.published_date else None,
                "pdf_url": p.pdf_url,
                "raw_text": p.pdf_content,
                "sections": p.sections or [],
            }
            for p in papers
        ]

        indexing_service = make_hybrid_indexing_service(settings)
        stats = await indexing_service.index_papers_batch(paper_dicts)

        return IndexResponse(
            papers_processed=stats.get("papers_processed", len(papers)),
            chunks_created=stats.get("total_chunks_created", 0),
            chunks_indexed=stats.get("total_chunks_indexed", 0),
            errors=stats.get("errors", []),
        )

    except Exception as e:
        logger.error(f"Index pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
