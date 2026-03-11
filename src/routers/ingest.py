"""Ingestion endpoint: fetch arXiv papers, download PDFs, parse, and store."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter

from src.dependency import PaperRepoDep, SessionDep
from src.schemas.api.ingest import IngestRequest, IngestResponse
from src.schemas.paper import PaperCreate
from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest/fetch", response_model=IngestResponse)
async def ingest_fetch(
    request: IngestRequest,
    repo: PaperRepoDep,
    session: SessionDep,
) -> IngestResponse:
    """Fetch arXiv papers for a target date, download PDFs, parse, and store.

    Idempotent: uses upsert for paper metadata and overwrites parsed content.
    """
    start_time = time.time()

    # Resolve target date
    if request.target_date:
        target_date = request.target_date
    else:
        yesterday = datetime.now(tz=UTC) - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info("Ingestion started for target_date=%s", target_date)

    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()

    # Step 1: Fetch papers from arXiv
    papers = await arxiv_client.fetch_papers(from_date=target_date, to_date=target_date)

    arxiv_ids: list[str] = []
    errors: list[str] = []
    pdfs_downloaded = 0
    pdfs_parsed = 0
    papers_stored = 0

    for paper in papers:
        arxiv_ids.append(paper.arxiv_id)

        # Step 2: Upsert paper metadata to DB
        try:
            paper_create = PaperCreate(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                categories=paper.categories,
                published_date=datetime.fromisoformat(paper.published_date.replace("Z", "+00:00")),
                updated_date=(datetime.fromisoformat(paper.updated_date.replace("Z", "+00:00")) if paper.updated_date else None),
                pdf_url=paper.pdf_url,
                parsing_status="pending",
            )
            await repo.upsert(paper_create)
            papers_stored += 1
        except Exception as e:
            logger.error("Failed to store paper %s: %s", paper.arxiv_id, e)
            errors.append(f"Store failed for {paper.arxiv_id}: {e}")
            continue

        # Step 3: Download PDF
        pdf_path = await arxiv_client.download_pdf(paper.arxiv_id, paper.pdf_url)
        if pdf_path is None:
            logger.warning("PDF download failed for %s", paper.arxiv_id)
            continue
        pdfs_downloaded += 1

        # Step 4: Parse PDF
        try:
            content = await pdf_parser.parse_pdf(pdf_path)
            sections_dicts = [s.model_dump() for s in content.sections]
            await repo.update_parsing_status(
                arxiv_id=paper.arxiv_id,
                status="success",
                pdf_content=content.raw_text,
                sections=sections_dicts,
            )
            pdfs_parsed += 1
        except Exception as e:
            logger.error("PDF parse failed for %s: %s", paper.arxiv_id, e)
            await repo.update_parsing_status(
                arxiv_id=paper.arxiv_id,
                status="failed",
                error=str(e),
            )
            errors.append(f"Parse failed for {paper.arxiv_id}: {e}")

    await session.commit()

    processing_time = round(time.time() - start_time, 2)
    logger.info(
        "Ingestion complete: %d fetched, %d downloaded, %d parsed, %d stored, %d errors in %.1fs",
        len(papers),
        pdfs_downloaded,
        pdfs_parsed,
        papers_stored,
        len(errors),
        processing_time,
    )

    return IngestResponse(
        papers_fetched=len(papers),
        pdfs_downloaded=pdfs_downloaded,
        pdfs_parsed=pdfs_parsed,
        papers_stored=papers_stored,
        arxiv_ids=arxiv_ids,
        errors=errors,
        processing_time=processing_time,
    )
