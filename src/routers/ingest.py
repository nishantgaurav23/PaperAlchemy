"""Ingestion endpoint: fetch arXiv papers, download PDFs, parse, chunk, embed, and index."""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter

from src.dependency import EmbeddingsDep, LLMProviderDep, OpenSearchDep, PaperRepoDep, SessionDep
from src.schemas.api.ingest import IngestRequest, IngestResponse, ReparseRequest, ReparseResponse
from src.schemas.paper import PaperCreate
from src.services.analysis import run_ai_analysis
from src.services.arxiv.factory import make_arxiv_client
from src.services.indexing.text_chunker import TextChunker
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_index_docs(
    chunks: list,
    embeddings: list[list[float]],
    arxiv_id: str,
    paper_id: str,
    title: str,
    abstract: str,
    authors: list[str],
) -> list[dict]:
    """Build OpenSearch index documents from chunks + embeddings."""
    index_docs = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_data = {
            "chunk_id": f"{arxiv_id}_chunk_{chunk.metadata.chunk_index}",
            "arxiv_id": arxiv_id,
            "paper_id": paper_id,
            "chunk_index": chunk.metadata.chunk_index,
            "chunk_text": chunk.text,
            "chunk_word_count": chunk.metadata.word_count,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "title": title,
            "abstract": abstract,
            "authors": authors,
        }
        index_docs.append({"chunk_data": chunk_data, "embedding": embedding})
    return index_docs


async def _chunk_embed_index(
    *,
    text_chunker: TextChunker,
    embeddings_client,
    opensearch_client,
    arxiv_id: str,
    paper_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    raw_text: str,
    sections: list | None,
) -> int:
    """Chunk, embed, and index a parsed paper into OpenSearch.

    Returns the number of chunks successfully indexed.
    """
    chunks = text_chunker.chunk_paper(
        title=title,
        abstract=abstract,
        full_text=raw_text,
        arxiv_id=arxiv_id,
        paper_id=paper_id,
        sections=sections,
    )

    if not chunks:
        logger.warning("No chunks generated for %s", arxiv_id)
        return 0

    chunk_texts = [c.text for c in chunks]
    embeddings = await embeddings_client.embed_passages(chunk_texts)

    index_docs = _build_index_docs(
        chunks=chunks,
        embeddings=embeddings,
        arxiv_id=arxiv_id,
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
    )

    result = opensearch_client.bulk_index_chunks(index_docs)
    indexed = result.get("success", 0)
    logger.info("Indexed %d chunks for %s", indexed, arxiv_id)
    return indexed


@router.post("/ingest/fetch", response_model=IngestResponse)
async def ingest_fetch(
    request: IngestRequest,
    repo: PaperRepoDep,
    session: SessionDep,
    embeddings_client: EmbeddingsDep,
    opensearch_client: OpenSearchDep,
    llm_provider: LLMProviderDep,
) -> IngestResponse:
    """Fetch arXiv papers for a target date, download PDFs, parse, and store.

    Full pipeline: fetch → download → parse → store → chunk → embed → index.
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
    text_chunker = TextChunker()

    # Ensure OpenSearch index exists
    try:
        opensearch_client.setup_indices()
    except Exception as e:
        logger.warning("OpenSearch index setup failed (may already exist): %s", e)

    # Step 1: Fetch papers from arXiv (date-filtered)
    papers = await arxiv_client.fetch_papers(
        from_date=target_date, to_date=target_date, max_results=request.max_results
    )

    # Fallback: if date filter returns nothing, fetch latest papers without date
    if not papers:
        logger.info("Date filter returned 0 papers, fetching latest %d papers instead", request.max_results)
        papers = await arxiv_client.fetch_papers(max_results=request.max_results)

    arxiv_ids: list[str] = []
    errors: list[str] = []
    pdfs_downloaded = 0
    pdfs_parsed = 0
    papers_stored = 0
    total_chunks_indexed = 0

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
            db_paper = await repo.upsert(paper_create)
            papers_stored += 1
        except Exception as e:
            logger.error("Failed to store paper %s: %s", paper.arxiv_id, e)
            errors.append(f"Store failed for {paper.arxiv_id}: {e}")
            continue

        # Step 3: Download PDF
        pdf_path = await arxiv_client.download_pdf(paper.arxiv_id, paper.pdf_url)
        if pdf_path is None:
            logger.warning("PDF download failed for %s", paper.arxiv_id)
            await repo.update_parsing_status(
                arxiv_id=paper.arxiv_id,
                status="failed",
                error=f"PDF download failed for {paper.pdf_url}",
            )
            errors.append(f"Download failed for {paper.arxiv_id}")
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
            continue

        # Step 5: Chunk + Embed + Index (graceful degradation)
        try:
            chunks_indexed = await _chunk_embed_index(
                text_chunker=text_chunker,
                embeddings_client=embeddings_client,
                opensearch_client=opensearch_client,
                arxiv_id=paper.arxiv_id,
                paper_id=str(db_paper.id),
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                raw_text=content.raw_text,
                sections=content.sections,
            )
            total_chunks_indexed += chunks_indexed
        except Exception as e:
            logger.warning("Indexing failed for %s: %s", paper.arxiv_id, e)
            errors.append(f"Indexing failed for {paper.arxiv_id}: {e}")
            # Paper is still saved in DB — indexing failure is non-fatal

        # Step 6: AI Analysis (summary, highlights, methodology) — graceful degradation
        try:
            analysis_results = await run_ai_analysis(
                paper_id=db_paper.id,
                paper_repo=repo,
                session=session,
                llm_provider=llm_provider,
            )
            for w in analysis_results.get("warnings", []):
                errors.append(f"Analysis warning for {paper.arxiv_id}: {w}")
        except Exception as e:
            logger.warning("AI analysis failed for %s: %s", paper.arxiv_id, e)
            errors.append(f"AI analysis failed for {paper.arxiv_id}: {e}")

    await session.commit()

    processing_time = round(time.time() - start_time, 2)
    logger.info(
        "Ingestion complete: %d fetched, %d downloaded, %d parsed, %d stored, %d chunks indexed, %d errors in %.1fs",
        len(papers),
        pdfs_downloaded,
        pdfs_parsed,
        papers_stored,
        total_chunks_indexed,
        len(errors),
        processing_time,
    )

    return IngestResponse(
        papers_fetched=len(papers),
        pdfs_downloaded=pdfs_downloaded,
        pdfs_parsed=pdfs_parsed,
        papers_stored=papers_stored,
        chunks_indexed=total_chunks_indexed,
        arxiv_ids=arxiv_ids,
        errors=errors,
        processing_time=processing_time,
    )


@router.post("/ingest/reparse", response_model=ReparseResponse)
async def ingest_reparse(
    request: ReparseRequest,
    repo: PaperRepoDep,
    session: SessionDep,
    embeddings_client: EmbeddingsDep,
    opensearch_client: OpenSearchDep,
    llm_provider: LLMProviderDep,
) -> ReparseResponse:
    """Re-download and re-parse papers that are pending or failed.

    Also re-indexes chunks into OpenSearch.
    Useful for retrying papers that failed during initial ingestion
    due to rate limits, timeouts, or transient errors.
    """
    start_time = time.time()

    if request.status_filter == "all":
        pending = await repo.get_pending_parsing(limit=request.limit)
        failed = await repo.search(parsing_status="failed", limit=request.limit)
        papers = pending + failed
    elif request.status_filter == "failed":
        papers = await repo.search(parsing_status="failed", limit=request.limit)
    else:
        papers = await repo.get_pending_parsing(limit=request.limit)

    logger.info("Re-parse: found %d papers with status_filter='%s'", len(papers), request.status_filter)

    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    text_chunker = TextChunker()

    errors: list[str] = []
    pdfs_downloaded = 0
    pdfs_parsed = 0
    total_chunks_indexed = 0

    for paper in papers:
        # Download PDF (will use cache if already downloaded)
        pdf_path = await arxiv_client.download_pdf(paper.arxiv_id, paper.pdf_url)
        if pdf_path is None:
            await repo.update_parsing_status(
                arxiv_id=paper.arxiv_id,
                status="failed",
                error=f"PDF download failed for {paper.pdf_url}",
            )
            errors.append(f"Download failed for {paper.arxiv_id}")
            continue
        pdfs_downloaded += 1

        # Parse PDF
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
            logger.error("Re-parse failed for %s: %s", paper.arxiv_id, e)
            await repo.update_parsing_status(
                arxiv_id=paper.arxiv_id,
                status="failed",
                error=str(e),
            )
            errors.append(f"Parse failed for {paper.arxiv_id}: {e}")
            continue

        # Chunk + Embed + Index
        try:
            # Delete old chunks before re-indexing
            with contextlib.suppress(Exception):
                opensearch_client.delete_paper_chunks(paper.arxiv_id)

            chunks_indexed = await _chunk_embed_index(
                text_chunker=text_chunker,
                embeddings_client=embeddings_client,
                opensearch_client=opensearch_client,
                arxiv_id=paper.arxiv_id,
                paper_id=str(paper.id),
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                raw_text=content.raw_text,
                sections=content.sections,
            )
            total_chunks_indexed += chunks_indexed
        except Exception as e:
            logger.warning("Indexing failed for %s: %s", paper.arxiv_id, e)
            errors.append(f"Indexing failed for {paper.arxiv_id}: {e}")

        # AI Analysis (summary, highlights, methodology) — graceful degradation
        try:
            analysis_results = await run_ai_analysis(
                paper_id=paper.id,
                paper_repo=repo,
                session=session,
                llm_provider=llm_provider,
            )
            for w in analysis_results.get("warnings", []):
                errors.append(f"Analysis warning for {paper.arxiv_id}: {w}")
        except Exception as e:
            logger.warning("AI analysis failed for %s: %s", paper.arxiv_id, e)
            errors.append(f"AI analysis failed for {paper.arxiv_id}: {e}")

    await session.commit()

    processing_time = round(time.time() - start_time, 2)
    logger.info(
        "Re-parse complete: %d found, %d downloaded, %d parsed, %d chunks indexed, %d errors in %.1fs",
        len(papers),
        pdfs_downloaded,
        pdfs_parsed,
        total_chunks_indexed,
        len(errors),
        processing_time,
    )

    return ReparseResponse(
        total_found=len(papers),
        pdfs_downloaded=pdfs_downloaded,
        pdfs_parsed=pdfs_parsed,
        chunks_indexed=total_chunks_indexed,
        errors=errors,
        processing_time=processing_time,
    )
