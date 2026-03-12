"""PDF upload endpoint: accept, validate, parse, store, and index uploaded papers."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from src.dependency import EmbeddingsDep, OpenSearchDep, PaperRepoDep, SessionDep
from src.exceptions import PDFParsingError, PDFValidationError
from src.schemas.api.upload import UploadResponse
from src.services.indexing.text_chunker import TextChunker
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.pdf_parser.service import PDFParserService
from src.services.upload.service import UploadService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_pdf_parser() -> PDFParserService:
    """Return a singleton PDFParserService."""
    return make_pdf_parser_service()


def get_text_chunker() -> TextChunker:
    """Return a TextChunker with default settings."""
    return TextChunker()


def get_upload_service() -> UploadService:
    """Return an UploadService with default settings."""
    return UploadService(max_file_size_mb=50)


PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
TextChunkerDep = Annotated[TextChunker, Depends(get_text_chunker)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]


# ---------------------------------------------------------------------------
# POST /api/v1/upload
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile,
    upload_service: UploadServiceDep,
    pdf_parser: PDFParserDep,
    paper_repo: PaperRepoDep,
    session: SessionDep,
    text_chunker: TextChunkerDep,
    embeddings: EmbeddingsDep,
    opensearch: OpenSearchDep,
) -> UploadResponse:
    """Upload a PDF paper, parse it, store metadata, and index for retrieval."""
    try:
        return await upload_service.process_upload(
            file=file,
            pdf_parser=pdf_parser,
            paper_repo=paper_repo,
            session=session,
            text_chunker=text_chunker,
            embeddings_client=embeddings,
            opensearch_client=opensearch,
        )
    except PDFValidationError as e:
        status = e.status_code if e.status_code != 500 else 422
        raise HTTPException(status_code=status, detail=e.detail) from e
    except PDFParsingError as e:
        raise HTTPException(status_code=422, detail=e.detail) from e
    except Exception as e:
        logger.exception("Unexpected error during upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e
