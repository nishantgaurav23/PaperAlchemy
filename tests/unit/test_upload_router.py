"""Tests for the PDF upload router (POST /api/v1/upload).

Tests the FastAPI endpoint that orchestrates PDF upload → parse → store → index.
All external services are mocked via dependency_overrides.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.schemas.api.upload import UploadResponse
from src.schemas.pdf import PDFContent, Section

VALID_PDF_BYTES = b"%PDF-1.4 fake content for testing" + b"\x00" * 1000


def _make_pdf_content() -> PDFContent:
    return PDFContent(
        raw_text="This is the full text of the paper about transformers.",
        sections=[
            Section(title="Abstract", content="We present a novel approach.", level=1),
            Section(title="Introduction", content="Transformers have revolutionized NLP.", level=1),
        ],
        tables=[],
        figures=[],
        page_count=10,
        parser_used="docling",
        parser_time_seconds=2.0,
    )


def _make_paper_orm():
    paper = MagicMock()
    paper.id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    paper.arxiv_id = "upload_test123"
    paper.title = "Test Paper"
    paper.authors = []
    paper.abstract = "Test abstract"
    paper.parsing_status = "success"
    return paper


def _make_chunks():
    from src.schemas.indexing import ChunkMetadata, TextChunk

    return [
        TextChunk(
            text="Chunk 1",
            metadata=ChunkMetadata(
                chunk_index=0, start_char=0, end_char=50, word_count=10, overlap_with_previous=0, overlap_with_next=5
            ),
            arxiv_id="upload_test123",
            paper_id="12345678-1234-5678-1234-567812345678",
        ),
    ]


@pytest.fixture
def mock_pdf_parser():
    parser = AsyncMock()
    parser.parse_pdf = AsyncMock(return_value=_make_pdf_content())
    return parser


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=_make_paper_orm())
    return repo


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_text_chunker():
    chunker = MagicMock()
    chunker.chunk_paper = MagicMock(return_value=_make_chunks())
    return chunker


@pytest.fixture
def mock_embeddings_client():
    client = AsyncMock()
    client.embed_passages = AsyncMock(return_value=[[0.1] * 1024])
    return client


@pytest.fixture
def mock_opensearch_client():
    client = MagicMock()
    client.bulk_index_chunks = MagicMock(return_value={"success": 1, "failed": 0})
    return client


@pytest.fixture
def app_with_mocks(
    mock_pdf_parser,
    mock_paper_repo,
    mock_session,
    mock_text_chunker,
    mock_embeddings_client,
    mock_opensearch_client,
):
    """Create a minimal FastAPI app with the upload router and mocked deps."""
    from src.dependency import (
        get_db_session,
        get_embeddings_client,
        get_opensearch_client,
        get_paper_repository,
    )
    from src.routers.upload import router
    from src.services.upload.service import UploadService

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    app.dependency_overrides[get_db_session] = lambda: mock_session
    app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
    app.dependency_overrides[get_opensearch_client] = lambda: mock_opensearch_client
    app.dependency_overrides[get_embeddings_client] = lambda: mock_embeddings_client

    # Override the upload service dependencies
    from src.routers.upload import get_pdf_parser, get_text_chunker, get_upload_service

    app.dependency_overrides[get_pdf_parser] = lambda: mock_pdf_parser
    app.dependency_overrides[get_text_chunker] = lambda: mock_text_chunker
    app.dependency_overrides[get_upload_service] = lambda: UploadService(max_file_size_mb=50)

    return app


@pytest.mark.asyncio
class TestUploadEndpoint:
    """Test POST /api/v1/upload endpoint."""

    async def test_upload_valid_pdf(self, app_with_mocks):
        """Upload valid PDF → 200 with paper_id and chunks_indexed."""
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("test_paper.pdf", BytesIO(VALID_PDF_BYTES), "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "paper_id" in data
        assert data["chunks_indexed"] >= 1
        assert data["parsing_status"] == "success"
        assert data["indexing_status"] == "success"

    async def test_upload_non_pdf_rejected(self, app_with_mocks):
        """Upload .txt file → 422 with error message."""
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("paper.txt", BytesIO(b"some text"), "text/plain")},
            )

        assert response.status_code == 422
        data = response.json()
        assert "pdf" in data["detail"].lower() or "PDF" in data["detail"]

    async def test_upload_oversized_rejected(self, app_with_mocks):
        """Upload >50MB file → 413 with size limit message."""
        big_content = b"%PDF-1.4 " + b"\x00" * (51 * 1024 * 1024)
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("big.pdf", BytesIO(big_content), "application/pdf")},
            )

        assert response.status_code == 413

    async def test_upload_empty_file_rejected(self, app_with_mocks):
        """Upload empty file → 422."""
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
            )

        assert response.status_code == 422

    async def test_upload_invalid_magic_bytes(self, app_with_mocks):
        """Upload file with .pdf ext but non-PDF content → 422."""
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("fake.pdf", BytesIO(b"not a pdf file"), "application/pdf")},
            )

        assert response.status_code == 422

    async def test_upload_parse_failure(self, app_with_mocks, mock_pdf_parser):
        """PDF parser raises error → 422 with parse error detail."""
        from src.exceptions import PDFParsingError

        mock_pdf_parser.parse_pdf = AsyncMock(side_effect=PDFParsingError("Docling crashed"))

        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("test.pdf", BytesIO(VALID_PDF_BYTES), "application/pdf")},
            )

        assert response.status_code == 422
        assert "parse" in response.json()["detail"].lower() or "Docling" in response.json()["detail"]

    async def test_upload_indexing_failure_graceful(self, app_with_mocks, mock_embeddings_client):
        """Indexing fails but paper saved → 200 with warning."""
        mock_embeddings_client.embed_passages = AsyncMock(side_effect=Exception("Jina down"))

        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("test.pdf", BytesIO(VALID_PDF_BYTES), "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["indexing_status"] == "failed"
        assert len(data["warnings"]) > 0
        assert data["paper_id"] is not None

    async def test_upload_response_schema(self, app_with_mocks):
        """Verify response matches UploadResponse schema exactly."""
        async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("test.pdf", BytesIO(VALID_PDF_BYTES), "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        # Validate all fields are present and correctly typed
        parsed = UploadResponse(**data)
        assert isinstance(parsed.paper_id, uuid.UUID)
        assert isinstance(parsed.title, str)
        assert isinstance(parsed.authors, list)
        assert isinstance(parsed.abstract, str)
        assert isinstance(parsed.page_count, int)
        assert isinstance(parsed.chunks_indexed, int)
        assert parsed.parsing_status in ("success", "partial")
        assert parsed.indexing_status in ("success", "partial", "failed")
        assert isinstance(parsed.warnings, list)
        assert isinstance(parsed.message, str)
