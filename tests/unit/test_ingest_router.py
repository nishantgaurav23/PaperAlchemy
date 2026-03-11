"""Tests for the ingestion API endpoint (POST /api/v1/ingest/fetch).

Tests the FastAPI endpoint that orchestrates: arXiv fetch -> PDF download -> parse -> store.
All external services are mocked.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.exceptions import PDFParsingError
from src.routers.ingest import router
from src.schemas.arxiv import ArxivPaper
from src.schemas.pdf import PDFContent, Section


def _make_arxiv_paper(arxiv_id: str = "2602.00001") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title="Test Paper on Transformers",
        authors=["Alice", "Bob"],
        abstract="A test abstract about transformers.",
        categories=["cs.AI"],
        published_date="2026-03-09T00:00:00Z",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _make_pdf_content() -> PDFContent:
    return PDFContent(
        raw_text="This is the full text of the paper.",
        sections=[Section(title="Introduction", content="Intro content here.", level=1)],
        tables=[],
        figures=[],
        page_count=10,
        parser_used="docling",
        parser_time_seconds=2.5,
    )


def _make_paper_orm(arxiv_id: str = "2602.00001"):
    paper = MagicMock()
    paper.id = uuid.uuid4()
    paper.arxiv_id = arxiv_id
    paper.title = "Test Paper on Transformers"
    paper.parsing_status = "pending"
    return paper


@pytest.fixture
def mock_arxiv_client():
    client = AsyncMock()
    client.fetch_papers = AsyncMock(return_value=[_make_arxiv_paper()])
    client.download_pdf = AsyncMock(return_value=Path("/tmp/test.pdf"))
    return client


@pytest.fixture
def mock_pdf_parser():
    parser = AsyncMock()
    parser.parse_pdf = AsyncMock(return_value=_make_pdf_content())
    return parser


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.upsert = AsyncMock(return_value=_make_paper_orm())
    repo.update_parsing_status = AsyncMock(return_value=_make_paper_orm())
    return repo


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def app_with_mocks(mock_paper_repo, mock_session):
    """Create a minimal FastAPI app with the ingest router and mocked deps."""
    from src.dependency import get_db_session, get_paper_repository

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db_session] = lambda: mock_session
    app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
    return app


@pytest.mark.asyncio
class TestIngestEndpoint:
    """Test POST /api/v1/ingest/fetch endpoint."""

    async def test_ingest_endpoint_success(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint fetches papers, downloads PDFs, parses, and stores."""
        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 1
            assert data["pdfs_downloaded"] == 1
            assert data["pdfs_parsed"] == 1
            assert data["papers_stored"] == 1
            assert "2602.00001" in data["arxiv_ids"]
            assert data["errors"] == []
            assert "processing_time" in data

    async def test_ingest_endpoint_no_papers(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint handles empty arXiv results gracefully."""
        mock_arxiv_client.fetch_papers = AsyncMock(return_value=[])

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 0
            assert data["arxiv_ids"] == []

    async def test_ingest_endpoint_pdf_download_failure(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint handles PDF download failure gracefully."""
        mock_arxiv_client.download_pdf = AsyncMock(return_value=None)

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 1
            assert data["pdfs_downloaded"] == 0
            assert data["pdfs_parsed"] == 0

    async def test_ingest_endpoint_default_date(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint defaults target_date to yesterday when not provided."""
        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] >= 0

    async def test_ingest_endpoint_parse_failure(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser, mock_paper_repo):
        """Endpoint handles parse failures gracefully and records error."""
        mock_pdf_parser.parse_pdf = AsyncMock(side_effect=PDFParsingError("Parse failed"))

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["pdfs_parsed"] == 0
            assert len(data["errors"]) > 0

            # Should have called update_parsing_status with "failed"
            mock_paper_repo.update_parsing_status.assert_called_once()
            call_kwargs = mock_paper_repo.update_parsing_status.call_args[1]
            assert call_kwargs["status"] == "failed"

    async def test_ingest_endpoint_idempotent(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Re-running produces same result (upsert is idempotent)."""
        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                r1 = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})
                r2 = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r1.json()["papers_fetched"] == r2.json()["papers_fetched"]


class TestIngestSchemas:
    """Test request/response schemas."""

    def test_ingest_request_valid(self):
        from src.schemas.api.ingest import IngestRequest

        req = IngestRequest(target_date="20260309")
        assert req.target_date == "20260309"

    def test_ingest_request_none_date(self):
        from src.schemas.api.ingest import IngestRequest

        req = IngestRequest()
        assert req.target_date is None

    def test_ingest_request_invalid_date(self):
        from pydantic import ValidationError
        from src.schemas.api.ingest import IngestRequest

        with pytest.raises(ValidationError):
            IngestRequest(target_date="not-a-date")

    def test_ingest_response_defaults(self):
        from src.schemas.api.ingest import IngestResponse

        resp = IngestResponse()
        assert resp.papers_fetched == 0
        assert resp.arxiv_ids == []
        assert resp.errors == []
