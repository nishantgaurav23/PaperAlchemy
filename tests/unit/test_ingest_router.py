"""Tests for the ingestion API endpoint (POST /api/v1/ingest/fetch).

Tests the FastAPI endpoint that orchestrates: arXiv fetch -> PDF download -> parse -> store -> chunk -> embed -> index.
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
        parser_used="pymupdf",
        parser_time_seconds=0.5,
    )


def _make_paper_orm(arxiv_id: str = "2602.00001"):
    paper = MagicMock()
    paper.id = uuid.uuid4()
    paper.arxiv_id = arxiv_id
    paper.title = "Test Paper on Transformers"
    paper.abstract = "A test abstract about transformers."
    paper.authors = ["Alice", "Bob"]
    paper.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
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
def mock_embeddings_client():
    client = AsyncMock()
    # Return a 1024-dim vector per chunk
    client.embed_passages = AsyncMock(return_value=[[0.1] * 1024])
    return client


@pytest.fixture
def mock_opensearch_client():
    client = MagicMock()
    client.setup_indices = MagicMock(return_value={"hybrid_index": True, "rrf_pipeline": True})
    client.bulk_index_chunks = MagicMock(return_value={"success": 1, "failed": 0})
    client.delete_paper_chunks = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value="mock LLM response")
    return provider


@pytest.fixture
def app_with_mocks(mock_paper_repo, mock_session, mock_embeddings_client, mock_opensearch_client, mock_llm_provider):
    """Create a minimal FastAPI app with the ingest router and mocked deps."""
    from src.dependency import get_db_session, get_embeddings_client, get_llm_provider, get_opensearch_client, get_paper_repository

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db_session] = lambda: mock_session
    app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
    app.dependency_overrides[get_embeddings_client] = lambda: mock_embeddings_client
    app.dependency_overrides[get_opensearch_client] = lambda: mock_opensearch_client
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider
    return app


@pytest.mark.asyncio
class TestIngestEndpoint:
    """Test POST /api/v1/ingest/fetch endpoint."""

    async def test_ingest_endpoint_success(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint fetches papers, downloads PDFs, parses, stores, indexes, and runs analysis."""
        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 1
            assert data["pdfs_downloaded"] == 1
            assert data["pdfs_parsed"] == 1
            assert data["papers_stored"] == 1
            assert data["chunks_indexed"] >= 0
            assert "2602.00001" in data["arxiv_ids"]
            assert data["errors"] == []
            assert "processing_time" in data

    async def test_ingest_endpoint_no_papers(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint handles empty arXiv results gracefully."""
        mock_arxiv_client.fetch_papers = AsyncMock(return_value=[])

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 0
            assert data["arxiv_ids"] == []

    async def test_ingest_endpoint_pdf_download_failure(
        self, app_with_mocks, mock_arxiv_client, mock_pdf_parser, mock_paper_repo
    ):
        """Endpoint handles PDF download failure gracefully and marks status as failed."""
        mock_arxiv_client.download_pdf = AsyncMock(return_value=None)

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            assert data["papers_fetched"] == 1
            assert data["pdfs_downloaded"] == 0
            assert data["pdfs_parsed"] == 0
            assert len(data["errors"]) > 0
            assert "Download failed" in data["errors"][0]

            # Should have called update_parsing_status with "failed"
            mock_paper_repo.update_parsing_status.assert_called_once()
            call_kwargs = mock_paper_repo.update_parsing_status.call_args[1]
            assert call_kwargs["status"] == "failed"

    async def test_ingest_endpoint_default_date(self, app_with_mocks, mock_arxiv_client, mock_pdf_parser):
        """Endpoint defaults target_date to yesterday when not provided."""
        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
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
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
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
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                r1 = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})
                r2 = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r1.json()["papers_fetched"] == r2.json()["papers_fetched"]

    async def test_ingest_endpoint_indexing_failure_graceful(
        self, app_with_mocks, mock_arxiv_client, mock_pdf_parser, mock_opensearch_client
    ):
        """Endpoint continues even if indexing fails (paper still saved)."""
        mock_opensearch_client.bulk_index_chunks = MagicMock(side_effect=RuntimeError("OpenSearch down"))

        with (
            patch("src.routers.ingest.make_arxiv_client", return_value=mock_arxiv_client),
            patch("src.routers.ingest.make_pdf_parser_service", return_value=mock_pdf_parser),
            patch("src.routers.ingest.run_ai_analysis", new_callable=AsyncMock, return_value={"warnings": []}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app_with_mocks), base_url="http://test") as client:
                response = await client.post("/api/v1/ingest/fetch", json={"target_date": "20260309"})

            assert response.status_code == 200
            data = response.json()
            # Paper was fetched, downloaded, and parsed — but indexing failed
            assert data["papers_fetched"] == 1
            assert data["pdfs_downloaded"] == 1
            assert data["pdfs_parsed"] == 1
            assert data["chunks_indexed"] == 0
            assert any("Indexing failed" in e for e in data["errors"])


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
        assert resp.chunks_indexed == 0
        assert resp.arxiv_ids == []
        assert resp.errors == []

    def test_ingest_response_with_chunks(self):
        from src.schemas.api.ingest import IngestResponse

        resp = IngestResponse(papers_fetched=5, chunks_indexed=42)
        assert resp.papers_fetched == 5
        assert resp.chunks_indexed == 42
