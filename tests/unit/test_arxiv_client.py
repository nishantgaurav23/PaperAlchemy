"""Tests for arXiv API client (S3.2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.schemas.arxiv import ArxivPaper
from src.services.arxiv.client import ArxivClient
from src.services.arxiv.factory import make_arxiv_client

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>  Attention Is All You Need
</title>
    <summary>  We propose a new architecture based on attention.
</summary>
    <published>2024-01-15T00:00:00Z</published>
    <updated>2024-01-16T00:00:00Z</updated>
    <author><name>Vaswani</name></author>
    <author><name>Shazeer</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.67890v2</id>
    <title>BERT: Pre-training</title>
    <summary>We introduce BERT.</summary>
    <published>2024-01-10T00:00:00Z</published>
    <updated>2024-01-10T00:00:00Z</updated>
    <author><name>Devlin</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>"""

EMPTY_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:totalResults>
</feed>"""

MALFORMED_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.11111v1</id>
    <title>Good Paper</title>
    <summary>Abstract here.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-01-01T00:00:00Z</updated>
    <author><name>Author</name></author>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <title>Missing ID Entry</title>
  </entry>
</feed>"""


@pytest.fixture
def client(tmp_path: Path) -> ArxivClient:
    """Create a test ArxivClient with a temp cache dir."""
    return ArxivClient(
        base_url="https://export.arxiv.org/api/query",
        rate_limit_delay=3.0,
        max_results=10,
        search_category="cs.AI",
        timeout=30,
        max_retries=3,
        retry_delay=0.01,  # Fast retries for tests
        cache_dir=str(tmp_path / "pdfs"),
    )


# ---------------------------------------------------------------------------
# FR-1: ArxivPaper Schema
# ---------------------------------------------------------------------------


class TestArxivPaperSchema:
    def test_create_valid(self):
        paper = ArxivPaper(
            arxiv_id="2401.12345",
            title="Test Paper",
            authors=["Author A", "Author B"],
            abstract="This is an abstract.",
            categories=["cs.AI"],
            published_date="2024-01-15T00:00:00Z",
            updated_date="2024-01-16T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        )
        assert paper.arxiv_id == "2401.12345"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345.pdf"

    def test_updated_date_optional(self):
        paper = ArxivPaper(
            arxiv_id="2401.12345",
            title="Test",
            authors=[],
            abstract="",
            categories=[],
            published_date="2024-01-15T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        )
        assert paper.updated_date is None

    def test_arxiv_url_property(self):
        paper = ArxivPaper(
            arxiv_id="2401.12345",
            title="Test",
            authors=[],
            abstract="",
            categories=[],
            published_date="2024-01-15",
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        )
        assert paper.arxiv_url == "https://arxiv.org/abs/2401.12345"


# ---------------------------------------------------------------------------
# FR-2: Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_first_request_no_delay(self, client: ArxivClient):
        """First request should not wait."""
        assert client._last_request_time is None
        await client._wait_for_rate_limit()
        assert client._last_request_time is not None

    @pytest.mark.asyncio
    async def test_enforces_minimum_delay(self, client: ArxivClient):
        """Second request within delay window must sleep."""
        # Simulate a recent request
        loop = asyncio.get_event_loop()
        client._last_request_time = loop.time()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._wait_for_rate_limit()
            mock_sleep.assert_called_once()
            # Sleep time should be close to rate_limit_delay (3.0s)
            sleep_time = mock_sleep.call_args[0][0]
            assert 0 < sleep_time <= client.rate_limit_delay

    @pytest.mark.asyncio
    async def test_no_delay_after_enough_time(self, client: ArxivClient):
        """No sleep needed if enough time has passed."""
        loop = asyncio.get_event_loop()
        # Simulate request from 10 seconds ago
        client._last_request_time = loop.time() - 10.0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._wait_for_rate_limit()
            mock_sleep.assert_not_called()

    def test_rate_limit_enforced_minimum(self):
        """Rate limit delay should be at least 3.0s even if set lower."""
        c = ArxivClient(rate_limit_delay=1.0, cache_dir="/tmp/test_arxiv")
        assert c.rate_limit_delay >= 3.0


# ---------------------------------------------------------------------------
# FR-3: Retry Logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self, client: ArxivClient):
        """Successful request returns response text."""
        mock_response = httpx.Response(200, text="OK", request=httpx.Request("GET", "https://example.com"))

        with (
            patch.object(httpx.AsyncClient, "get", return_value=mock_response),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
        ):
            result = await client._make_request("https://example.com")
            assert result == "OK"

    @pytest.mark.asyncio
    async def test_retry_on_503(self, client: ArxivClient):
        """Should retry on 503 with backoff."""
        resp_503 = httpx.Response(503, text="Unavailable", request=httpx.Request("GET", "https://example.com"))
        resp_200 = httpx.Response(200, text="OK", request=httpx.Request("GET", "https://example.com"))

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return resp_503
            return resp_200

        with (
            patch.object(httpx.AsyncClient, "get", side_effect=mock_get),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client._make_request("https://example.com")
            assert result == "OK"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_429(self, client: ArxivClient):
        """Should retry on 429 with extended wait."""
        resp_429 = httpx.Response(429, text="Rate Limited", request=httpx.Request("GET", "https://example.com"))
        resp_200 = httpx.Response(200, text="OK", request=httpx.Request("GET", "https://example.com"))

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return resp_429
            return resp_200

        with (
            patch.object(httpx.AsyncClient, "get", side_effect=mock_get),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await client._make_request("https://example.com")
            assert result == "OK"
            # 429 wait should be longer (retry_delay * 2^attempt * 10)
            mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, client: ArxivClient):
        """Should retry on request timeout."""
        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("timeout")
            return httpx.Response(200, text="OK", request=httpx.Request("GET", "https://example.com"))

        with (
            patch.object(httpx.AsyncClient, "get", side_effect=mock_get),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client._make_request("https://example.com")
            assert result == "OK"

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self, client: ArxivClient):
        """Non-retryable errors should raise immediately."""
        from src.exceptions import ArxivAPIError

        resp_400 = httpx.Response(400, text="Bad Request", request=httpx.Request("GET", "https://example.com"))

        with (
            patch.object(httpx.AsyncClient, "get", return_value=resp_400),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            pytest.raises(ArxivAPIError, match="HTTP 400"),
        ):
            await client._make_request("https://example.com")

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client: ArxivClient):
        """Should raise after exhausting all retries."""
        from src.exceptions import ArxivAPIError

        resp_503 = httpx.Response(503, text="Unavailable", request=httpx.Request("GET", "https://example.com"))

        with (
            patch.object(httpx.AsyncClient, "get", return_value=resp_503),
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ArxivAPIError, match="Failed after"),
        ):
            await client._make_request("https://example.com")


# ---------------------------------------------------------------------------
# FR-4: Query Building
# ---------------------------------------------------------------------------


class TestQueryBuilding:
    def test_category_only(self, client: ArxivClient):
        q = client._build_query(category="cs.AI")
        assert q == "cat:cs.AI"

    def test_default_category(self, client: ArxivClient):
        q = client._build_query()
        assert q == "cat:cs.AI"

    def test_date_range(self, client: ArxivClient):
        q = client._build_query(from_date="20240101", to_date="20240131")
        assert "cat:cs.AI" in q
        assert "submittedDate:[20240101 TO 20240131]" in q

    def test_from_date_only(self, client: ArxivClient):
        q = client._build_query(from_date="20240101")
        assert "submittedDate:[20240101 TO 99991231]" in q

    def test_to_date_only(self, client: ArxivClient):
        q = client._build_query(to_date="20240131")
        assert "submittedDate:[00000101 TO 20240131]" in q

    def test_search_query(self, client: ArxivClient):
        q = client._build_query(search_query="transformers")
        assert "all:transformers" in q

    def test_combined(self, client: ArxivClient):
        q = client._build_query(
            category="cs.CL",
            from_date="20240101",
            to_date="20240131",
            search_query="transformers",
        )
        assert "cat:cs.CL" in q
        assert "submittedDate:[20240101 TO 20240131]" in q
        assert "all:transformers" in q
        assert " AND " in q


# ---------------------------------------------------------------------------
# FR-5: Fetch Papers
# ---------------------------------------------------------------------------


class TestFetchPapers:
    @pytest.mark.asyncio
    async def test_fetch_papers_success(self, client: ArxivClient):
        """Parse valid Atom feed into ArxivPaper list."""
        with (
            patch.object(client, "_make_request", new_callable=AsyncMock, return_value=SAMPLE_ATOM_FEED),
        ):
            papers = await client.fetch_papers()
            assert len(papers) == 2

            p1 = papers[0]
            assert p1.arxiv_id == "2401.12345"
            assert p1.title == "Attention Is All You Need"
            assert p1.authors == ["Vaswani", "Shazeer"]
            assert "cs.AI" in p1.categories
            assert "cs.CL" in p1.categories
            assert p1.pdf_url == "https://arxiv.org/pdf/2401.12345.pdf"
            # updated != published so updated_date should be set
            assert p1.updated_date is not None

            p2 = papers[1]
            assert p2.arxiv_id == "2401.67890"
            # updated == published so updated_date should be None
            assert p2.updated_date is None

    @pytest.mark.asyncio
    async def test_fetch_papers_empty(self, client: ArxivClient):
        """Empty feed returns empty list."""
        with patch.object(client, "_make_request", new_callable=AsyncMock, return_value=EMPTY_ATOM_FEED):
            papers = await client.fetch_papers()
            assert papers == []

    @pytest.mark.asyncio
    async def test_fetch_papers_malformed_entry(self, client: ArxivClient):
        """Malformed entries are skipped gracefully."""
        with patch.object(client, "_make_request", new_callable=AsyncMock, return_value=MALFORMED_ATOM_FEED):
            papers = await client.fetch_papers()
            # First entry is good, second has no id — should be skipped or handled
            assert len(papers) >= 1
            assert papers[0].arxiv_id == "2401.11111"

    @pytest.mark.asyncio
    async def test_fetch_papers_passes_params(self, client: ArxivClient):
        """Verify URL params are built correctly."""
        with patch.object(client, "_make_request", new_callable=AsyncMock, return_value=EMPTY_ATOM_FEED) as mock_req:
            await client.fetch_papers(
                max_results=5,
                category="cs.CL",
                sort_by="relevance",
                sort_order="ascending",
                start=10,
            )
            url = mock_req.call_args[0][0]
            assert "max_results=5" in url
            assert "sortBy=relevance" in url
            assert "sortOrder=ascending" in url
            assert "start=10" in url


# ---------------------------------------------------------------------------
# FR-6: PDF Download
# ---------------------------------------------------------------------------


class TestDownloadPdf:
    @pytest.mark.asyncio
    async def test_download_success(self, client: ArxivClient):
        """Successful PDF download with validation."""
        pdf_bytes = b"%PDF-1.4 fake pdf content"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "100"}
        mock_response.content = pdf_bytes

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
            assert path is not None
            assert path.exists()
            assert path.name == "2401.12345.pdf"

    @pytest.mark.asyncio
    async def test_download_cached(self, client: ArxivClient, tmp_path: Path):
        """Return cached path without downloading."""
        # Pre-create cached file
        client.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = client.cache_dir / "2401.12345.pdf"
        cached.write_bytes(b"%PDF-1.4 cached")

        path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
        assert path == cached

    @pytest.mark.asyncio
    async def test_download_force(self, client: ArxivClient):
        """Force re-download even if cached."""
        client.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = client.cache_dir / "2401.12345.pdf"
        cached.write_bytes(b"%PDF-1.4 old")

        new_content = b"%PDF-1.4 new content"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "100"}
        mock_response.content = new_content

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf(
                "2401.12345", "https://arxiv.org/pdf/2401.12345.pdf", force=True
            )
            assert path is not None
            assert path.read_bytes() == new_content

    @pytest.mark.asyncio
    async def test_download_invalid_content_type(self, client: ArxivClient):
        """Return None on non-PDF content type."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html", "Content-Length": "100"}
        mock_response.content = b"<html>not a pdf</html>"

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
            assert path is None

    @pytest.mark.asyncio
    async def test_download_too_large(self, client: ArxivClient):
        """Return None for >50MB."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": str(51 * 1024 * 1024),
        }

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
            assert path is None

    @pytest.mark.asyncio
    async def test_download_invalid_magic(self, client: ArxivClient):
        """Return None and cleanup on bad magic bytes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "100"}
        mock_response.content = b"NOT-A-PDF content here"

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
            assert path is None
            # Temp file should be cleaned up
            assert not (client.cache_dir / "2401.12345.tmp").exists()

    @pytest.mark.asyncio
    async def test_download_http_error(self, client: ArxivClient):
        """Return None on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}

        with (
            patch.object(client, "_wait_for_rate_limit", new_callable=AsyncMock),
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response),
        ):
            path = await client.download_pdf("2401.12345", "https://arxiv.org/pdf/2401.12345.pdf")
            assert path is None


# ---------------------------------------------------------------------------
# FR-7: Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_client(self):
        """make_arxiv_client returns an ArxivClient instance."""
        make_arxiv_client.cache_clear()
        client = make_arxiv_client()
        assert isinstance(client, ArxivClient)

    def test_caches_instance(self):
        """Same instance on multiple calls."""
        make_arxiv_client.cache_clear()
        c1 = make_arxiv_client()
        c2 = make_arxiv_client()
        assert c1 is c2
        make_arxiv_client.cache_clear()
