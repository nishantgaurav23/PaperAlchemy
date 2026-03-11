"""Tests for S4.4 — Hybrid Search (BM25 + KNN + RRF).

Tests cover:
- Schema validation (FR-1, FR-2)
- Result mapping from raw OpenSearch hits (FR-5)
- Hybrid search endpoint with DI mocks (FR-3, FR-4, FR-6, FR-7)
- BM25 fallback on embedding failure (FR-3)
- Health check endpoint (FR-8)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from src.schemas.api.search import HybridSearchRequest, SearchHit, SearchResponse

# ── Fixtures ──────────────────────────────────────────────────────


def _fake_opensearch_hit(
    arxiv_id: str = "2301.00001",
    title: str = "Test Paper",
    score: float = 0.85,
) -> dict:
    """Return a raw hit dict mimicking OpenSearch _format_results output."""
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["Author A", "Author B"],
        "abstract": "This is an abstract.",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "score": score,
        "chunk_text": "Some chunk text about transformers.",
        "chunk_id": f"{arxiv_id}_chunk_0",
        "section_title": "Introduction",
        "highlights": {"chunk_text": ["<em>transformers</em> are great"]},
    }


def _fake_search_results(n: int = 3) -> dict:
    """Return a dict matching OpenSearchClient.search_unified() output."""
    hits = [_fake_opensearch_hit(arxiv_id=f"2301.0000{i}", score=0.9 - i * 0.1) for i in range(n)]
    return {"total": n, "hits": hits}


def _make_mock_opensearch(healthy: bool = True, results: dict | None = None):
    """Create a mock OpenSearchClient."""
    mock = MagicMock()
    mock.health_check.return_value = healthy
    mock.search_unified.return_value = results or _fake_search_results()
    return mock


def _make_mock_embeddings(embedding: list[float] | None = None, error: Exception | None = None):
    """Create a mock JinaEmbeddingsClient."""
    mock = AsyncMock()
    if error:
        mock.embed_query.side_effect = error
    else:
        mock.embed_query.return_value = embedding or [0.1] * 1024
    return mock


def _create_test_app():
    """Create a FastAPI test app with search router and DI overrides."""
    from src.dependency import EmbeddingsDep, OpenSearchDep
    from src.main import create_app

    app = create_app()

    mock_os = _make_mock_opensearch()
    mock_emb = _make_mock_embeddings()

    app.dependency_overrides[OpenSearchDep] = lambda: mock_os
    app.dependency_overrides[EmbeddingsDep] = lambda: mock_emb

    return app, mock_os, mock_emb


# ── FR-1: Search Request Schema Tests ────────────────────────────


class TestHybridSearchRequest:
    def test_valid_request(self):
        req = HybridSearchRequest(query="transformers attention")
        assert req.query == "transformers attention"

    def test_defaults(self):
        req = HybridSearchRequest(query="test")
        assert req.size == 10
        assert req.from_ == 0
        assert req.use_hybrid is True
        assert req.latest_papers is False
        assert req.categories is None
        assert req.min_score is None

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="x" * 501)

    def test_size_out_of_range(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", size=0)
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", size=101)

    def test_from_negative_rejected(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", **{"from": -1})

    def test_min_score_range(self):
        with pytest.raises(ValidationError):
            HybridSearchRequest(query="test", min_score=1.5)

    def test_categories_accepted(self):
        req = HybridSearchRequest(query="test", categories=["cs.AI", "cs.CL"])
        assert req.categories == ["cs.AI", "cs.CL"]


# ── FR-2: Search Response Schema Tests ───────────────────────────


class TestSearchHit:
    def test_from_raw_dict(self):
        raw = _fake_opensearch_hit()
        hit = SearchHit(**raw)
        assert hit.arxiv_id == "2301.00001"
        assert hit.title == "Test Paper"
        assert hit.score == 0.85
        assert hit.chunk_text == "Some chunk text about transformers."
        assert hit.chunk_id == "2301.00001_chunk_0"
        assert hit.section_title == "Introduction"
        assert len(hit.authors) == 2

    def test_missing_fields_use_defaults(self):
        hit = SearchHit(**{"arxiv_id": "2301.00001"})
        assert hit.title == ""
        assert hit.authors == []
        assert hit.score == 0.0
        assert hit.chunk_text == ""
        assert hit.highlights == {}
        assert hit.section_title is None


class TestSearchResponse:
    def test_response_construction(self):
        resp = SearchResponse(
            query="test",
            total=1,
            hits=[SearchHit(arxiv_id="2301.00001")],
            size=10,
            from_=0,
            search_mode="hybrid",
        )
        assert resp.search_mode == "hybrid"
        assert resp.total == 1


# ── FR-5: Result Mapping Tests ───────────────────────────────────


class TestResultMapping:
    def test_map_raw_hits_to_search_hits(self):
        raw_results = _fake_search_results(3)
        hits = [SearchHit(**h) for h in raw_results["hits"]]
        assert len(hits) == 3
        assert hits[0].arxiv_id == "2301.00000"
        assert hits[2].arxiv_id == "2301.00002"

    def test_map_empty_results(self):
        hits = [SearchHit(**h) for h in []]
        assert hits == []

    def test_map_hit_with_partial_data(self):
        raw = {"arxiv_id": "2301.00001", "score": 0.5}
        hit = SearchHit(**raw)
        assert hit.arxiv_id == "2301.00001"
        assert hit.title == ""
        assert hit.chunk_text == ""


# ── FR-3, FR-4, FR-6, FR-7: Search Endpoint Tests ────────────────


class TestSearchEndpoint:
    @pytest.fixture
    def app_with_mocks(self):
        from src.dependency import get_embeddings_client, get_opensearch_client
        from src.main import create_app

        app = create_app()
        mock_os = _make_mock_opensearch()
        mock_emb = _make_mock_embeddings()

        app.dependency_overrides[get_opensearch_client] = lambda: mock_os
        app.dependency_overrides[get_embeddings_client] = lambda: mock_emb

        return app, mock_os, mock_emb

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "transformers attention"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_mode"] == "hybrid"
        assert data["total"] == 3
        assert len(data["hits"]) == 3
        mock_emb.embed_query.assert_awaited_once_with("transformers attention")
        mock_os.search_unified.assert_called_once()

    @pytest.mark.asyncio
    async def test_bm25_fallback_on_embedding_failure(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        from src.exceptions import EmbeddingServiceError

        mock_emb.embed_query.side_effect = EmbeddingServiceError("Jina API timeout")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "transformers"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_mode"] == "bm25"
        # search_unified should be called with query_embedding=None
        call_kwargs = mock_os.search_unified.call_args
        assert call_kwargs.kwargs.get("query_embedding") is None or call_kwargs[1].get("query_embedding") is None

    @pytest.mark.asyncio
    async def test_bm25_only_when_use_hybrid_false(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "transformers", "use_hybrid": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["search_mode"] == "bm25"
        # embed_query should NOT have been called
        mock_emb.embed_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_opensearch_down(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        mock_os.health_check.return_value = False

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "transformers"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_search_empty_results(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        mock_os.search_unified.return_value = {"total": 0, "hits": []}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "nonexistent topic"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["hits"] == []

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "test", "categories": ["cs.AI"]})
        assert resp.status_code == 200
        call_kwargs = mock_os.search_unified.call_args
        assert call_kwargs.kwargs.get("categories") == ["cs.AI"] or call_kwargs[1].get("categories") == ["cs.AI"]

    @pytest.mark.asyncio
    async def test_search_pagination(self, app_with_mocks):
        app, mock_os, mock_emb = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "test", "size": 5, "from": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == 5
        call_kwargs = mock_os.search_unified.call_args
        assert call_kwargs.kwargs.get("size") == 5 or call_kwargs[1].get("size") == 5
        assert call_kwargs.kwargs.get("from_") == 10 or call_kwargs[1].get("from_") == 10

    @pytest.mark.asyncio
    async def test_request_validation_empty_query(self, app_with_mocks):
        app, _, _ = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_request_validation_query_too_long(self, app_with_mocks):
        app, _, _ = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={"query": "x" * 501})
        assert resp.status_code == 422


# ── FR-8: Health Check Endpoint Tests ─────────────────────────────


class TestSearchHealthEndpoint:
    @pytest.fixture
    def app_with_mocks(self):
        from src.dependency import get_opensearch_client
        from src.main import create_app

        app = create_app()
        mock_os = _make_mock_opensearch(healthy=True)

        app.dependency_overrides[get_opensearch_client] = lambda: mock_os
        return app, mock_os

    @pytest.mark.asyncio
    async def test_health_ok(self, app_with_mocks):
        app, mock_os = app_with_mocks
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/search/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["opensearch"] is True

    @pytest.mark.asyncio
    async def test_health_opensearch_down(self, app_with_mocks):
        app, mock_os = app_with_mocks
        mock_os.health_check.return_value = False
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/search/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["opensearch"] is False
