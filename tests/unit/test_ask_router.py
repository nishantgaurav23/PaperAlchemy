"""Tests for S5.3 — Streaming Responses (SSE).

Tests cover:
- SSE streaming endpoint (FR-1, FR-4): token events, sources event, done event
- Non-streaming fallback (FR-3): JSON AskResponse
- Request/response models (FR-2): validation, schema fields
- Error handling: LLM errors → error events, empty query → 422
- DI integration (FR-5): RAGChainDep override
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference

# ── Helpers ──────────────────────────────────────────────────────


def _make_source(index: int = 1, arxiv_id: str = "1706.03762") -> SourceReference:
    return SourceReference(
        index=index,
        arxiv_id=arxiv_id,
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer"],
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        chunk_text="Self-attention mechanism allows...",
        score=0.95,
    )


def _make_rag_response(query: str = "What are transformers?") -> RAGResponse:
    return RAGResponse(
        answer="Transformers are based on self-attention [1]. They enable parallel processing [2].",
        sources=[
            _make_source(1, "1706.03762"),
            _make_source(2, "1810.04805"),
        ],
        query=query,
        retrieval_metadata=RetrievalMetadata(
            stages_executed=["multi_query", "hybrid_search", "rerank"],
            total_candidates=20,
            timings={"retrieval": 0.5},
        ),
        llm_metadata=LLMMetadata(provider="gemini", model="gemini-3-flash"),
    )


async def _fake_stream(*args, **kwargs):
    """Simulate RAGChain.aquery_stream() yielding tokens then [SOURCES]."""
    sources = [_make_source(1, "1706.03762"), _make_source(2, "1810.04805")]
    yield "Transformers "
    yield "are based on "
    yield "self-attention [1]."
    sources_json = json.dumps([s.model_dump() for s in sources])
    yield f"\n\n[SOURCES]{sources_json}"


async def _fake_empty_stream(*args, **kwargs):
    """Simulate RAGChain.aquery_stream() with no documents."""
    yield "I could not find any relevant papers in the knowledge base for your query."


async def _fake_error_stream(*args, **kwargs):
    """Simulate RAGChain.aquery_stream() that raises an LLM error."""
    from src.exceptions import LLMTimeoutError

    raise LLMTimeoutError("LLM generation timed out")
    yield  # noqa: F541 — unreachable yield makes this an async generator


def _create_test_app(mock_rag_chain=None):
    """Create a FastAPI test app with RAGChain DI override."""
    from src.dependency import get_rag_chain
    from src.main import create_app

    app = create_app()

    if mock_rag_chain is None:
        mock_rag_chain = AsyncMock()
        mock_rag_chain.aquery.return_value = _make_rag_response()
        mock_rag_chain.aquery_stream = _fake_stream

    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
    return app, mock_rag_chain


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = None

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
        elif line.startswith("data: "):
            current_data = line[len("data: ") :]
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": current_data})
            current_event = None
            current_data = None

    # Catch final event if no trailing blank line
    if current_event is not None and current_data is not None:
        events.append({"event": current_event, "data": current_data})

    return events


# ── FR-2: Request Model Validation ──────────────────────────────


class TestAskRequestValidation:
    def test_valid_request(self):
        from src.schemas.api.ask import AskRequest

        req = AskRequest(query="What are transformers?")
        assert req.query == "What are transformers?"
        assert req.stream is True  # default
        assert req.top_k is None
        assert req.categories is None
        assert req.temperature is None

    def test_stream_false(self):
        from src.schemas.api.ask import AskRequest

        req = AskRequest(query="test", stream=False)
        assert req.stream is False

    def test_empty_query_rejected(self):
        from pydantic import ValidationError
        from src.schemas.api.ask import AskRequest

        with pytest.raises(ValidationError):
            AskRequest(query="")

    def test_whitespace_only_query_rejected(self):
        from pydantic import ValidationError
        from src.schemas.api.ask import AskRequest

        with pytest.raises(ValidationError):
            AskRequest(query="   ")

    def test_temperature_range(self):
        from pydantic import ValidationError
        from src.schemas.api.ask import AskRequest

        AskRequest(query="test", temperature=0.0)
        AskRequest(query="test", temperature=2.0)
        with pytest.raises(ValidationError):
            AskRequest(query="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            AskRequest(query="test", temperature=2.1)

    def test_top_k_positive(self):
        from pydantic import ValidationError
        from src.schemas.api.ask import AskRequest

        AskRequest(query="test", top_k=5)
        with pytest.raises(ValidationError):
            AskRequest(query="test", top_k=0)

    def test_categories_accepted(self):
        from src.schemas.api.ask import AskRequest

        req = AskRequest(query="test", categories=["cs.AI", "cs.CL"])
        assert req.categories == ["cs.AI", "cs.CL"]


# ── FR-1: SSE Streaming Endpoint ────────────────────────────────


class TestAskStreamingEndpoint:
    @pytest.fixture
    def app_with_mock(self):
        return _create_test_app()

    @pytest.mark.asyncio
    async def test_ask_streaming_returns_event_stream(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_ask_streaming_yields_token_events(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) >= 1
        # Each token event has JSON data with "text" field
        for te in token_events:
            data = json.loads(te["data"])
            assert "text" in data
            assert isinstance(data["text"], str)

    @pytest.mark.asyncio
    async def test_ask_streaming_yields_sources_event(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        events = _parse_sse_events(resp.text)
        sources_events = [e for e in events if e["event"] == "sources"]
        assert len(sources_events) == 1
        sources = json.loads(sources_events[0]["data"])
        assert isinstance(sources, list)
        assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_ask_streaming_yields_done_event(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_ask_streaming_sources_have_arxiv_links(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        events = _parse_sse_events(resp.text)
        sources_events = [e for e in events if e["event"] == "sources"]
        sources = json.loads(sources_events[0]["data"])
        for source in sources:
            assert "arxiv_url" in source
            assert source["arxiv_url"].startswith("https://arxiv.org/abs/")
            assert "title" in source
            assert "authors" in source
            assert "arxiv_id" in source

    @pytest.mark.asyncio
    async def test_ask_streaming_event_order(self, app_with_mock):
        """Token events come first, then sources, then done."""
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        events = _parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]
        # All tokens before sources, sources before done
        sources_idx = event_types.index("sources")
        done_idx = event_types.index("done")
        for i, et in enumerate(event_types):
            if et == "token":
                assert i < sources_idx
        assert sources_idx < done_idx


# ── FR-3: Non-Streaming Fallback ────────────────────────────────


class TestAskNonStreamingEndpoint:
    @pytest.fixture
    def app_with_mock(self):
        return _create_test_app()

    @pytest.mark.asyncio
    async def test_ask_non_streaming_returns_json(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?", "stream": False})
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_ask_non_streaming_has_citations(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?", "stream": False})
        data = resp.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["arxiv_id"] == "1706.03762"
        assert data["sources"][0]["arxiv_url"] == "https://arxiv.org/abs/1706.03762"

    @pytest.mark.asyncio
    async def test_ask_non_streaming_has_metadata(self, app_with_mock):
        app, _ = app_with_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?", "stream": False})
        data = resp.json()
        assert "retrieval_metadata" in data
        assert "llm_metadata" in data


# ── Edge Cases ───────────────────────────────────────────────────


class TestAskEdgeCases:
    @pytest.mark.asyncio
    async def test_ask_empty_query_returns_422(self):
        app, _ = _create_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_no_documents_streams_gracefully(self):
        mock_rag = AsyncMock()
        mock_rag.aquery_stream = _fake_empty_stream
        app, _ = _create_test_app(mock_rag)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "unknown topic"})
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) >= 1
        # Should have sources event (empty list) and done
        sources_events = [e for e in events if e["event"] == "sources"]
        assert len(sources_events) == 1
        sources = json.loads(sources_events[0]["data"])
        assert sources == []
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_ask_llm_error_streams_error_event(self):
        mock_rag = AsyncMock()
        mock_rag.aquery_stream = _fake_error_stream
        app, _ = _create_test_app(mock_rag)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?"})
        assert resp.status_code == 200  # SSE always returns 200, errors in stream
        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "error" in error_data

    @pytest.mark.asyncio
    async def test_ask_llm_error_non_streaming_returns_503(self):
        from src.exceptions import LLMTimeoutError

        mock_rag = AsyncMock()
        mock_rag.aquery.side_effect = LLMTimeoutError("timeout")
        app, _ = _create_test_app(mock_rag)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/ask", json={"query": "What are transformers?", "stream": False})
        assert resp.status_code == 503


# ── FR-5: DI and Parameter Forwarding ───────────────────────────


class TestAskParameterForwarding:
    @pytest.mark.asyncio
    async def test_ask_categories_forwarded(self):
        mock_rag = AsyncMock()
        mock_rag.aquery.return_value = _make_rag_response()
        app, _ = _create_test_app(mock_rag)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ask",
                json={"query": "test", "stream": False, "categories": ["cs.AI"]},
            )
        assert resp.status_code == 200
        mock_rag.aquery.assert_awaited_once()
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs.kwargs.get("categories") == ["cs.AI"]

    @pytest.mark.asyncio
    async def test_ask_temperature_forwarded(self):
        mock_rag = AsyncMock()
        mock_rag.aquery.return_value = _make_rag_response()
        app, _ = _create_test_app(mock_rag)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/ask",
                json={"query": "test", "stream": False, "temperature": 0.7},
            )
        assert resp.status_code == 200
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.7
