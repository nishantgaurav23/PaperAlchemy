"""Tests for S7.3 — Chat API.

TDD: All tests written before implementation.
Covers: ChatRequest/ChatResponse models, POST /chat (JSON + streaming),
session history, session clear, error handling, graceful degradation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from src.services.chat.follow_up import FollowUpResult
from src.services.chat.memory import ChatMessage
from src.services.rag.models import RAGResponse, SourceReference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rag_response(answer: str = "Transformers [1].", query: str = "test") -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=[
            SourceReference(
                index=1,
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                arxiv_url="https://arxiv.org/abs/1706.03762",
            )
        ],
        query=query,
    )


def _make_follow_up_result(
    query: str = "test",
    is_follow_up: bool = False,
    rewritten: str | None = None,
) -> FollowUpResult:
    return FollowUpResult(
        original_query=query,
        rewritten_query=rewritten or query,
        is_follow_up=is_follow_up,
        response=_make_rag_response(query=query),
    )


async def _mock_stream(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_follow_up_handler():
    handler = AsyncMock()
    handler.handle = AsyncMock(return_value=_make_follow_up_result())
    handler.handle_stream = AsyncMock(return_value=_mock_stream("Hello ", "world [1]."))
    return handler


@pytest.fixture()
def mock_memory():
    memory = AsyncMock()
    memory.get_history = AsyncMock(return_value=[])
    memory.clear_session = AsyncMock(return_value=True)
    memory.add_message = AsyncMock()
    return memory


@pytest.fixture()
def app(mock_follow_up_handler, mock_memory):
    """Create a test app with mocked dependencies."""
    from src.main import create_app

    app = create_app()

    # Override dependencies
    from src.dependency import get_conversation_memory, get_follow_up_handler

    app.dependency_overrides[get_follow_up_handler] = lambda: mock_follow_up_handler
    app.dependency_overrides[get_conversation_memory] = lambda: mock_memory
    return app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# FR-1: ChatRequest / ChatResponse validation
# ---------------------------------------------------------------------------


class TestChatRequestValidation:
    """Validate ChatRequest field constraints."""

    def test_valid_request(self):
        from src.schemas.api.chat import ChatRequest

        req = ChatRequest(session_id="sess-1", query="What is a transformer?")
        assert req.session_id == "sess-1"
        assert req.query == "What is a transformer?"
        assert req.stream is True
        assert req.top_k is None
        assert req.categories is None
        assert req.temperature is None

    def test_query_stripped(self):
        from src.schemas.api.chat import ChatRequest

        req = ChatRequest(session_id="s1", query="  hello  ")
        assert req.query == "hello"

    def test_empty_query_rejected(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="s1", query="")

    def test_whitespace_query_rejected(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="s1", query="   ")

    def test_empty_session_id_rejected(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="", query="hello")

    def test_session_id_max_length(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="x" * 129, query="hello")

    def test_temperature_bounds(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="s1", query="hi", temperature=-0.1)
        with pytest.raises(ValueError):
            ChatRequest(session_id="s1", query="hi", temperature=2.1)

    def test_top_k_positive(self):
        from src.schemas.api.chat import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest(session_id="s1", query="hi", top_k=0)

    def test_stream_default_true(self):
        from src.schemas.api.chat import ChatRequest

        req = ChatRequest(session_id="s1", query="hi")
        assert req.stream is True

    def test_non_streaming_mode(self):
        from src.schemas.api.chat import ChatRequest

        req = ChatRequest(session_id="s1", query="hi", stream=False)
        assert req.stream is False


class TestChatResponseModel:
    """Verify ChatResponse serialization."""

    def test_basic_response(self):
        from src.schemas.api.chat import ChatResponse

        resp = ChatResponse(
            answer="Transformers [1].",
            sources=[
                SourceReference(
                    index=1,
                    arxiv_id="1706.03762",
                    title="Attention Is All You Need",
                    authors=["Vaswani"],
                    arxiv_url="https://arxiv.org/abs/1706.03762",
                )
            ],
            session_id="sess-1",
            is_follow_up=False,
            rewritten_query=None,
            query="What is a transformer?",
        )
        assert resp.answer == "Transformers [1]."
        assert len(resp.sources) == 1
        assert resp.session_id == "sess-1"
        assert resp.is_follow_up is False
        assert resp.rewritten_query is None

    def test_follow_up_response(self):
        from src.schemas.api.chat import ChatResponse

        resp = ChatResponse(
            answer="It has limitations [1].",
            sources=[],
            session_id="sess-1",
            is_follow_up=True,
            rewritten_query="What are the limitations of transformers?",
            query="What about its limitations?",
        )
        assert resp.is_follow_up is True
        assert resp.rewritten_query == "What are the limitations of transformers?"

    def test_serialization_roundtrip(self):
        from src.schemas.api.chat import ChatResponse

        resp = ChatResponse(
            answer="Test [1].",
            sources=[],
            session_id="s1",
            is_follow_up=False,
            rewritten_query=None,
            query="test",
        )
        data = resp.model_dump()
        assert data["session_id"] == "s1"
        assert data["is_follow_up"] is False


# ---------------------------------------------------------------------------
# FR-2: POST /api/v1/chat — JSON mode
# ---------------------------------------------------------------------------


class TestChatEndpointJSON:
    """POST /chat with stream=False returns ChatResponse JSON."""

    @pytest.mark.asyncio()
    async def test_json_mode_returns_chat_response(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?", "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["session_id"] == "sess-1"
        assert "is_follow_up" in data
        assert "query" in data
        mock_follow_up_handler.handle.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_json_mode_passes_rag_params(self, client, mock_follow_up_handler):
        await client.post(
            "/api/v1/chat",
            json={
                "session_id": "s1",
                "query": "transformers",
                "stream": False,
                "top_k": 3,
                "categories": ["cs.AI"],
                "temperature": 0.5,
            },
        )
        call_kwargs = mock_follow_up_handler.handle.call_args
        assert call_kwargs.kwargs.get("top_k") == 3 or call_kwargs[1].get("top_k") == 3

    @pytest.mark.asyncio()
    async def test_json_mode_follow_up(self, client, mock_follow_up_handler):
        mock_follow_up_handler.handle.return_value = _make_follow_up_result(
            query="What about its limitations?",
            is_follow_up=True,
            rewritten="What are the limitations of transformers?",
        )
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "query": "What about its limitations?", "stream": False},
        )
        data = response.json()
        assert data["is_follow_up"] is True
        assert data["rewritten_query"] == "What are the limitations of transformers?"

    @pytest.mark.asyncio()
    async def test_empty_query_returns_422(self, client):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "query": "", "stream": False},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# FR-2: POST /api/v1/chat — Streaming mode
# ---------------------------------------------------------------------------


class TestChatEndpointStreaming:
    """POST /chat with stream=True returns SSE events."""

    @pytest.mark.asyncio()
    async def test_streaming_returns_sse(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio()
    async def test_streaming_metadata_event_first(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?"},
        )
        body = response.text
        # First event should be metadata
        first_event_start = body.index("event: ")
        first_event_type = body[first_event_start:].split("\n")[0]
        assert first_event_type == "event: metadata"

    @pytest.mark.asyncio()
    async def test_streaming_has_token_events(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?"},
        )
        body = response.text
        assert "event: token" in body

    @pytest.mark.asyncio()
    async def test_streaming_has_done_event(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?"},
        )
        body = response.text
        assert "event: done" in body

    @pytest.mark.asyncio()
    async def test_streaming_metadata_contains_follow_up_info(self, client, mock_follow_up_handler):
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "sess-1", "query": "What is a transformer?"},
        )
        body = response.text
        # Parse metadata event
        for line in body.split("\n"):
            if line.startswith("data:") and "session_id" in line:
                data = json.loads(line[len("data:") :].strip())
                assert data["session_id"] == "sess-1"
                assert "is_follow_up" in data
                break
        else:
            pytest.fail("No metadata event with session_id found")


# ---------------------------------------------------------------------------
# FR-2: Error handling
# ---------------------------------------------------------------------------


class TestChatErrorHandling:
    """503 on RAG pipeline failure, error SSE event on streaming failure."""

    @pytest.mark.asyncio()
    async def test_json_mode_service_error(self, client, mock_follow_up_handler):
        mock_follow_up_handler.handle.side_effect = Exception("LLM is down")
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "query": "test", "stream": False},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio()
    async def test_streaming_error_event(self, client, mock_follow_up_handler):
        mock_follow_up_handler.handle_stream.side_effect = Exception("LLM is down")
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "query": "test", "stream": True},
        )
        body = response.text
        assert "event: error" in body


# ---------------------------------------------------------------------------
# FR-2: Graceful degradation (no Redis / no memory)
# ---------------------------------------------------------------------------


class TestChatGracefulDegradation:
    """Chat works when ConversationMemory is None."""

    @pytest.fixture()
    def app_no_memory(self, mock_follow_up_handler):
        from src.main import create_app

        app = create_app()
        from src.dependency import get_conversation_memory, get_follow_up_handler

        app.dependency_overrides[get_follow_up_handler] = lambda: mock_follow_up_handler
        app.dependency_overrides[get_conversation_memory] = lambda: None
        return app

    @pytest.mark.asyncio()
    async def test_chat_works_without_memory(self, app_no_memory, mock_follow_up_handler):
        transport = ASGITransport(app=app_no_memory)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/chat",
                json={"session_id": "s1", "query": "test", "stream": False},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio()
    async def test_session_history_no_memory(self, app_no_memory):
        transport = ASGITransport(app=app_no_memory)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/chat/sessions/s1/history")
        assert response.status_code == 200
        assert response.json()["messages"] == []

    @pytest.mark.asyncio()
    async def test_session_clear_no_memory(self, app_no_memory):
        transport = ASGITransport(app=app_no_memory)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.delete("/api/v1/chat/sessions/s1")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# FR-3: Session management
# ---------------------------------------------------------------------------


class TestSessionHistory:
    """GET /chat/sessions/{id}/history returns messages."""

    @pytest.mark.asyncio()
    async def test_returns_history(self, client, mock_memory):
        mock_memory.get_history.return_value = [
            ChatMessage(role="user", content="What is attention?"),
            ChatMessage(role="assistant", content="Self-attention is... [1]."),
        ]
        response = await client.get("/api/v1/chat/sessions/sess-1/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        assert data["session_id"] == "sess-1"

    @pytest.mark.asyncio()
    async def test_empty_session(self, client, mock_memory):
        mock_memory.get_history.return_value = []
        response = await client.get("/api/v1/chat/sessions/nonexistent/history")
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []


class TestSessionClear:
    """DELETE /chat/sessions/{id} clears session."""

    @pytest.mark.asyncio()
    async def test_clear_existing_session(self, client, mock_memory):
        mock_memory.clear_session.return_value = True
        response = await client.delete("/api/v1/chat/sessions/sess-1")
        assert response.status_code == 200
        data = response.json()
        assert data["cleared"] is True

    @pytest.mark.asyncio()
    async def test_clear_nonexistent_session(self, client, mock_memory):
        mock_memory.clear_session.return_value = False
        response = await client.delete("/api/v1/chat/sessions/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["cleared"] is False

    @pytest.mark.asyncio()
    async def test_clear_idempotent(self, client, mock_memory):
        """Clearing same session twice always returns 200."""
        mock_memory.clear_session.return_value = True
        r1 = await client.delete("/api/v1/chat/sessions/sess-1")
        mock_memory.clear_session.return_value = False
        r2 = await client.delete("/api/v1/chat/sessions/sess-1")
        assert r1.status_code == 200
        assert r2.status_code == 200
