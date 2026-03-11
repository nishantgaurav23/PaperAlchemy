"""Tests for JinaEmbeddingsClient (FR-2, FR-3, FR-4, FR-6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.exceptions import EmbeddingServiceError
from src.services.embeddings.client import JinaEmbeddingsClient


def _make_jina_response(num_embeddings: int, dim: int = 1024) -> dict:
    """Helper to build a realistic Jina API response."""
    return {
        "model": "jina-embeddings-v3",
        "data": [{"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * dim} for i in range(num_embeddings)],
        "usage": {"total_tokens": num_embeddings * 10},
    }


@pytest.fixture
def client():
    return JinaEmbeddingsClient(api_key="test-key", model="jina-embeddings-v3", dimensions=1024, timeout=30)


class TestEmbedPassages:
    """FR-2: Batch embedding of passages."""

    @pytest.mark.asyncio
    async def test_single_batch(self, client):
        """Embed 3 texts in one batch — verify request payload and returned vectors."""
        texts = ["hello world", "foo bar", "test text"]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_jina_response(3)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await client.embed_passages(texts, batch_size=100)

        assert len(result) == 3
        assert len(result[0]) == 1024
        mock_post.assert_called_once()
        # Verify task is retrieval.passage
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["task"] == "retrieval.passage"

    @pytest.mark.asyncio
    async def test_multi_batch(self, client):
        """250 texts with batch_size=100 → 3 API calls."""
        texts = [f"text {i}" for i in range(250)]

        def make_response(batch_size):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _make_jina_response(batch_size)
            resp.raise_for_status = MagicMock()
            return resp

        # 3 batches: 100, 100, 50
        responses = [make_response(100), make_response(100), make_response(50)]
        with patch.object(client._client, "post", new_callable=AsyncMock, side_effect=responses) as mock_post:
            result = await client.embed_passages(texts, batch_size=100)

        assert len(result) == 250
        assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_list(self, client):
        """Empty input returns [] without making any API call."""
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            result = await client.embed_passages([])

        assert result == []
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_text(self, client):
        """Single text works correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_jina_response(1)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.embed_passages(["single text"])

        assert len(result) == 1
        assert len(result[0]) == 1024


class TestEmbedQuery:
    """FR-3: Single query embedding."""

    @pytest.mark.asyncio
    async def test_success(self, client):
        """Embed a query — verify retrieval.query task and 1024-dim result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_jina_response(1)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await client.embed_query("attention mechanism")

        assert len(result) == 1024
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["task"] == "retrieval.query"

    @pytest.mark.asyncio
    async def test_empty_string_raises(self, client):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await client.embed_query("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self, client):
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await client.embed_query("   ")


class TestErrorHandling:
    """FR-6: HTTP errors wrapped in EmbeddingServiceError."""

    @pytest.mark.asyncio
    async def test_http_500_error(self, client):
        """Server error → EmbeddingServiceError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        with (
            patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response),
            pytest.raises(EmbeddingServiceError),
        ):
            await client.embed_query("test")

    @pytest.mark.asyncio
    async def test_http_401_auth_error(self, client):
        """Auth error → EmbeddingServiceError with auth message."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        with (
            patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response),
            pytest.raises(EmbeddingServiceError, match="[Aa]uth"),
        ):
            await client.embed_query("test")

    @pytest.mark.asyncio
    async def test_http_429_rate_limit(self, client):
        """Rate limit → EmbeddingServiceError with rate limit message."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=mock_response
        )

        with (
            patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response),
            pytest.raises(EmbeddingServiceError, match="[Rr]ate"),
        ):
            await client.embed_passages(["test"])

    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Timeout → EmbeddingServiceError."""
        with (
            patch.object(client._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")),
            pytest.raises(EmbeddingServiceError, match="[Tt]imeout"),
        ):
            await client.embed_query("test")

    @pytest.mark.asyncio
    async def test_connection_error(self, client):
        """Connection error → EmbeddingServiceError."""
        with (
            patch.object(client._client, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("connection refused")),
            pytest.raises(EmbeddingServiceError),
        ):
            await client.embed_query("test")


class TestContextManager:
    """FR-4: Async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """async with properly opens and closes client."""
        async with JinaEmbeddingsClient(api_key="test-key") as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        """Multiple close() calls are safe."""
        client = JinaEmbeddingsClient(api_key="test-key")
        await client.close()
        await client.close()  # Should not raise
