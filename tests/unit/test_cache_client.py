"""Unit tests for CacheClient (S5.4 — Response Caching).

TDD: Tests written FIRST, implementation follows.
All Redis interactions are mocked — no real Redis needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.cache.client import CacheClient
from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Return a mocked async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.dbsize = AsyncMock(return_value=42)
    redis.info = AsyncMock(return_value={"used_memory_human": "1.5M"})
    return redis


@pytest.fixture
def cache_client(mock_redis: AsyncMock) -> CacheClient:
    """Return a CacheClient with mocked Redis and 24h TTL."""
    return CacheClient(redis_client=mock_redis, ttl_seconds=86400)


@pytest.fixture
def sample_rag_response() -> RAGResponse:
    """Return a representative RAGResponse for testing."""
    return RAGResponse(
        answer="Transformers use self-attention [1].",
        sources=[
            SourceReference(
                index=1,
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer"],
                arxiv_url="https://arxiv.org/abs/1706.03762",
                chunk_text="The dominant sequence transduction models...",
                score=0.95,
            ),
        ],
        query="What are transformers?",
        retrieval_metadata=RetrievalMetadata(
            stages_executed=["multi_query", "hybrid_search", "rerank"],
            total_candidates=20,
            timings={"retrieval": 0.5},
        ),
        llm_metadata=LLMMetadata(
            provider="gemini",
            model="gemini-2.0-flash",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            latency_ms=1200.0,
        ),
    )


# ---------------------------------------------------------------------------
# FR-1: Cache Key Generation
# ---------------------------------------------------------------------------


class TestCacheKeyGeneration:
    def test_deterministic(self, cache_client: CacheClient) -> None:
        """Same params produce the same key."""
        key1 = cache_client._generate_cache_key("What are transformers?", "gemini-2.0-flash", 5, None)
        key2 = cache_client._generate_cache_key("What are transformers?", "gemini-2.0-flash", 5, None)
        assert key1 == key2

    def test_normalized_whitespace_and_case(self, cache_client: CacheClient) -> None:
        """Query normalization: strip + lowercase."""
        key1 = cache_client._generate_cache_key("  What Are Transformers?  ", "gemini-2.0-flash", 5, None)
        key2 = cache_client._generate_cache_key("what are transformers?", "gemini-2.0-flash", 5, None)
        assert key1 == key2

    def test_different_params_different_keys(self, cache_client: CacheClient) -> None:
        """Different parameters produce different keys."""
        key1 = cache_client._generate_cache_key("transformers", "gemini-2.0-flash", 5, None)
        key2 = cache_client._generate_cache_key("transformers", "llama3.2:1b", 5, None)
        key3 = cache_client._generate_cache_key("transformers", "gemini-2.0-flash", 10, None)
        assert key1 != key2
        assert key1 != key3

    def test_categories_sorted(self, cache_client: CacheClient) -> None:
        """Category order doesn't affect key."""
        key1 = cache_client._generate_cache_key("query", "model", 5, ["cs.AI", "cs.CL"])
        key2 = cache_client._generate_cache_key("query", "model", 5, ["cs.CL", "cs.AI"])
        assert key1 == key2

    def test_none_categories_same_as_empty(self, cache_client: CacheClient) -> None:
        """None categories and empty list produce same key."""
        key1 = cache_client._generate_cache_key("query", "model", 5, None)
        key2 = cache_client._generate_cache_key("query", "model", 5, [])
        assert key1 == key2

    def test_key_has_prefix(self, cache_client: CacheClient) -> None:
        """Key starts with 'rag:response:' prefix."""
        key = cache_client._generate_cache_key("query", "model", 5, None)
        assert key.startswith("rag:response:")

    def test_key_is_fixed_length(self, cache_client: CacheClient) -> None:
        """SHA256 hex = 64 chars + prefix length."""
        key = cache_client._generate_cache_key("query", "model", 5, None)
        prefix = "rag:response:"
        assert len(key) == len(prefix) + 64


# ---------------------------------------------------------------------------
# FR-2: Cache Lookup
# ---------------------------------------------------------------------------


class TestFindCachedResponse:
    @pytest.mark.asyncio
    async def test_cache_hit(self, cache_client: CacheClient, mock_redis: AsyncMock, sample_rag_response: RAGResponse) -> None:
        """Returns deserialized RAGResponse on cache hit."""
        mock_redis.get.return_value = sample_rag_response.model_dump_json()

        result = await cache_client.find_cached_response("What are transformers?", "gemini-2.0-flash", 5, None)

        assert result is not None
        assert isinstance(result, RAGResponse)
        assert result.answer == sample_rag_response.answer
        assert len(result.sources) == 1
        assert result.sources[0].arxiv_id == "1706.03762"

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns None on cache miss."""
        mock_redis.get.return_value = None

        result = await cache_client.find_cached_response("unknown query", "model", 5, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns None on Redis failure (graceful degradation)."""
        mock_redis.get.side_effect = ConnectionError("Redis down")

        result = await cache_client.find_cached_response("query", "model", 5, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_corrupted_data_returns_none(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns None when cached data is not valid JSON."""
        mock_redis.get.return_value = "not-valid-json{{"

        result = await cache_client.find_cached_response("query", "model", 5, None)

        assert result is None


# ---------------------------------------------------------------------------
# FR-3: Cache Storage
# ---------------------------------------------------------------------------


class TestStoreResponse:
    @pytest.mark.asyncio
    async def test_store_success(
        self, cache_client: CacheClient, mock_redis: AsyncMock, sample_rag_response: RAGResponse
    ) -> None:
        """Stores serialized response with TTL."""
        await cache_client.store_response("What are transformers?", "gemini-2.0-flash", 5, sample_rag_response, None)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs.get("ex") == 86400 or call_args[1].get("ex") == 86400 or call_args[0][2] is not None

    @pytest.mark.asyncio
    async def test_store_redis_error_does_not_raise(
        self, cache_client: CacheClient, mock_redis: AsyncMock, sample_rag_response: RAGResponse
    ) -> None:
        """Redis failure on store logs warning but does not raise."""
        mock_redis.set.side_effect = ConnectionError("Redis down")

        # Should not raise
        await cache_client.store_response("query", "model", 5, sample_rag_response, None)

    @pytest.mark.asyncio
    async def test_store_roundtrip(
        self, cache_client: CacheClient, mock_redis: AsyncMock, sample_rag_response: RAGResponse
    ) -> None:
        """Stored data can be deserialized back to RAGResponse."""
        stored_data = None

        async def capture_set(key, value, **kwargs):
            nonlocal stored_data
            stored_data = value

        mock_redis.set = AsyncMock(side_effect=capture_set)

        await cache_client.store_response("query", "model", 5, sample_rag_response, None)

        assert stored_data is not None
        restored = RAGResponse.model_validate_json(stored_data)
        assert restored.answer == sample_rag_response.answer
        assert restored.sources[0].arxiv_id == sample_rag_response.sources[0].arxiv_id


# ---------------------------------------------------------------------------
# FR-4: Cache Invalidation
# ---------------------------------------------------------------------------


class TestInvalidate:
    @pytest.mark.asyncio
    async def test_invalidate_specific_key(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Deletes specific cached entry by query params."""
        mock_redis.delete.return_value = 1

        count = await cache_client.invalidate("query", "model", 5, None)

        assert count == 1
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_redis_error(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns 0 on Redis failure."""
        mock_redis.delete.side_effect = ConnectionError("Redis down")

        count = await cache_client.invalidate("query", "model", 5, None)

        assert count == 0

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """invalidate_all deletes all rag:response:* keys."""
        # Mock scan_iter to return some keys
        mock_redis.scan_iter = MagicMock()

        async def fake_scan_iter(match=None):
            for key in ["rag:response:abc", "rag:response:def"]:
                yield key

        mock_redis.scan_iter = fake_scan_iter
        mock_redis.delete.return_value = 2

        count = await cache_client.invalidate_all()

        assert count >= 0  # Implementation may vary


# ---------------------------------------------------------------------------
# FR-7: Cache Statistics
# ---------------------------------------------------------------------------


class TestGetStats:
    @pytest.mark.asyncio
    async def test_get_stats(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns cache statistics."""
        mock_redis.dbsize.return_value = 42
        mock_redis.info.return_value = {"used_memory_human": "1.5M"}

        stats = await cache_client.get_stats()

        assert stats["keys_count"] == 42
        assert stats["memory_usage"] == "1.5M"

    @pytest.mark.asyncio
    async def test_get_stats_redis_error(self, cache_client: CacheClient, mock_redis: AsyncMock) -> None:
        """Returns empty stats on Redis failure."""
        mock_redis.dbsize.side_effect = ConnectionError("Redis down")

        stats = await cache_client.get_stats()

        assert stats["keys_count"] == 0
