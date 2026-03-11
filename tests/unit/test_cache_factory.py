"""Unit tests for cache factory functions (S5.4 — Response Caching).

TDD: Tests written FIRST, implementation follows.
All Redis interactions are mocked — no real Redis needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from src.services.cache.factory import make_cache_client, make_redis_client

# ---------------------------------------------------------------------------
# FR-5: Redis Client Factory
# ---------------------------------------------------------------------------


class TestMakeRedisClient:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Returns Redis client after successful ping."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True

        with patch("src.services.cache.factory.Redis", return_value=mock_redis_instance):
            client = await make_redis_client()

        assert client is not None
        mock_redis_instance.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_returns_none(self) -> None:
        """Returns None when Redis is unavailable."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.side_effect = ConnectionError("Connection refused")

        with patch("src.services.cache.factory.Redis", return_value=mock_redis_instance):
            client = await make_redis_client()

        assert client is None

    @pytest.mark.asyncio
    async def test_uses_settings(self) -> None:
        """Passes settings values to Redis constructor."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True

        with (
            patch("src.services.cache.factory.Redis", return_value=mock_redis_instance) as mock_redis_cls,
            patch("src.services.cache.factory.get_settings") as mock_settings,
        ):
            redis_cfg = mock_settings.return_value.redis
            redis_cfg.host = "redis-host"
            redis_cfg.port = 6380
            redis_cfg.password = "secret"
            redis_cfg.db = 1
            redis_cfg.decode_responses = True

            client = await make_redis_client()

        assert client is not None
        mock_redis_cls.assert_called_once_with(
            host="redis-host",
            port=6380,
            password="secret",
            db=1,
            decode_responses=True,
        )


# ---------------------------------------------------------------------------
# FR-6: CacheClient Factory
# ---------------------------------------------------------------------------


class TestMakeCacheClient:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Returns CacheClient when Redis is available."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True

        with (
            patch("src.services.cache.factory.Redis", return_value=mock_redis_instance),
            patch("src.services.cache.factory.get_settings") as mock_settings,
        ):
            mock_settings.return_value.redis.host = "localhost"
            mock_settings.return_value.redis.port = 6379
            mock_settings.return_value.redis.password = ""
            mock_settings.return_value.redis.db = 0
            mock_settings.return_value.redis.decode_responses = True
            mock_settings.return_value.redis.ttl_hours = 24

            client = await make_cache_client()

        assert client is not None

    @pytest.mark.asyncio
    async def test_failure_returns_none(self) -> None:
        """Returns None when Redis is unavailable."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.side_effect = ConnectionError("Connection refused")

        with (
            patch("src.services.cache.factory.Redis", return_value=mock_redis_instance),
            patch("src.services.cache.factory.get_settings") as mock_settings,
        ):
            mock_settings.return_value.redis.host = "localhost"
            mock_settings.return_value.redis.port = 6379
            mock_settings.return_value.redis.password = ""
            mock_settings.return_value.redis.db = 0
            mock_settings.return_value.redis.decode_responses = True
            mock_settings.return_value.redis.ttl_hours = 24

            client = await make_cache_client()

        assert client is None

    @pytest.mark.asyncio
    async def test_ttl_from_settings(self) -> None:
        """CacheClient TTL is derived from settings.redis.ttl_hours."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True

        with (
            patch("src.services.cache.factory.Redis", return_value=mock_redis_instance),
            patch("src.services.cache.factory.get_settings") as mock_settings,
        ):
            mock_settings.return_value.redis.host = "localhost"
            mock_settings.return_value.redis.port = 6379
            mock_settings.return_value.redis.password = ""
            mock_settings.return_value.redis.db = 0
            mock_settings.return_value.redis.decode_responses = True
            mock_settings.return_value.redis.ttl_hours = 12  # 12 hours

            client = await make_cache_client()

        assert client is not None
        assert client._ttl == 12 * 3600  # 43200 seconds
