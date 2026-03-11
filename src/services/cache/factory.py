"""Factory functions for creating Redis and CacheClient instances (S5.4).

Centralizes Redis connection setup and CacheClient creation.
Returns None when Redis is unavailable (graceful degradation).
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis
from src.config import get_settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)


async def make_redis_client() -> Redis | None:
    """Create and verify an async Redis client.

    Returns None if Redis is unavailable (graceful degradation).
    """
    settings = get_settings()
    rs = settings.redis
    try:
        client = Redis(
            host=rs.host,
            port=rs.port,
            password=rs.password or None,
            db=rs.db,
            decode_responses=rs.decode_responses,
        )
        await client.ping()
        logger.info("Redis connected at %s:%d", rs.host, rs.port)
        return client
    except Exception:
        logger.warning("Redis unavailable (caching disabled)", exc_info=True)
        return None


async def make_cache_client() -> CacheClient | None:
    """Create a CacheClient backed by Redis.

    Returns None if Redis is unavailable.
    """
    redis_client = await make_redis_client()
    if redis_client is None:
        return None

    settings = get_settings()
    ttl_seconds = settings.redis.ttl_hours * 3600
    return CacheClient(redis_client=redis_client, ttl_seconds=ttl_seconds)
