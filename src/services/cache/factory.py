"""
Factory functions for creating Redis and CacheClient instances.

Why it's needed:
    Centralizes Redis connection setup (pooling, timeouts, ping test)
    and CacheClient creation. The lifespan manager calls these once
    at startup and tests can patch them to return mocks.

What it does:
    - make_redis_client(): Creates an async Redis client with connection
      pooling and validates connectivity with ping.
    - make_cache_client(): Wraps a Redis client in a CacheClient.

How it helps:
    - Single place for Redis connection and configuration
    - Ping test fails fasts if  Redis is down at startup
    - TTL configured from settings, not hardcoded
"""

import logging
from typing import Optional

from redis.asyncio import Redis

from src.config import Settings, get_settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)

async def make_redis_client(settings: Optional[Settings] = None) -> Optional[Redis]:
    """Create and verify an async Redis client.
    Returns None if Redis is unavailable (graceful degradation)
    """

    if settings is None:
        settings = get_settings()

    rs = settings.redis
    try:
        client = Redis(
            host=rs.host,
            port=rs.port,
            password=rs.password or None,
            db=rs.db,
            socket_timeout=rs.socket_timeout,
            socket_connect_timeout=rs.socket_connect_timeout,
            decode_responses=rs.decode_responses,
        )
        await client.ping()
        logger.info(f"Redis connected at {rs.host}:{rs.port}")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable (caching disabled): {e}")
        return None
    
async def make_cache_client(settings: Optional[Settings] = None) -> Optional[CacheClient]:
    """Create a CacheClient backed by Redis.

    Returns None if Redis in unavailable.
    """
    if settings is None:
        settings = get_settings()

    redis_client = await make_redis_client(settings)
    if redis_client is None:
        return None
    
    ttl_seconds = settings.redis.ttl_hours * 3600
    return CacheClient(redis_client=redis_client, ttl_seconds=ttl_seconds)