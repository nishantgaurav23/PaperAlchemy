"""Redis cache client for exact-match RAG response caching (S5.4).

Generates deterministic SHA256 keys from query parameters, stores serialized
RAGResponse objects with configurable TTL, and returns cached responses on
exact match. All Redis failures are caught and logged — never propagated.
"""

from __future__ import annotations

import hashlib
import json
import logging

from redis.asyncio import Redis
from src.services.rag.models import RAGResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Exact-match cache for RAG responses backed by Redis."""

    def __init__(self, redis_client: Redis, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _generate_cache_key(
        query: str,
        model: str,
        top_k: int,
        categories: list[str] | None = None,
    ) -> str:
        """Generate a deterministic SHA256 cache key from query parameters."""
        key_data = {
            "query": query.strip().lower(),
            "model": model,
            "top_k": top_k,
            "categories": sorted(categories) if categories else [],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"rag:response:{key_hash}"

    async def find_cached_response(
        self,
        query: str,
        model: str,
        top_k: int,
        categories: list[str] | None = None,
    ) -> RAGResponse | None:
        """Look up a cached RAGResponse. Returns None on miss or error."""
        try:
            key = self._generate_cache_key(query, model, top_k, categories)
            cached = await self._redis.get(key)
            if cached is None:
                return None
            logger.info("Cache HIT for key %s...", key[:25])
            return RAGResponse.model_validate_json(cached)
        except Exception:
            logger.warning("Cache lookup failed (graceful skip)", exc_info=True)
            return None

    async def store_response(
        self,
        query: str,
        model: str,
        top_k: int,
        response: RAGResponse,
        categories: list[str] | None = None,
    ) -> None:
        """Store a RAGResponse in cache with TTL. Never raises."""
        try:
            key = self._generate_cache_key(query, model, top_k, categories)
            data = response.model_dump_json()
            await self._redis.set(key, data, ex=self._ttl)
            logger.info("Cache STORE for key %s... (TTL=%ds)", key[:25], self._ttl)
        except Exception:
            logger.warning("Cache store failed (graceful skip)", exc_info=True)

    async def invalidate(
        self,
        query: str,
        model: str,
        top_k: int,
        categories: list[str] | None = None,
    ) -> int:
        """Delete a specific cached entry. Returns number of keys deleted."""
        try:
            key = self._generate_cache_key(query, model, top_k, categories)
            return await self._redis.delete(key)
        except Exception:
            logger.warning("Cache invalidation failed (graceful skip)", exc_info=True)
            return 0

    async def invalidate_all(self) -> int:
        """Delete all rag:response:* keys. Returns number deleted."""
        try:
            keys = []
            async for key in self._redis.scan_iter(match="rag:response:*"):
                keys.append(key)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception:
            logger.warning("Cache invalidate_all failed (graceful skip)", exc_info=True)
            return 0

    async def get_stats(self) -> dict:
        """Return basic cache statistics."""
        try:
            keys_count = await self._redis.dbsize()
            info = await self._redis.info("memory")
            return {
                "keys_count": keys_count,
                "memory_usage": info.get("used_memory_human", "unknown"),
            }
        except Exception:
            logger.warning("Cache get_stats failed (graceful skip)", exc_info=True)
            return {"keys_count": 0, "memory_usage": "unknown"}
