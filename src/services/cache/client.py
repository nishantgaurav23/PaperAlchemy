"""
Redis cache client for exact-match RAG response caching.

Why it's needed:
    Repeated identical queries (same question, model, parameters) should
    return instantly instead of re-running the full RAG pipeline (embed →
    search → generate). Redis provides sub-millisecond lookups for a
    150-400x speedup on cache hits.

What it does:
    - Generates a deterministic SHA256 cache key from query parameters
    - Stores serialized AskResponse objects with configurable TTL
    - Returns cached responses on exact match, None on miss
    - Gracefully handles Redis connection failures (log + continue)

How it helps:
    - Dramatically reduces latency for repeated queries
    - Reduces load on Ollama and OpenSearch
    - TTL ensures stale responses expire automatically
    - Graceful fallback means cache failures never break the pipeline
"""

import hashlib
import json
import logging
from typing import Optional

from redis.asyncio import Redis

from src.schemas.api.ask import AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """Exact-match cache for RAG responses backed by Redis."""

    def __init__(self, redis_client: Redis, ttl_seconds: int = 86400):
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _generate_cache_key(
        query: str,
        model: str,
        top_k: int,
        use_hybrid: bool,
        categories: Optional[list[str]] = None,
    ) -> str:
        """Generate a deterministic cache key from query parameters.

        Uses SHA256 hash of normalized parameters so keys are fixed-length
        and safe for Redis key names.
        """
        key_data = {
            "query": query.strip().lower(),
            "model": model,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "categories": sorted(categories) if categories else [],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"rag:ask:{key_hash}"

    async def find_cached_response(
        self,
        query: str,
        model: str,
        top_k: int,
        use_hybrid: bool,
        categories: Optional[list[str]] = None,
    ) -> Optional[AskResponse]:
        """Look up a cached response for the given query parameters.

        Returns None on cache miss or Redis error.
        """
        try:
            key = self._generate_cache_key(query, model, top_k, use_hybrid, categories)
            cached = await self._redis.get(key)
            if cached is None:
                return None
            data = json.loads(cached)
            logger.info(f"Cache HIT for key {key[:20]}...")
            return AskResponse(**data)
        except Exception as e:
            logger.warning(f"Cache lookup failed (graceful skip): {e}")
            return None

    async def store_response(
        self,
        query: str,
        model: str,
        top_k: int,
        use_hybrid: bool,
        response: AskResponse,
        categories: Optional[list[str]] = None,
    ) -> None:
        """Store a RAG response in cache with TTL."""
        try:
            key = self._generate_cache_key(query, model, top_k, use_hybrid, categories)
            data = response.model_dump_json()
            await self._redis.set(key, data, ex=self._ttl)
            logger.info(f"Cache STORE for key {key[:20]}... (TTL={self._ttl}s)")
        except Exception as e:
            logger.warning(f"Cache store failed (graceful skip): {e}")
