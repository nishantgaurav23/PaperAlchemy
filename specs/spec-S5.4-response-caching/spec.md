# Spec S5.4 -- Response Caching (Redis)

## Overview
Redis-backed exact-match caching for RAG responses. When the same query (with same parameters) is repeated, return the cached response instantly instead of re-running the full RAG pipeline (retrieve → prompt → generate). This provides ~150-400x speedup on cache hits while maintaining graceful degradation when Redis is unavailable.

## Dependencies
- **S5.2** (RAG chain) — provides `RAGResponse` model and `RAGChain` to wrap with caching

## Target Location
- `src/services/cache/client.py` — CacheClient class
- `src/services/cache/factory.py` — Redis + CacheClient creation
- `src/services/cache/__init__.py` — Public exports

## Functional Requirements

### FR-1: Cache Key Generation
- **What**: Generate deterministic SHA256 cache keys from query parameters
- **Inputs**: `query: str`, `model: str`, `top_k: int`, `categories: list[str] | None`
- **Outputs**: `str` — Redis key in format `rag:response:{sha256_hex}`
- **Edge cases**: Query normalization (strip + lowercase), sorted categories, None categories → empty list
- **Key composition**: JSON-serialized dict of normalized params → SHA256 → prefixed key

### FR-2: Cache Lookup (find_cached_response)
- **What**: Look up a cached RAGResponse for given query parameters
- **Inputs**: query, model, top_k, categories
- **Outputs**: `RAGResponse | None` — deserialized response on hit, None on miss
- **Edge cases**: Redis connection failure → return None (graceful), corrupted data → return None + log warning

### FR-3: Cache Storage (store_response)
- **What**: Store a RAGResponse in Redis with configurable TTL
- **Inputs**: query, model, top_k, categories, response (RAGResponse), ttl from settings
- **Outputs**: None (fire-and-forget, log on failure)
- **Edge cases**: Redis connection failure → log warning + continue, never raise

### FR-4: Cache Invalidation (invalidate)
- **What**: Delete cached entries matching a pattern or specific key
- **Inputs**: query params (specific key) or pattern string (e.g., `rag:response:*`)
- **Outputs**: `int` — number of keys deleted
- **Edge cases**: Redis failure → return 0 + log warning

### FR-5: Redis Client Factory
- **What**: Create async Redis client with connection pooling and connectivity check
- **Inputs**: Settings (host, port, password, db, TTL)
- **Outputs**: `Redis | None` — None if Redis unavailable (graceful degradation)
- **Edge cases**: Connection refused, auth failure, timeout → return None + log warning

### FR-6: CacheClient Factory
- **What**: Create CacheClient wrapping a Redis client
- **Inputs**: Settings
- **Outputs**: `CacheClient | None` — None if Redis unavailable
- **Edge cases**: Propagates None from Redis factory

### FR-7: Cache Statistics (get_stats)
- **What**: Return basic cache stats (hit count, key count)
- **Inputs**: None
- **Outputs**: `CacheStats` — dict with keys_count, memory_usage

## Tangible Outcomes
- [ ] `CacheClient` class with `find_cached_response`, `store_response`, `invalidate`, `get_stats` methods
- [ ] `make_redis_client()` factory returns `Redis | None` with ping check
- [ ] `make_cache_client()` factory returns `CacheClient | None`
- [ ] SHA256 key generation is deterministic (same params → same key)
- [ ] TTL is configurable via `REDIS__TTL_HOURS` env var (default 24h)
- [ ] All Redis failures are caught and logged — never propagate exceptions
- [ ] Cache stores/retrieves `RAGResponse` via Pydantic serialization
- [ ] `CacheDep` Annotated type available in `src/dependency.py`
- [ ] All tests pass with mocked Redis (no real Redis needed)

## Test-Driven Requirements

### Tests to Write First
1. `test_generate_cache_key_deterministic`: Same params produce same key
2. `test_generate_cache_key_normalized`: Whitespace/case variations produce same key
3. `test_generate_cache_key_different_params`: Different params produce different keys
4. `test_generate_cache_key_categories_sorted`: Category order doesn't matter
5. `test_find_cached_response_hit`: Returns deserialized RAGResponse on cache hit
6. `test_find_cached_response_miss`: Returns None on cache miss
7. `test_find_cached_response_redis_error`: Returns None on Redis failure
8. `test_store_response_success`: Stores serialized response with TTL
9. `test_store_response_redis_error`: Logs warning, does not raise
10. `test_invalidate_specific_key`: Deletes specific cached entry
11. `test_invalidate_redis_error`: Returns 0 on Redis failure
12. `test_get_stats`: Returns key count and memory info
13. `test_make_redis_client_success`: Returns Redis client after ping
14. `test_make_redis_client_failure`: Returns None when Redis unavailable
15. `test_make_cache_client_success`: Returns CacheClient when Redis available
16. `test_make_cache_client_failure`: Returns None when Redis unavailable

### Mocking Strategy
- Mock `redis.asyncio.Redis` for all unit tests — no real Redis connection
- Use `AsyncMock` for async Redis methods (get, set, delete, ping, info, dbsize)
- Mock `Settings` with test values for factory tests

### Coverage
- All public methods tested
- All Redis failure paths tested (graceful degradation)
- Key generation edge cases tested
- Serialization/deserialization round-trip tested
