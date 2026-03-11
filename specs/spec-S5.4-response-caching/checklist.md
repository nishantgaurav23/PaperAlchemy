# Checklist -- Spec S5.4: Response Caching (Redis)

## Phase 1: Setup & Dependencies
- [x] Verify S5.2 (RAG chain) is "done"
- [x] Create `src/services/cache/` directory
- [x] Create `src/services/cache/__init__.py`
- [x] Verify `redis[hiredis]` in pyproject.toml

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_cache_client.py`
- [x] Write failing tests for key generation (FR-1)
- [x] Write failing tests for cache lookup (FR-2)
- [x] Write failing tests for cache storage (FR-3)
- [x] Write failing tests for cache invalidation (FR-4)
- [x] Write failing tests for cache stats (FR-7)
- [x] Create `tests/unit/test_cache_factory.py`
- [x] Write failing tests for Redis factory (FR-5)
- [x] Write failing tests for CacheClient factory (FR-6)
- [x] Run tests -- expect failures (Red) ✓

## Phase 3: Implementation
- [x] Implement `src/services/cache/client.py` — CacheClient class
- [x] Implement `src/services/cache/factory.py` — make_redis_client, make_cache_client
- [x] Implement `src/services/cache/__init__.py` — exports
- [x] Run tests -- expect pass (Green) ✓ 25/25 passed
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add `CacheDep` to `src/dependency.py`
- [x] Run lint (`ruff check src/services/cache/`) — clean
- [x] Run full test suite — 589 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S5.4_caching.ipynb`
- [x] Update roadmap.md status to "done"
