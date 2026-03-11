# Checklist — Spec S1.2: Environment Configuration

## Phase 1: Setup & Dependencies
- [x] Verify S1.1 is "done" (pydantic-settings in pyproject.toml)
- [x] Create `src/` directory if needed
- [x] Create `src/__init__.py` if needed

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_config.py`
- [x] Write test_default_settings_creation
- [x] Write test_postgres_async_url
- [x] Write test_postgres_sync_url
- [x] Write test_ollama_url
- [x] Write test_redis_url_with_password
- [x] Write test_redis_url_without_password
- [x] Write test_env_prefix_override
- [x] Write test_gemini_settings_defaults
- [x] Write test_reranker_settings_defaults
- [x] Write test_jina_settings_defaults
- [x] Write test_chunking_settings_defaults
- [x] Write test_get_settings_returns_same_instance
- [x] Write test_settings_composition
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Create `src/config.py` with all 11 sub-settings classes
- [x] Implement PostgresSettings (url + sync_url properties)
- [x] Implement OpenSearchSettings
- [x] Implement OllamaSettings (url property)
- [x] Implement GeminiSettings (NEW)
- [x] Implement RedisSettings (url property)
- [x] Implement JinaSettings (NEW — extracted from root jina_api_key)
- [x] Implement RerankerSettings (NEW)
- [x] Implement LangfuseSettings
- [x] Implement ArxivSettings
- [x] Implement ChunkingSettings
- [x] Implement AppSettings
- [x] Implement root Settings class (compose all 11)
- [x] Implement get_settings() with lru_cache
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Update `.env.example` with all new variables (Gemini, Reranker, Jina)
- [x] Create `.env.test` with test-safe defaults
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S1.2_config.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
