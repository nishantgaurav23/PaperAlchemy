# Spec S1.2 — Environment Configuration

## Overview
Pydantic settings with environment variable validation for all PaperAlchemy services. Each service gets its own nested settings class with `env_prefix` (double underscore convention), composed into a single root `Settings` object. Supports `.env` files, env var overrides, and type-safe validation.

## Dependencies
- S1.1 (Dependency declaration) — `pydantic-settings` must be declared

## Target Location
- `src/config.py` — all settings classes
- `.env.example` — documented example with all supported variables
- `.env.test` — test-safe defaults (no real credentials)

## Functional Requirements

### FR-1: Nested Settings Classes
- **What**: Each external service gets its own `BaseSettings` subclass with `env_prefix`
- **Sub-settings required** (11 total):
  1. `PostgresSettings` — env_prefix=`POSTGRES__` — host, port, user, password, db + async `url` property
  2. `OpenSearchSettings` — env_prefix=`OPENSEARCH__` — host, port, index_name, chunk_index_suffix, vector_dimension, rrf_pipeline_name
  3. `OllamaSettings` — env_prefix=`OLLAMA__` — host, port, default_model, default_timeout, default_temperature, default_top_p + `url` property
  4. `GeminiSettings` — env_prefix=`GEMINI__` — api_key, model (default: `gemini-2.0-flash`), temperature, max_output_tokens, timeout
  5. `RedisSettings` — env_prefix=`REDIS__` — host, port, password, db, ttl_hours, decode_responses
  6. `JinaSettings` — env_prefix=`JINA__` — api_key, model (default: `jina-embeddings-v3`), dimensions (1024), batch_size (100), timeout
  7. `RerankerSettings` — env_prefix=`RERANKER__` — model (default: `cross-encoder/ms-marco-MiniLM-L-12-v2`), top_k (5), device (`cpu`)
  8. `LangfuseSettings` — env_prefix=`LANGFUSE__` — public_key, secret_key, host, enabled (bool)
  9. `ArxivSettings` — env_prefix=`ARXIV__` — base_url, rate_limit_delay (3.0), max_results, category, timeout, max_retries
  10. `ChunkingSettings` — env_prefix=`CHUNKING__` — chunk_size (600), overlap_size (100), min_chunk_size (100), section_based (True)
  11. `AppSettings` — env_prefix=`APP__` — debug (bool), log_level, title, version
- **Edge cases**: Missing env vars use defaults; empty strings for API keys (not required at startup)

### FR-2: Root Settings Composition
- **What**: A single `Settings` class composes all 11 sub-settings
- **Behavior**: Reads from `.env` file, env vars override, `extra="ignore"`
- **Properties**: All sub-settings accessible as `settings.postgres`, `settings.gemini`, etc.

### FR-3: Settings Factory
- **What**: `get_settings()` function returns a cached `Settings` instance
- **Caching**: Use `@lru_cache` for singleton behavior
- **Override**: Support `Settings` constructor kwargs for testing

### FR-4: URL Properties
- **PostgresSettings.url**: Returns `postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}` (async driver)
- **PostgresSettings.sync_url**: Returns `postgresql://{user}:{password}@{host}:{port}/{db}` (for Alembic)
- **OllamaSettings.url**: Returns `http://{host}:{port}`
- **RedisSettings.url**: Returns `redis://:{password}@{host}:{port}/{db}` (omit password segment if empty)

### FR-5: .env.example File
- **What**: Documented example with all supported environment variables
- **Format**: Grouped by service, with comments explaining each section
- **Includes**: Gemini, Reranker, and Jina sections (not present in old .env.example)

### FR-6: .env.test File
- **What**: Test-safe defaults that don't connect to real services
- **Values**: Use `localhost`, dummy API keys like `test-key`, `APP__DEBUG=true`

## Tangible Outcomes
- [ ] `src/config.py` exists with all 11 sub-settings + root Settings class
- [ ] `get_settings()` returns a valid Settings instance
- [ ] All env_prefix conventions work (e.g., `POSTGRES__HOST=x` sets `settings.postgres.host`)
- [ ] `settings.postgres.url` returns async connection string with `asyncpg` driver
- [ ] `settings.gemini` sub-settings exist (new vs reference)
- [ ] `settings.reranker` sub-settings exist (new vs reference)
- [ ] `settings.jina` sub-settings exist (extracted from root-level jina_api_key)
- [ ] `.env.example` documents all supported variables
- [ ] `.env.test` provides test-safe defaults
- [ ] No hardcoded secrets in source code
- [ ] All settings have sensible defaults (app runs without .env in dev)

## Test-Driven Requirements

### Tests to Write First
1. `test_default_settings_creation`: Settings() works with no env vars, all defaults applied
2. `test_postgres_async_url`: PostgresSettings.url uses `postgresql+asyncpg://` prefix
3. `test_postgres_sync_url`: PostgresSettings.sync_url uses `postgresql://` prefix
4. `test_ollama_url`: OllamaSettings.url returns correct http URL
5. `test_redis_url_with_password`: RedisSettings.url includes password when set
6. `test_redis_url_without_password`: RedisSettings.url omits password when empty
7. `test_env_prefix_override`: Setting `POSTGRES__HOST=custom` overrides default
8. `test_gemini_settings_defaults`: GeminiSettings has correct defaults
9. `test_reranker_settings_defaults`: RerankerSettings has correct defaults
10. `test_jina_settings_defaults`: JinaSettings has correct defaults
11. `test_chunking_settings_defaults`: ChunkingSettings has correct defaults
12. `test_get_settings_returns_same_instance`: get_settings() is cached
13. `test_settings_composition`: Root Settings has all 11 sub-settings accessible

### Mocking Strategy
- Use `monkeypatch.setenv()` to test env var overrides
- Use `monkeypatch.delenv()` to test missing env vars
- Clear `lru_cache` between tests with `get_settings.cache_clear()`
- No external services needed — this is pure configuration

### Coverage
- All settings classes tested for defaults
- URL properties tested for correct format
- Env var overrides tested
- Edge cases: empty passwords, missing API keys
