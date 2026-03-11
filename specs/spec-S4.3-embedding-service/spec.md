# Spec S4.3 -- Embedding Service (Jina AI)

## Overview
Async client for the Jina AI Embeddings v3 API that converts text into 1024-dimensional vectors. Supports two embedding tasks: `retrieval.passage` for document chunks (used at indexing time) and `retrieval.query` for search queries (used at query time). This asymmetric encoding improves retrieval accuracy. Processes passages in configurable batches (default 100) to maximize throughput.

## Dependencies
- **S1.2** (Environment Config): `JinaSettings` in `src/config.py` — provides `api_key`, `model`, `dimensions`, `batch_size`, `timeout`

## Target Location
- `src/services/embeddings/__init__.py` — module exports
- `src/services/embeddings/client.py` — `JinaEmbeddingsClient` class
- `src/services/embeddings/factory.py` — `make_embeddings_client()` factory
- `src/schemas/embeddings.py` — Pydantic request/response models

## Functional Requirements

### FR-1: Pydantic Schemas for Jina API
- **What**: Request and response models for the Jina Embeddings API
- **Inputs**: `JinaEmbeddingRequest(model, task, dimensions, input)`, `JinaEmbeddingResponse(data, usage)`
- **Outputs**: Validated Pydantic models
- **Edge cases**: Empty input list, missing fields in response

### FR-2: Embed Passages (Batch)
- **What**: Convert a list of text passages into embedding vectors, processing in batches
- **Inputs**: `texts: list[str]`, optional `batch_size: int` (default from settings)
- **Outputs**: `list[list[float]]` — one 1024-dim vector per input text, order preserved
- **Edge cases**: Empty list returns `[]`, single text works, large list batched correctly, API error raises `EmbeddingError`

### FR-3: Embed Query (Single)
- **What**: Convert a single search query into an embedding vector
- **Inputs**: `query: str`
- **Outputs**: `list[float]` — single 1024-dim vector
- **Edge cases**: Empty string raises `ValueError`, API error raises `EmbeddingError`

### FR-4: Async Context Manager
- **What**: Support `async with` for proper HTTP client lifecycle management
- **Inputs**: N/A
- **Outputs**: Client properly initializes and closes httpx.AsyncClient
- **Edge cases**: Multiple close() calls are safe (idempotent)

### FR-5: Factory Function
- **What**: Create `JinaEmbeddingsClient` from application settings
- **Inputs**: Optional `Settings` (defaults to `get_settings()`)
- **Outputs**: Configured `JinaEmbeddingsClient` instance
- **Edge cases**: Missing API key raises `ConfigurationError`

### FR-6: Error Handling
- **What**: Wrap HTTP errors in domain-specific `EmbeddingError` exception
- **Inputs**: Various API failure modes (auth, rate limit, timeout, server error)
- **Outputs**: `EmbeddingError` with descriptive message and original cause
- **Edge cases**: Network timeout, 401 unauthorized, 429 rate limited, 500 server error

## Tangible Outcomes
- [ ] `src/services/embeddings/client.py` exists with `JinaEmbeddingsClient` class
- [ ] `src/services/embeddings/factory.py` exists with `make_embeddings_client()`
- [ ] `src/schemas/embeddings.py` exists with `JinaEmbeddingRequest` and `JinaEmbeddingResponse`
- [ ] `embed_passages()` returns `list[list[float]]` with correct dimensions (1024)
- [ ] `embed_passages()` batches correctly (e.g., 250 texts with batch_size=100 → 3 API calls)
- [ ] `embed_query()` returns `list[float]` with 1024 dimensions
- [ ] Empty input to `embed_passages()` returns `[]` without API call
- [ ] API errors are wrapped in `EmbeddingError`
- [ ] Async context manager properly opens and closes httpx client
- [ ] Factory reads `settings.jina` for configuration
- [ ] All external HTTP calls are mocked in tests

## Test-Driven Requirements

### Tests to Write First
1. `test_embed_passages_single_batch`: Mock API, verify correct request payload and returned vectors
2. `test_embed_passages_multi_batch`: 250 texts, batch_size=100, verify 3 API calls made
3. `test_embed_passages_empty_list`: Returns `[]` without making API call
4. `test_embed_query_success`: Mock API, verify `retrieval.query` task and 1024-dim result
5. `test_embed_query_empty_string`: Raises `ValueError`
6. `test_api_error_wraps_in_embedding_error`: HTTP 500 → `EmbeddingError`
7. `test_auth_error`: HTTP 401 → `EmbeddingError` with auth message
8. `test_rate_limit_error`: HTTP 429 → `EmbeddingError` with rate limit message
9. `test_timeout_error`: httpx.TimeoutException → `EmbeddingError`
10. `test_context_manager`: `async with` properly opens and closes client
11. `test_factory_creates_client`: Factory uses settings.jina config
12. `test_factory_missing_api_key`: Empty API key raises `ConfigurationError`
13. `test_request_schema_validation`: JinaEmbeddingRequest validates fields
14. `test_response_schema_parsing`: JinaEmbeddingResponse parses API response

### Mocking Strategy
- Mock `httpx.AsyncClient.post` for all API calls
- Use `unittest.mock.AsyncMock` for async mocking
- Provide realistic Jina API response fixtures (matching actual API format)
- Never call the real Jina API in tests

### Coverage
- All public methods tested
- All error paths tested (401, 429, 500, timeout)
- Batch boundary conditions (0, 1, exactly batch_size, batch_size+1)
- Schema validation for request and response
