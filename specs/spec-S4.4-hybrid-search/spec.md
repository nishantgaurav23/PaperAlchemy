# Spec S4.4 -- Hybrid Search (BM25 + KNN + RRF)

## Overview
Unified search endpoint that combines BM25 keyword search with KNN vector search using Reciprocal Rank Fusion (RRF). The endpoint orchestrates the OpenSearch client (S4.1) and Jina embedding service (S4.3) to deliver hybrid search results. When embeddings fail, the system gracefully degrades to BM25-only search.

## Dependencies
- **S4.1** — OpenSearch client (`src/services/opensearch/`) — `search_unified()`, health check
- **S4.2** — Text chunker (`src/services/indexing/text_chunker.py`) — chunk indexing context
- **S4.3** — Embedding service (`src/services/embeddings/`) — `embed_query()` for KNN vectors

## Target Location
- `src/routers/search.py` — Search router with hybrid search endpoint
- `src/schemas/api/search.py` — Request/response Pydantic models
- `tests/unit/test_search_router.py` — Unit tests

## Functional Requirements

### FR-1: Search Request Schema
- **What**: Pydantic model for hybrid search input
- **Inputs**: `query` (str, required, 1-500 chars), `size` (int, default 10, 1-100), `from_` (int, default 0, >= 0), `categories` (Optional[list[str]]), `use_hybrid` (bool, default True), `latest_papers` (bool, default False), `min_score` (Optional[float], 0.0-1.0)
- **Outputs**: Validated `HybridSearchRequest` model
- **Edge cases**: Empty query, query > 500 chars, size out of range

### FR-2: Search Response Schema
- **What**: Pydantic model for search results
- **Fields**: `query` (str), `total` (int), `hits` (list[SearchHit]), `size` (int), `from_` (int), `search_mode` (str: "hybrid" | "bm25")
- **SearchHit fields**: `arxiv_id` (str), `title` (str), `authors` (list[str]), `abstract` (str), `pdf_url` (str), `score` (float), `highlights` (dict), `chunk_text` (str), `chunk_id` (str), `section_title` (Optional[str])

### FR-3: Query Embedding with Graceful Fallback
- **What**: Attempt to embed the query using Jina; on failure, fall back to BM25-only
- **Inputs**: `query` string, `use_hybrid` flag
- **Outputs**: `query_embedding` (list[float] | None), `search_mode` ("hybrid" | "bm25")
- **Edge cases**: Jina API timeout, rate limit, auth error — all should trigger fallback, not HTTP error

### FR-4: Unified Search Execution
- **What**: Call `opensearch_client.search_unified()` with query + optional embedding
- **Inputs**: Query, embedding (or None), pagination, filters
- **Outputs**: Raw OpenSearch results dict
- **Edge cases**: OpenSearch unreachable → 503, empty results → return empty hits list

### FR-5: Result Mapping
- **What**: Transform raw OpenSearch hits into typed `SearchHit` models
- **Inputs**: Raw hit dicts from OpenSearch
- **Outputs**: List of `SearchHit` with all fields populated
- **Edge cases**: Missing fields in hit → use defaults (empty string, empty list, 0.0)

### FR-6: Health Check Guard
- **What**: Before executing search, verify OpenSearch is reachable
- **Inputs**: OpenSearch client
- **Outputs**: Proceeds if healthy, raises 503 if not
- **Edge cases**: Timeout on health check

### FR-7: POST /api/v1/search Endpoint
- **What**: Main search endpoint combining FR-3 through FR-6
- **Route**: `POST /api/v1/search`
- **Dependencies**: `OpenSearchDep`, `EmbeddingsDep` (FastAPI DI)
- **Returns**: `SearchResponse` with `search_mode` indicating actual mode used
- **Error handling**: 422 for validation, 503 for OpenSearch down, 500 for unexpected errors

### FR-8: GET /api/v1/search/health Endpoint
- **What**: Quick health check for search subsystem
- **Route**: `GET /api/v1/search/health`
- **Returns**: `{"status": "ok", "opensearch": true/false}`

## Tangible Outcomes
- [ ] `POST /api/v1/search` returns hybrid search results with BM25 + KNN + RRF fusion
- [ ] Graceful fallback to BM25-only when embedding fails (search_mode="bm25" in response)
- [ ] Request validation rejects invalid queries (empty, too long, bad pagination)
- [ ] 503 response when OpenSearch is unreachable
- [ ] Search results include chunk-level data (chunk_text, chunk_id, section_title)
- [ ] `GET /api/v1/search/health` returns OpenSearch connectivity status
- [ ] All tests pass with mocked OpenSearch and embeddings services
- [ ] Router registered in FastAPI app and accessible via /docs

## Test-Driven Requirements

### Tests to Write First
1. `test_search_request_valid`: Valid request passes validation
2. `test_search_request_empty_query`: Empty query rejected
3. `test_search_request_query_too_long`: >500 char query rejected
4. `test_search_request_defaults`: Default values correct (size=10, from_=0, use_hybrid=True)
5. `test_search_hit_from_raw`: Raw OpenSearch hit maps to SearchHit correctly
6. `test_search_hit_missing_fields`: Missing fields use defaults
7. `test_hybrid_search_success`: Full hybrid search returns results with search_mode="hybrid"
8. `test_bm25_fallback_on_embedding_failure`: Embedding error triggers BM25 fallback
9. `test_bm25_only_when_use_hybrid_false`: Explicit BM25 mode skips embedding
10. `test_search_opensearch_down`: Returns 503 when OpenSearch unreachable
11. `test_search_empty_results`: Empty results return empty hits list
12. `test_search_with_category_filter`: Categories passed to OpenSearch
13. `test_search_health_endpoint`: Health check returns correct status
14. `test_search_pagination`: from_ and size correctly paginate results

### Mocking Strategy
- Mock `OpenSearchClient.search_unified()` → return fake hits dict
- Mock `OpenSearchClient.health_check()` → return True/False
- Mock `JinaEmbeddingsClient.embed_query()` → return fake 1024-dim vector or raise error
- Use `httpx.AsyncClient` + `ASGITransport` for endpoint testing
- Override FastAPI dependencies with mocks

### Coverage
- All public functions tested
- Edge cases: empty results, embedding failures, OpenSearch down, invalid input
- Error paths: 422, 503, 500 responses
