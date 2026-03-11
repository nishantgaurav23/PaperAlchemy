# Spec S4b.1 -- Cross-Encoder Re-ranking

## Overview
Second-stage re-ranking service that takes the top-N results from hybrid search (S4.4) and re-scores them using a cross-encoder model for more accurate relevance ranking. Supports local cross-encoder (ms-marco-MiniLM-L-12-v2 via sentence-transformers) or cloud-based Cohere Rerank API. The service re-scores the top 20 results and returns the top 5 most relevant.

## Dependencies
- **S4.4** — Hybrid search (`src/routers/search.py`, `src/schemas/api/search.py`) — provides `SearchHit` results to re-rank

## Target Location
- `src/services/reranking/__init__.py` — Package init with exports
- `src/services/reranking/service.py` — RerankerService with provider abstraction
- `src/services/reranking/factory.py` — Factory function for DI
- `tests/unit/test_reranker.py` — Unit tests
- `notebooks/specs/S4b.1_reranker.ipynb` — Interactive verification

## Functional Requirements

### FR-1: RerankerService Interface
- **What**: Async service class with a `rerank(query, documents, top_k)` method
- **Inputs**: `query` (str), `documents` (list of dicts with at least `text` and an identifier), `top_k` (int, default from config)
- **Outputs**: List of `RerankResult` (document reference + relevance_score), sorted by score descending, truncated to top_k
- **Edge cases**: Empty documents list → return empty list; top_k > len(documents) → return all documents sorted

### FR-2: Local Cross-Encoder Provider
- **What**: Use `sentence-transformers` `CrossEncoder` to score query-document pairs
- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (configurable via `RerankerSettings`)
- **Scoring**: Score each (query, document_text) pair, normalize scores to 0.0-1.0 range
- **Device**: CPU by default (configurable via settings)
- **Thread safety**: Model loaded once, predict in thread pool executor to avoid blocking async loop
- **Edge cases**: Model download failure → raise `RerankerError`; empty text → score 0.0

### FR-3: Cohere Rerank Provider (Optional Cloud Fallback)
- **What**: Alternative provider using Cohere Rerank API via httpx
- **Inputs**: Same as FR-1
- **Outputs**: Same as FR-1 (normalized scores)
- **Config**: `RERANKER__PROVIDER` = "local" (default) or "cohere"; `RERANKER__COHERE_API_KEY`
- **Edge cases**: API failure → raise `RerankerError` with descriptive message

### FR-4: Integration with Search Results
- **What**: Accept `SearchHit` objects from hybrid search, extract text for re-ranking, return re-ranked hits
- **Method**: `rerank_search_hits(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]`
- **Text extraction**: Use `chunk_text` if available, else `abstract`, else `title`
- **Outputs**: Original `SearchHit` objects re-ordered by cross-encoder score, with `score` field updated
- **Edge cases**: Hits with no text content → assign score 0.0 and rank last

### FR-5: Configuration via Settings
- **What**: Use existing `RerankerSettings` from `src/config.py`
- **Fields**: `model` (str), `top_k` (int, default 5), `device` (str, default "cpu")
- **Additional**: Add `provider` (str, default "local") and `cohere_api_key` (str, default "") to `RerankerSettings`

### FR-6: Factory Function for DI
- **What**: `create_reranker_service(settings)` factory returning configured `RerankerService`
- **Behavior**: Based on `settings.reranker.provider`, instantiate local or Cohere provider
- **Registration**: Add `RerankerDep` type alias to `src/dependency.py`

## Tangible Outcomes
- [ ] `src/services/reranking/service.py` exists with `RerankerService` class
- [ ] `rerank()` accepts query + document list, returns top-K scored results
- [ ] `rerank_search_hits()` accepts `SearchHit` list, returns re-ranked `SearchHit` list
- [ ] Local cross-encoder loads `ms-marco-MiniLM-L-12-v2` model and scores pairs
- [ ] Scores are normalized to 0.0-1.0 range
- [ ] Model inference runs in thread pool (non-blocking async)
- [ ] Empty input returns empty output (no crash)
- [ ] Factory creates service based on provider config
- [ ] `RerankerDep` available in `src/dependency.py`
- [ ] All tests pass with mocked model (no real model download in tests)

## Test-Driven Requirements

### Tests to Write First
1. `test_rerank_returns_sorted_results`: Verify results sorted by score descending
2. `test_rerank_respects_top_k`: Verify only top_k results returned
3. `test_rerank_empty_documents`: Verify empty list in → empty list out
4. `test_rerank_top_k_exceeds_docs`: Verify returns all docs when top_k > len(docs)
5. `test_rerank_search_hits`: Verify SearchHit objects re-ranked correctly
6. `test_rerank_search_hits_text_extraction`: Verify text extracted from chunk_text, then abstract, then title
7. `test_rerank_search_hits_no_text`: Verify hits with no text get score 0.0
8. `test_scores_normalized`: Verify all scores in 0.0-1.0 range
9. `test_factory_creates_local_provider`: Verify factory with provider="local"
10. `test_factory_creates_cohere_provider`: Verify factory with provider="cohere"

### Mocking Strategy
- Mock `CrossEncoder.predict()` to return fake scores (avoid model download)
- Mock `httpx.AsyncClient` for Cohere API calls
- Use real `SearchHit` Pydantic models with test data

### Coverage
- All public methods tested
- Edge cases (empty, oversized top_k, missing text)
- Error paths (model load failure, API failure)
