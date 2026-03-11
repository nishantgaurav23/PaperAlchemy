# Spec S4b.3 -- Multi-Query Retrieval

## Overview
Multi-query retrieval improves search recall by generating multiple variations of the user's query via an LLM, running parallel searches for each variation, then deduplicating and fusing the results. This addresses the vocabulary mismatch problem — a single query may miss relevant documents that use different terminology for the same concept.

Flow: User query → LLM generates 3-5 query variations → parallel hybrid search per variation → deduplicate by chunk_id → RRF fusion of per-query rankings → return top-K fused results

## Dependencies
- **S4.4** -- Hybrid search (`src/routers/search.py`, `src/schemas/api/search.py`) -- provides `SearchHit` schema and hybrid search infrastructure
- **S5.1** -- LLM client (`src/services/llm/`) -- provides `LLMProvider` interface for generating query variations

## Target Location
- `src/services/retrieval/multi_query.py` -- MultiQueryService implementation
- `tests/unit/test_multi_query.py` -- Unit tests
- `notebooks/specs/S4b.3_multi_query.ipynb` -- Interactive verification

## Functional Requirements

### FR-1: MultiQueryService Interface
- **What**: Async service class that generates query variations and performs fused retrieval
- **Inputs**: `query` (str) -- the user's research question
- **Outputs**: `MultiQueryResult` containing: `original_query` (str), `generated_queries` (list[str]), `results` (list[SearchHit])
- **Edge cases**: Empty query → raise ValueError; LLM failure → fallback to single original query search

### FR-2: Query Variation Generation
- **What**: Use LLM to generate 3-5 semantically distinct reformulations of the original query
- **Prompt**: System prompt instructs the LLM to generate diverse query variations (synonyms, broader/narrower scope, different angles) while preserving the original intent
- **Output Parsing**: LLM returns numbered list (1. ... 2. ... etc.), parse into list of strings
- **Temperature**: Moderate (0.7) for creative but relevant variations
- **Variation count**: Configurable, default 3 (`num_queries` setting)
- **Edge cases**: LLM returns empty → use original query only; LLM returns fewer than requested → use what's available; Parsing fails → use original query only

### FR-3: Parallel Hybrid Search
- **What**: Execute hybrid search (BM25 + KNN + RRF) for each query variation concurrently using `asyncio.gather`
- **Inputs**: list of query strings, `top_k` per query (default 20)
- **Outputs**: list of per-query result lists (list[list[SearchHit]])
- **Behavior**: Each query goes through the full hybrid search pipeline (embed → BM25 + KNN → RRF)
- **Edge cases**: Individual query search failure → skip that query, continue with others; All queries fail → return empty results

### FR-4: Result Deduplication
- **What**: Remove duplicate chunks across query result sets, identified by `chunk_id`
- **Behavior**: When the same chunk appears in multiple query results, keep the occurrence with the highest score
- **Edge cases**: No duplicates → return all results as-is

### FR-5: Reciprocal Rank Fusion (RRF)
- **What**: Fuse results from multiple queries using RRF scoring
- **Formula**: `rrf_score(doc) = Σ 1 / (k + rank_in_query_i)` where k=60 (standard RRF constant)
- **Behavior**: For each unique chunk, compute RRF score across all query result sets, sort by descending RRF score, return top-K
- **Inputs**: Per-query ranked result lists, `top_k` final results (default 20)
- **Outputs**: Fused, deduplicated, re-ranked list[SearchHit] with updated scores

### FR-6: Graceful Fallback
- **What**: If query generation or all searches fail, fall back to single standard hybrid search with original query
- **Behavior**: Log a warning, perform standard hybrid search, return results as-is
- **Principle**: Multi-query is an enhancement — search must always work

### FR-7: Configuration
- **What**: Add `MultiQuerySettings` to `src/config.py`
- **Fields**: `enabled` (bool, default True), `num_queries` (int, default 3), `temperature` (float, default 0.7), `max_tokens` (int, default 300), `rrf_k` (int, default 60)
- **Env prefix**: `MULTI_QUERY__`

### FR-8: Factory Function for DI
- **What**: `create_multi_query_service(settings, llm_client, embeddings_client, opensearch_client)` factory
- **Behavior**: Returns configured `MultiQueryService` instance
- **Registration**: Add to `src/dependency.py` when ready

## Tangible Outcomes
- [ ] `src/services/retrieval/multi_query.py` exists with `MultiQueryService` class
- [ ] `generate_query_variations(query)` returns 3-5 query variations via LLM
- [ ] `retrieve_with_multi_query(query, top_k)` returns `MultiQueryResult` with variations + fused results
- [ ] Parallel search via `asyncio.gather` for all query variations
- [ ] RRF fusion deduplicates and re-ranks results across queries
- [ ] Fallback to single query search on any failure
- [ ] `MultiQuerySettings` added to `src/config.py` with `MULTI_QUERY__` env prefix
- [ ] All external services mocked in tests (LLM, Jina, OpenSearch)
- [ ] All tests pass
- [ ] Notebook created at `notebooks/specs/S4b.3_multi_query.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_generate_query_variations`: Verify LLM called with correct prompt, returns list of variations
2. `test_generate_query_variations_empty_query`: Verify ValueError raised
3. `test_generate_query_variations_llm_failure`: Verify fallback returns [original_query]
4. `test_generate_query_variations_empty_response`: Verify fallback returns [original_query]
5. `test_generate_query_variations_parsing`: Verify numbered list parsing handles various formats
6. `test_retrieve_with_multi_query_full_flow`: Verify end-to-end: generate variations → parallel search → deduplicate → fuse → return
7. `test_retrieve_with_multi_query_deduplication`: Verify same chunk_id across queries is deduplicated (highest score kept)
8. `test_retrieve_with_multi_query_rrf_fusion`: Verify RRF scoring ranks results correctly
9. `test_retrieve_with_multi_query_fallback_on_generation_error`: Verify falls back to single query search
10. `test_retrieve_with_multi_query_partial_search_failure`: Verify continues when some queries fail
11. `test_retrieve_with_multi_query_respects_top_k`: Verify top_k parameter limits final results
12. `test_retrieve_with_multi_query_empty_results`: Verify returns empty MultiQueryResult when no hits

### Mocking Strategy
- Mock LLM client (`LLMProvider.generate()`) to return numbered query variations
- Mock `JinaEmbeddingsClient.embed_query()` to return fake 1024-dim vectors
- Mock `OpenSearchClient.hybrid_search()` to return fake `SearchHit` results
- Use `AsyncMock` for all async service methods
- No real API calls in unit tests

### Coverage
- All public methods tested
- Fallback paths tested (LLM failure, partial search failure, all search failure)
- Edge cases (empty query, empty results, parsing edge cases)
- RRF fusion math verified
- Deduplication logic verified
- Configuration validation tested
