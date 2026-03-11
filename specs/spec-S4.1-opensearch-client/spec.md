# Spec S4.1 -- OpenSearch Client + Index Configuration

## Overview
OpenSearch client wrapping opensearch-py with PaperAlchemy-specific logic: index management (BM25 + KNN hybrid mappings), RRF search pipeline setup, bulk chunk indexing, BM25/vector/hybrid search, and chunk lifecycle management. This is the search engine foundation for all retrieval in the RAG pipeline.

## Dependencies
- S1.3 (Docker infrastructure) -- OpenSearch 2.19 service in compose.yml

## Target Location
- `src/services/opensearch/` (client.py, index_config.py, query_builder.py, factory.py, __init__.py)

## Functional Requirements

### FR-1: Index Configuration
- **What**: Define OpenSearch index mappings for hybrid search (BM25 + KNN)
- **Inputs**: Vector dimension (1024), space type (cosinesimil), analyzer configs
- **Outputs**: `ARXIV_PAPERS_CHUNKS_MAPPING` dict with settings + mappings
- **Details**:
  - Custom analyzers: `text_analyzer` (lowercase + stop + snowball), `standard_analyzer`
  - Fields: chunk_id, arxiv_id, paper_id, chunk_index, chunk_text, chunk_word_count, start_char, end_char, embedding (knn_vector 1024-dim HNSW), title, authors, abstract, categories, published_date, section_title, parent_chunk_id, embedding_model, created_at, updated_at
  - `dynamic: "strict"` to prevent accidental field creation
  - HNSW params: ef_construction=512, m=16, engine=nmslib
  - `index.knn: true` for vector search
- **Edge cases**: Index already exists, invalid mapping fields

### FR-2: RRF Search Pipeline
- **What**: Define and create OpenSearch search pipeline for Reciprocal Rank Fusion
- **Inputs**: Pipeline ID, rank_constant (k=60)
- **Outputs**: `HYBRID_RRF_PIPELINE` config dict
- **Details**:
  - Uses `score-ranker-processor` with `rrf` technique
  - Formula: score = sum(1/(k+rank)) where k=60
  - Pipeline ID from settings: `settings.opensearch.rrf_pipeline_name`

### FR-3: OpenSearch Client - Connection & Health
- **What**: Async-compatible OpenSearch client with health check
- **Inputs**: Host URL from settings, SSL/auth config
- **Outputs**: Connected client instance, health check result (bool)
- **Details**:
  - Uses `opensearchpy.OpenSearch` (sync client, run in executor for async compat)
  - Health check verifies cluster status is green or yellow
  - Index name: `{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}`
- **Edge cases**: Connection refused, timeout, cluster red status

### FR-4: Index & Pipeline Setup
- **What**: Create hybrid index and RRF pipeline on demand
- **Inputs**: force flag (recreate if exists)
- **Outputs**: Dict with creation status per component
- **Details**:
  - `setup_indices(force=False)` orchestrates both
  - `_create_hybrid_index(force)`: create/recreate chunk index
  - `_create_rrf_pipeline(force)`: create/recreate search pipeline via `/_search/pipeline/` API
- **Edge cases**: Index exists (skip), force=True (delete + recreate), pipeline exists

### FR-5: BM25 Search (Query Builder)
- **What**: Build and execute BM25 keyword search queries
- **Inputs**: query text, size, from_, categories filter, latest flag, min_score
- **Outputs**: Dict with total count and hits list
- **Details**:
  - QueryBuilder constructs multi_match across chunk_text^3, title^2, abstract^1
  - Fuzziness="AUTO", prefix_length=2
  - Category filter via bool/filter
  - Highlighting with `<mark>` tags
  - Source excludes embedding field
- **Edge cases**: Empty query (match_all), no results, min_score filtering

### FR-6: Vector Search (KNN)
- **What**: Pure KNN vector search on chunk embeddings
- **Inputs**: query_embedding (List[float]), size, categories filter
- **Outputs**: Dict with total count and hits list
- **Details**:
  - KNN query on `embedding` field
  - Optional category filter via bool/must+filter
  - Excludes embedding from _source
- **Edge cases**: Wrong dimension vector, no indexed docs

### FR-7: Hybrid Search (BM25 + KNN + RRF)
- **What**: Native OpenSearch hybrid search combining BM25 and vector scores
- **Inputs**: query text, query_embedding, size, categories, min_score
- **Outputs**: Dict with total count and hits list (RRF-scored)
- **Details**:
  - Uses `hybrid` query with BM25 + KNN sub-queries
  - Executes with `search_pipeline` param pointing to RRF pipeline
  - Falls back to BM25-only if no embedding provided
- **Edge cases**: RRF pipeline not found, embedding failure → BM25 fallback

### FR-8: Unified Search Entry Point
- **What**: Single search method that routes to BM25 or hybrid based on inputs
- **Inputs**: query, optional embedding, size, from_, categories, latest, use_hybrid, min_score
- **Outputs**: Dict with total count and hits list
- **Details**:
  - If no embedding or use_hybrid=False → BM25 only
  - If embedding provided and use_hybrid=True → hybrid search
- **Edge cases**: All error cases from sub-methods, graceful fallback

### FR-9: Bulk Chunk Indexing
- **What**: Bulk index multiple chunks with embeddings
- **Inputs**: List of dicts with chunk_data and embedding
- **Outputs**: Dict with success/failed counts
- **Details**:
  - Uses opensearchpy.helpers.bulk
  - Refresh after bulk operation
- **Edge cases**: Empty list, partial failures, mapping violations

### FR-10: Chunk Lifecycle
- **What**: Delete chunks by paper, retrieve chunks by paper
- **Inputs**: arxiv_id
- **Outputs**: bool for delete, List[Dict] for retrieve
- **Details**:
  - `delete_paper_chunks(arxiv_id)`: delete_by_query on arxiv_id term
  - `get_chunks_by_paper(arxiv_id)`: sorted by chunk_index, excludes embedding
- **Edge cases**: Paper not found, no chunks exist

### FR-11: Index Statistics
- **What**: Get document count, size, existence status for the index
- **Inputs**: None (uses configured index name)
- **Outputs**: Dict with index_name, exists, document_count, size_in_bytes
- **Edge cases**: Index doesn't exist

### FR-12: Factory Functions
- **What**: Cached singleton and fresh client factory functions
- **Inputs**: Optional settings, optional host override
- **Outputs**: OpenSearchClient instance
- **Details**:
  - `make_opensearch_client()`: @lru_cache singleton for app use
  - `make_opensearch_client_fresh()`: new instance for notebooks/tests
- **Edge cases**: None settings (use defaults)

## Tangible Outcomes
- [ ] `src/services/opensearch/__init__.py` exports OpenSearchClient, factory, config
- [ ] `src/services/opensearch/index_config.py` with ARXIV_PAPERS_CHUNKS_MAPPING and HYBRID_RRF_PIPELINE
- [ ] `src/services/opensearch/query_builder.py` with QueryBuilder class
- [ ] `src/services/opensearch/client.py` with OpenSearchClient (all FR methods)
- [ ] `src/services/opensearch/factory.py` with make_opensearch_client + make_opensearch_client_fresh
- [ ] All tests pass: `pytest tests/unit/test_opensearch_client.py -v`
- [ ] Lint passes: `ruff check src/services/opensearch/`
- [ ] Notebook created: `notebooks/specs/S4.1_opensearch.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_index_config_chunks_mapping_structure`: Validates CHUNKS_MAPPING has correct settings, mappings, knn_vector field
2. `test_index_config_rrf_pipeline_structure`: Validates RRF pipeline has correct processor config
3. `test_query_builder_bm25_basic`: QueryBuilder.build() produces correct multi_match + bool structure
4. `test_query_builder_with_categories`: Adds filter clause for categories
5. `test_query_builder_chunk_mode_fields`: Correct fields for chunk search (chunk_text^3, title^2, abstract^1)
6. `test_query_builder_empty_query_match_all`: Empty query uses match_all
7. `test_query_builder_latest_papers_sort`: Sort by published_date desc
8. `test_query_builder_source_excludes_embedding`: Chunk mode excludes embedding
9. `test_client_health_check_healthy`: Mock cluster.health → green → True
10. `test_client_health_check_unhealthy`: Mock cluster.health → exception → False
11. `test_client_setup_indices`: Calls _create_hybrid_index + _create_rrf_pipeline
12. `test_client_create_hybrid_index_new`: Index doesn't exist → creates
13. `test_client_create_hybrid_index_exists`: Index exists → skips
14. `test_client_create_hybrid_index_force`: force=True → deletes + creates
15. `test_client_search_bm25`: BM25 search returns formatted results
16. `test_client_search_vectors`: KNN search returns formatted results
17. `test_client_search_hybrid`: Hybrid search with RRF pipeline
18. `test_client_search_unified_bm25_fallback`: No embedding → BM25 only
19. `test_client_search_unified_hybrid`: Embedding provided → hybrid
20. `test_client_bulk_index_chunks`: Bulk index succeeds with stats
21. `test_client_delete_paper_chunks`: Delete by arxiv_id
22. `test_client_get_chunks_by_paper`: Retrieve sorted chunks
23. `test_client_get_index_stats`: Returns stats dict
24. `test_client_get_index_stats_not_exists`: Index missing → exists=False
25. `test_factory_cached_singleton`: make_opensearch_client returns same instance
26. `test_factory_fresh_instance`: make_opensearch_client_fresh returns new instances

### Mocking Strategy
- Mock `opensearchpy.OpenSearch` entirely (no real OpenSearch connection)
- Mock `opensearchpy.helpers.bulk` for bulk indexing tests
- Use `unittest.mock.MagicMock` and `patch` for all OpenSearch API calls
- Mock `client.cluster.health()`, `client.indices.exists()`, `client.indices.create()`, `client.search()`, etc.

### Coverage
- All public methods tested
- Edge cases: empty results, connection errors, index not found
- Error paths: exceptions from OpenSearch API calls
