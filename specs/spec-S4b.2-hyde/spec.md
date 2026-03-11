# Spec S4b.2 -- HyDE (Hypothetical Document Embeddings)

## Overview
HyDE (Hypothetical Document Embeddings) is an advanced retrieval technique that improves search recall by bridging the vocabulary gap between user queries and indexed documents. Instead of embedding the raw query directly, the LLM generates a hypothetical answer/document passage, which is then embedded and used for vector similarity search. This finds documents that are semantically similar to what a good answer would look like, rather than what the question looks like.

Flow: User query -> LLM generates hypothetical document -> Embed hypothetical doc -> KNN search for similar real docs -> Return real docs

## Dependencies
- **S4.4** -- Hybrid search (`src/routers/search.py`, `src/schemas/api/search.py`) -- provides search infrastructure and `SearchHit` schema
- **S5.1** -- LLM client (`src/services/llm/`) -- provides `LLMProvider` interface for generating hypothetical documents

## Target Location
- `src/services/retrieval/__init__.py` -- Package init
- `src/services/retrieval/hyde.py` -- HyDEService implementation
- `tests/unit/test_hyde.py` -- Unit tests
- `notebooks/specs/S4b.2_hyde.ipynb` -- Interactive verification

## Functional Requirements

### FR-1: HyDEService Interface
- **What**: Async service class that generates hypothetical documents and uses them for retrieval
- **Inputs**: `query` (str) -- the user's research question
- **Outputs**: `HyDEResult` containing: `hypothetical_document` (str), `query_embedding` (list[float]), `results` (list[SearchHit])
- **Edge cases**: Empty query -> raise ValueError; LLM generation failure -> fallback to standard query embedding

### FR-2: Hypothetical Document Generation
- **What**: Use LLM to generate a hypothetical passage that would answer the query
- **Prompt**: System prompt instructs the LLM to write a short (150-200 word) academic passage answering the question, as if from a real research paper
- **Temperature**: Low (0.3) for factual consistency
- **Timeout**: Configurable, default 30s
- **Edge cases**: LLM returns empty -> fallback to original query; LLM timeout -> fallback to original query

### FR-3: Hypothetical Document Embedding
- **What**: Embed the hypothetical document using `JinaEmbeddingsClient.embed_query()` (treats the hypothetical doc as the query for asymmetric search)
- **Inputs**: Hypothetical document text (str)
- **Outputs**: 1024-dim embedding vector
- **Edge cases**: Embedding failure -> fallback to embedding original query

### FR-4: Vector Search with HyDE Embedding
- **What**: Use the hypothetical document's embedding for KNN search via OpenSearch
- **Behavior**: Perform KNN-only search (not hybrid) using the hypothetical embedding, then return top-K results
- **Inputs**: HyDE embedding vector, `top_k` (int, default 20)
- **Outputs**: list[SearchHit] from OpenSearch KNN search
- **Edge cases**: No results -> return empty list

### FR-5: Graceful Fallback
- **What**: If HyDE generation or embedding fails at any stage, fall back to standard query embedding
- **Behavior**: Log a warning, embed the original query, perform standard KNN search
- **Principle**: HyDE is an enhancement, not a requirement -- search must always work

### FR-6: Configuration
- **What**: Add `HyDESettings` to `src/config.py` or use existing LLM settings
- **Fields**: `enabled` (bool, default True), `max_tokens` (int, default 300), `temperature` (float, default 0.3), `timeout` (int, default 30)
- **Env prefix**: `HYDE__`

### FR-7: Factory Function for DI
- **What**: `create_hyde_service(settings, llm_client, embeddings_client, opensearch_client)` factory
- **Behavior**: Returns configured `HyDEService` instance
- **Registration**: Add to `src/dependency.py` when LLM client (S5.1) is available

## Tangible Outcomes
- [ ] `src/services/retrieval/hyde.py` exists with `HyDEService` class
- [ ] `generate_hypothetical_document(query)` returns a hypothetical passage via LLM
- [ ] `retrieve_with_hyde(query, top_k)` returns `HyDEResult` with hypothetical doc + search hits
- [ ] Fallback to standard query embedding on any failure (LLM or embedding)
- [ ] `HyDESettings` added to `src/config.py` with `HYDE__` env prefix
- [ ] Low temperature (0.3) used for factual hypothetical generation
- [ ] All external services mocked in tests (LLM, Jina, OpenSearch)
- [ ] All tests pass
- [ ] Notebook created at `notebooks/specs/S4b.2_hyde.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_generate_hypothetical_document`: Verify LLM called with correct prompt, returns hypothetical text
2. `test_generate_hypothetical_document_empty_query`: Verify ValueError raised
3. `test_generate_hypothetical_document_llm_failure`: Verify fallback returns original query
4. `test_retrieve_with_hyde_full_flow`: Verify end-to-end: generate -> embed -> search -> return results
5. `test_retrieve_with_hyde_fallback_on_llm_error`: Verify falls back to standard embedding when LLM fails
6. `test_retrieve_with_hyde_fallback_on_embed_error`: Verify falls back when embedding of hypothetical doc fails
7. `test_retrieve_with_hyde_empty_results`: Verify returns empty HyDEResult when no search hits
8. `test_retrieve_with_hyde_respects_top_k`: Verify top_k parameter passed to search
9. `test_hyde_prompt_format`: Verify the system prompt is academic/research-focused
10. `test_hyde_uses_low_temperature`: Verify temperature=0.3 is used for generation

### Mocking Strategy
- Mock LLM client (`LLMProvider.generate()`) to return fake hypothetical passages
- Mock `JinaEmbeddingsClient.embed_query()` to return fake 1024-dim vectors
- Mock `OpenSearchClient.knn_search()` to return fake `SearchHit` results
- Use `AsyncMock` for all async service methods
- No real API calls in unit tests

### Coverage
- All public methods tested
- Fallback paths tested (LLM failure, embedding failure)
- Edge cases (empty query, empty results)
- Configuration validation tested
