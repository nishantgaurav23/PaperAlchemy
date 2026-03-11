# Spec S5.2 -- RAG Chain

## Overview
RAG (Retrieval-Augmented Generation) chain that orchestrates the full pipeline: retrieve relevant paper chunks via the advanced retrieval pipeline (S4b.5), build a citation-enforcing prompt with the retrieved context, and generate an answer using the unified LLM client (S5.1). Every response must include inline citations [1],[2] and a source list with paper title, authors, and arXiv link.

## Dependencies
- **S5.1** (LLM Client) — `LLMProvider` protocol for generate/stream
- **S4b.5** (Retrieval Pipeline) — `RetrievalPipeline.retrieve()` → `RetrievalResult`

## Target Location
- `src/services/rag/chain.py` — RAG chain service
- `src/services/rag/prompts.py` — Prompt templates (system + user)
- `src/services/rag/models.py` — RAG response models
- `src/services/rag/factory.py` — Factory function
- `src/services/rag/__init__.py` — Exports

## Functional Requirements

### FR-1: RAG Chain Service
- **What**: `RAGChain` class that accepts a query, retrieves relevant documents, builds a prompt, and generates a citation-backed answer.
- **Inputs**:
  - `query: str` — User's research question
  - `top_k: int | None = None` — Number of chunks to retrieve (default from settings)
  - `categories: list[str] | None = None` — Optional arXiv category filter
  - `temperature: float | None = None` — LLM temperature override
- **Outputs**: `RAGResponse` containing answer text, sources list, retrieval metadata
- **Edge cases**:
  - Empty query → raise `ValueError`
  - No documents retrieved → return response with "no relevant papers found" message
  - LLM failure → propagate `LLMServiceError` / `LLMConnectionError` / `LLMTimeoutError`
  - Retrieval failure → propagate exception (don't silently return empty)

### FR-2: Citation-Enforcing Prompt Templates
- **What**: System and user prompt templates that instruct the LLM to cite sources inline using [1],[2] notation and append a numbered source list.
- **Inputs**: Retrieved `SearchHit` list (with arxiv_id, title, authors, chunk_text)
- **Outputs**: Formatted prompt string with context chunks numbered and labeled
- **Edge cases**:
  - Chunks with missing metadata (no title/authors) → use "Unknown" fallback
  - Very long context (many chunks) → truncate to fit within token budget
- **Prompt structure**:
  1. System prompt: "You are a research assistant. Always cite sources using [N] notation..."
  2. Context block: Numbered chunks with title, authors, arXiv ID
  3. User question
  4. Instruction to append source list at end

### FR-3: RAG Response Model
- **What**: Pydantic models for structured RAG responses
- **Fields**:
  - `answer: str` — Generated text with inline citations
  - `sources: list[SourceReference]` — Ordered list of cited papers
  - `query: str` — Original query
  - `retrieval_metadata: RetrievalMetadata` — Pipeline stats (stages, timings, candidate count)
  - `llm_metadata: LLMMetadata` — Provider, model, token usage
- **SourceReference fields**: `index: int`, `arxiv_id: str`, `title: str`, `authors: list[str]`, `arxiv_url: str`, `chunk_text: str`, `score: float`

### FR-4: Streaming RAG
- **What**: `aquery_stream()` method that streams LLM tokens via `AsyncIterator[str]`, then yields source metadata as a final JSON chunk.
- **Inputs**: Same as FR-1
- **Outputs**: `AsyncIterator[str]` — text tokens followed by `\n\n[SOURCES]` JSON
- **Edge cases**:
  - Stream interruption → clean up resources
  - No documents → yield "no relevant papers" message, no streaming

### FR-5: Factory Function
- **What**: `create_rag_chain(settings, llm_provider, retrieval_pipeline) -> RAGChain`
- **Inputs**: Settings, pre-built LLM provider, pre-built retrieval pipeline
- **Outputs**: Configured `RAGChain` instance

## Tangible Outcomes
- [ ] `RAGChain.aquery(query)` returns `RAGResponse` with answer + sources
- [ ] Answer text contains inline citations [1], [2], etc.
- [ ] Sources list includes arxiv_id, title, authors, arxiv_url for each cited paper
- [ ] `RAGChain.aquery_stream(query)` yields tokens then source JSON
- [ ] Empty retrieval results → graceful "no papers found" response (not an error)
- [ ] Prompt templates enforce citation format in LLM output
- [ ] All external services (LLM, retrieval) are injected — fully mockable
- [ ] Factory function creates properly wired RAGChain

## Test-Driven Requirements

### Tests to Write First
1. `test_rag_chain_returns_answer_with_sources` — Query returns RAGResponse with non-empty answer and sources
2. `test_rag_chain_answer_contains_citations` — Answer text includes [1], [2] inline refs
3. `test_rag_chain_sources_have_metadata` — Each source has arxiv_id, title, authors, arxiv_url
4. `test_rag_chain_empty_query_raises_error` — Empty string query raises ValueError
5. `test_rag_chain_no_documents_found` — Retrieval returns empty → graceful message
6. `test_rag_chain_llm_error_propagates` — LLM failure raises appropriate exception
7. `test_rag_chain_prompt_includes_context` — Built prompt contains chunk text and metadata
8. `test_rag_chain_prompt_enforces_citations` — System prompt includes citation instructions
9. `test_rag_chain_streaming_yields_tokens` — Stream method yields text chunks
10. `test_rag_chain_streaming_ends_with_sources` — Stream ends with [SOURCES] JSON block
11. `test_rag_chain_categories_passed_to_retrieval` — Category filter forwarded to pipeline
12. `test_rag_chain_temperature_passed_to_llm` — Temperature override forwarded to LLM
13. `test_factory_creates_rag_chain` — Factory returns configured RAGChain instance

### Mocking Strategy
- **LLM Provider**: `AsyncMock` implementing `LLMProvider` protocol — mock `generate()` to return citation-formatted text, mock `generate_stream()` to yield tokens
- **Retrieval Pipeline**: `AsyncMock` for `RetrievalPipeline.retrieve()` — return pre-built `RetrievalResult` with test `SearchHit` data
- **No real API calls**: All external services fully mocked

### Coverage
- All public methods tested (`aquery`, `aquery_stream`, prompt building)
- Edge cases: empty query, no results, LLM errors, missing metadata
- Integration: categories forwarded, temperature forwarded
- Factory function tested
