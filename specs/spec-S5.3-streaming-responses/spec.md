# Spec S5.3 -- Streaming Responses (SSE)

## Overview
SSE (Server-Sent Events) streaming endpoint that exposes the RAG chain's `aquery_stream()` method as a FastAPI route. Clients receive token-by-token LLM output in real time via the SSE protocol, followed by source metadata as a final JSON event. This enables responsive UX in the Next.js frontend (S9.4) and future chat interface (S7.3).

## Dependencies
- **S5.2** (RAG Chain) — `RAGChain.aquery_stream()` → `AsyncIterator[str]` with `[SOURCES]` JSON at end

## Target Location
- `src/routers/ask.py` — SSE streaming endpoint + non-streaming fallback
- `src/schemas/api/ask.py` — Request/response Pydantic models
- `src/schemas/api/__init__.py` — Updated exports

## Functional Requirements

### FR-1: SSE Streaming Endpoint
- **What**: `POST /api/v1/ask` endpoint that streams RAG responses via Server-Sent Events
- **Inputs**: `AskRequest` body with:
  - `query: str` — Research question (required, min 1 char)
  - `top_k: int | None = None` — Number of chunks to retrieve
  - `categories: list[str] | None = None` — arXiv category filter
  - `temperature: float | None = None` — LLM temperature override (0.0-2.0)
  - `stream: bool = True` — Whether to stream (default True)
- **Outputs (stream=True)**: `text/event-stream` response with SSE events:
  - `event: token` / `data: {text chunk}` — Individual LLM tokens
  - `event: sources` / `data: {JSON array of SourceReference}` — Source metadata at end
  - `event: done` / `data: {}` — Stream complete signal
  - `event: error` / `data: {error JSON}` — On failure
- **Outputs (stream=False)**: Standard JSON `AskResponse` with answer + sources
- **Edge cases**:
  - Empty/whitespace query → 422 validation error
  - Client disconnect mid-stream → clean generator close, no server error
  - LLM timeout → `event: error` with timeout message
  - No documents found → stream "no relevant papers" message + empty sources
  - RAG chain raises exception → `event: error` with structured error

### FR-2: Request/Response Models
- **What**: Pydantic models for the ask endpoint
- **AskRequest**: query, top_k, categories, temperature, stream
- **AskResponse** (non-streaming): answer, sources (list of SourceReference from rag models), query, retrieval_metadata, llm_metadata
- **SSE Event Models**: Token event, sources event, done event, error event — all serializable

### FR-3: Non-Streaming Fallback
- **What**: When `stream=False`, call `RAGChain.aquery()` and return full JSON response
- **Inputs**: Same `AskRequest`
- **Outputs**: `AskResponse` JSON with complete answer + sources
- **Edge cases**: Same error handling as streaming path

### FR-4: SSE Event Formatting
- **What**: Proper SSE protocol formatting with event types and JSON data
- **Format**:
  ```
  event: token
  data: {"text": "Transformers are"}

  event: token
  data: {"text": " a neural network"}

  event: sources
  data: [{"index": 1, "arxiv_id": "1706.03762", "title": "Attention Is All You Need", ...}]

  event: done
  data: {}

  ```
- **Rules**:
  - Each event separated by double newline
  - Data field is always valid JSON
  - Source references parsed from RAG chain's `[SOURCES]` block
  - Token events contain the raw text chunk from the LLM

### FR-5: Dependency Injection Integration
- **What**: Wire into FastAPI DI using existing `RAGChainDep` from `src/dependency.py`
- **Router registration**: Add to `src/main.py` alongside ping, search, ingest routers

## Tangible Outcomes
- [ ] `POST /api/v1/ask` with `stream=true` returns `text/event-stream` content type
- [ ] SSE events follow `event: {type}\ndata: {json}\n\n` format
- [ ] Token events stream in real time (not buffered until complete)
- [ ] Sources event contains full paper metadata (arxiv_id, title, authors, arxiv_url)
- [ ] Done event signals stream completion
- [ ] Error events returned on LLM/retrieval failures (not 500 crashes)
- [ ] `POST /api/v1/ask` with `stream=false` returns JSON `AskResponse`
- [ ] Empty query → 422 validation error (not 500)
- [ ] Client disconnect handled gracefully (no server-side exceptions)
- [ ] Router registered in `src/main.py`
- [ ] `RAGChainDep` injected — fully mockable in tests

## Test-Driven Requirements

### Tests to Write First
1. `test_ask_streaming_returns_event_stream` — Response has content-type `text/event-stream`
2. `test_ask_streaming_yields_token_events` — Stream contains `event: token` with text data
3. `test_ask_streaming_yields_sources_event` — Stream contains `event: sources` with paper metadata
4. `test_ask_streaming_yields_done_event` — Stream ends with `event: done`
5. `test_ask_streaming_sources_have_arxiv_links` — Sources contain arxiv_url field
6. `test_ask_non_streaming_returns_json` — stream=false returns AskResponse JSON
7. `test_ask_non_streaming_has_citations` — Non-streaming response contains sources
8. `test_ask_empty_query_returns_422` — Validation error on empty query
9. `test_ask_no_documents_streams_gracefully` — Empty retrieval → message + empty sources
10. `test_ask_llm_error_streams_error_event` — LLM failure → `event: error`
11. `test_ask_request_model_validation` — AskRequest validates field constraints
12. `test_ask_categories_forwarded` — Category filter passed to RAG chain
13. `test_ask_temperature_forwarded` — Temperature override passed to RAG chain

### Mocking Strategy
- **RAGChain**: `AsyncMock` — mock `aquery()` to return `RAGResponse`, mock `aquery_stream()` to yield test tokens + `[SOURCES]` JSON
- **FastAPI TestClient**: Use `httpx.AsyncClient` with `ASGITransport` for SSE testing
- **DI Override**: Use `app.dependency_overrides[get_rag_chain]` to inject mock

### Coverage
- All public endpoints tested (streaming + non-streaming)
- SSE event format validated (event type, data JSON)
- Edge cases: empty query, no docs, LLM errors, client disconnect
- DI integration: mock injected via overrides
