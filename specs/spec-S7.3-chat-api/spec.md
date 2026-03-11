# Spec S7.3 -- Chat API

## Overview
Chat endpoint with session management for conversational research Q&A. Provides `POST /api/v1/chat` with session-based conversation history, follow-up detection, coreference resolution, and citation-backed streaming responses. This is the API that the Next.js chat interface (S9.4) consumes.

Every response MUST cite papers with title, authors, and arXiv link. The endpoint uses the FollowUpHandler (S7.2) which detects follow-up questions, rewrites them with full context, and runs them through the RAG pipeline.

## Dependencies
- **S7.1** — Conversation memory (Redis-backed session history)
- **S7.2** — Follow-up handler (coreference resolution + RAG)

## Target Location
- `src/routers/chat.py` — FastAPI router
- `src/schemas/api/chat.py` — Request/response Pydantic models

## Functional Requirements

### FR-1: Chat Request/Response Models
- **What**: Pydantic models for chat API input and output
- **Inputs**:
  - `ChatRequest`:
    - `session_id: str` — conversation session ID (required, min 1 char, max 128 chars)
    - `query: str` — user's message (required, min 1 char, max 500 chars)
    - `stream: bool = True` — streaming SSE or JSON response
    - `top_k: int | None` — number of sources to retrieve (optional, gt 0)
    - `categories: list[str] | None` — arXiv category filter (optional)
    - `temperature: float | None` — LLM temperature (optional, 0.0-2.0)
  - `ChatResponse`:
    - `answer: str` — citation-backed answer
    - `sources: list[SourceReference]` — cited papers with arXiv links
    - `session_id: str` — echoed session ID
    - `is_follow_up: bool` — whether query was detected as follow-up
    - `rewritten_query: str | None` — rewritten query if follow-up detected
    - `query: str` — original user query
- **Edge cases**: Empty session_id, empty query, invalid temperature

### FR-2: Chat Endpoint (POST /api/v1/chat)
- **What**: Main chat endpoint supporting both streaming SSE and JSON responses
- **Streaming mode** (`stream=True`, default):
  - Returns `StreamingResponse` with `text/event-stream` media type
  - SSE events:
    - `event: metadata` / `data: {"session_id": "...", "is_follow_up": bool, "rewritten_query": "..."}` — sent first
    - `event: token` / `data: {"text": "..."}` — one per token
    - `event: sources` / `data: [SourceReference, ...]` — after all tokens
    - `event: done` / `data: {}` — signals completion
    - `event: error` / `data: {"detail": "..."}` — on failure
  - Uses `FollowUpHandler.handle_stream()` for streaming tokens
  - Stores conversation history automatically (handled by FollowUpHandler)
- **JSON mode** (`stream=False`):
  - Returns `ChatResponse` JSON
  - Uses `FollowUpHandler.handle()` for non-streaming
- **Error handling**:
  - 400: Invalid request (empty query, bad params)
  - 503: Service unavailable (LLM down, RAG pipeline failure)
- **Edge cases**: First message in session (no history), Redis down (graceful degradation)

### FR-3: Session Management Endpoints
- **What**: Endpoints for managing chat sessions
- `GET /api/v1/chat/sessions/{session_id}/history` — retrieve conversation history
  - Returns list of messages with roles, content, timestamps
  - Returns empty list if session doesn't exist
- `DELETE /api/v1/chat/sessions/{session_id}` — clear a session
  - Returns 200 with success status
  - Returns 200 even if session didn't exist (idempotent)
- **Edge cases**: Non-existent session, Redis unavailable

## Tangible Outcomes
- [ ] `POST /api/v1/chat` accepts ChatRequest and returns streaming SSE or JSON
- [ ] Streaming mode sends metadata, token, sources, done events in correct order
- [ ] JSON mode returns ChatResponse with answer, sources, session_id, follow-up info
- [ ] Follow-up questions are detected and rewritten with context
- [ ] Every response includes paper citations with arXiv links
- [ ] `GET /api/v1/chat/sessions/{session_id}/history` returns conversation history
- [ ] `DELETE /api/v1/chat/sessions/{session_id}` clears session
- [ ] Graceful degradation when Redis is unavailable (chat still works, no memory)
- [ ] Router registered in main.py at `/api/v1` prefix

## Test-Driven Requirements

### Tests to Write First
1. `test_chat_request_validation`: Validate ChatRequest field constraints
2. `test_chat_response_model`: Verify ChatResponse serialization
3. `test_chat_endpoint_json_mode`: POST /chat with stream=False returns ChatResponse
4. `test_chat_endpoint_streaming`: POST /chat with stream=True returns SSE events
5. `test_chat_follow_up_detection`: Follow-up query triggers rewrite
6. `test_chat_first_message`: First message in new session works (no history)
7. `test_chat_session_history`: GET /chat/sessions/{id}/history returns messages
8. `test_chat_session_clear`: DELETE /chat/sessions/{id} clears session
9. `test_chat_empty_query_rejected`: Empty query returns 422
10. `test_chat_service_error_handling`: 503 on RAG pipeline failure
11. `test_chat_no_memory_graceful`: Chat works when ConversationMemory is None
12. `test_chat_streaming_metadata_event`: First SSE event is metadata with follow-up info

### Mocking Strategy
- Mock `FollowUpHandler` (handle and handle_stream methods)
- Mock `ConversationMemory` (get_history, add_message, clear_session)
- Use `httpx.AsyncClient` with `ASGITransport` for endpoint tests
- Mock `RAGChain` and `LLMProvider` via dependency overrides

### Coverage
- All public endpoints tested
- Streaming and JSON modes tested
- Follow-up detection path tested
- Error handling paths tested
- Edge cases: empty session, Redis down, first message
