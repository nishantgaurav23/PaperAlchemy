# Checklist -- Spec S7.3: Chat API

## Phase 1: Setup & Dependencies
- [x] Verify S7.1 (conversation-memory) is "done"
- [x] Verify S7.2 (follow-up-handler) is "done"
- [x] Create `src/schemas/api/chat.py`
- [x] Create `src/routers/chat.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_chat_api.py`
- [x] Write failing tests for ChatRequest/ChatResponse models
- [x] Write failing tests for POST /api/v1/chat (JSON mode)
- [x] Write failing tests for POST /api/v1/chat (streaming mode)
- [x] Write failing tests for follow-up detection in chat
- [x] Write failing tests for GET /api/v1/chat/sessions/{id}/history
- [x] Write failing tests for DELETE /api/v1/chat/sessions/{id}
- [x] Write failing tests for error handling (empty query, service failure)
- [x] Write failing tests for graceful degradation (no Redis)
- [x] Run tests -- expect failures (Red) — 32 FAILED

## Phase 3: Implementation
- [x] Implement ChatRequest and ChatResponse schemas (FR-1)
- [x] Implement POST /api/v1/chat endpoint (FR-2)
- [x] Implement streaming SSE mode with metadata/token/sources/done events
- [x] Implement JSON response mode
- [x] Implement GET /api/v1/chat/sessions/{id}/history (FR-3)
- [x] Implement DELETE /api/v1/chat/sessions/{id} (FR-3)
- [x] Run tests -- expect pass (Green) — 32 passed
- [x] Refactor if needed

## Phase 4: Integration
- [x] Register chat_router in src/main.py
- [x] Export from src/routers/__init__.py
- [x] Run lint (ruff check) — clean
- [x] Run full test suite — 886 passed (9 pre-existing failures from S6.8)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: notebooks/specs/S7.3_chat_api.ipynb
- [x] Append summary to docs/spec-summaries.md
- [x] Update roadmap.md status to "done"
