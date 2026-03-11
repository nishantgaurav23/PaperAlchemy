# Checklist -- Spec S5.3: Streaming Responses (SSE)

## Phase 1: Setup & Dependencies
- [x] Verify S5.2 (RAG Chain) is "done"
- [x] Create target files: `src/routers/ask.py`, `src/schemas/api/ask.py`

## Phase 2: Tests First (TDD)
- [x] Create test file: `tests/unit/test_ask_router.py`
- [x] Write failing tests for SSE streaming (token, sources, done events)
- [x] Write failing tests for non-streaming JSON response
- [x] Write failing tests for validation (empty query → 422)
- [x] Write failing tests for error handling (LLM error → error event)
- [x] Run tests — expect failures (Red) — 22 FAILED

## Phase 3: Implementation
- [x] Implement `AskRequest` / `AskResponse` schemas (`src/schemas/api/ask.py`)
- [x] Implement SSE streaming endpoint (`src/routers/ask.py`)
- [x] Implement non-streaming fallback
- [x] Implement SSE event formatting (token, sources, done, error)
- [x] Implement client disconnect handling
- [x] Run tests — expect pass (Green) — 22/22 passed
- [x] Refactor if needed (lint fixes)

## Phase 4: Integration
- [x] Register ask router in `src/main.py`
- [x] Update `src/routers/__init__.py` exports
- [x] Update `src/schemas/api/__init__.py` exports
- [x] Add `RAGChainDep` usage (already exists in dependency.py)
- [x] Run lint (`ruff check`) — All checks passed
- [x] Run full test suite — 589 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S5.3_streaming.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
