# Checklist -- Spec S7.2: Follow-up Handler

## Phase 1: Setup & Dependencies
- [x] Verify S7.1 (Conversation Memory) is "done"
- [x] Verify S5.2 (RAG Chain) is "done"
- [x] Create target file: `src/services/chat/follow_up.py`

## Phase 2: Tests First (TDD)
- [x] Create test file: `tests/unit/test_follow_up_handler.py`
- [x] Write failing tests for FR-1 (query rewriting with coreference resolution)
- [x] Write failing tests for FR-2 (follow-up detection heuristics)
- [x] Write failing tests for FR-3 (follow-up orchestration)
- [x] Run tests -- expect failures (Red) — 30 failing

## Phase 3: Implementation
- [x] Implement `FollowUpResult` Pydantic model
- [x] Implement `is_follow_up()` heuristic function (FR-2)
- [x] Implement `rewrite_query()` with LLM coreference resolution (FR-1)
- [x] Implement `FollowUpHandler.handle()` orchestration (FR-3)
- [x] Implement `FollowUpHandler.handle_stream()` streaming mode (FR-3)
- [x] Run tests -- expect pass (Green) — 30 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Update `src/services/chat/__init__.py` exports
- [x] Add factory function + DI wiring (`FollowUpHandlerDep`) in `src/dependency.py`
- [x] Run lint — all checks passed
- [x] Run full test suite — 854 passed (9 pre-existing failures in S6.8)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S7.2_follow_up.ipynb`
- [x] Update roadmap.md status to "done"
