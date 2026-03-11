# Checklist -- Spec S7.1: Conversation Memory

## Phase 1: Setup & Dependencies
- [x] Verify S5.4 (Response Caching) is "done"
- [x] Create `src/services/chat/` directory
- [x] Create `src/services/chat/__init__.py`
- [x] Create `src/services/chat/memory.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_conversation_memory.py`
- [x] Write ChatMessage model tests (serialization, validation)
- [x] Write add_message tests (success, empty content, redis error, metadata)
- [x] Write sliding window tests (trims at limit, no trim under limit)
- [x] Write get_history tests (success, empty, limit, redis error, TTL refresh)
- [x] Write clear_session tests (success, not found, redis error)
- [x] Write list_sessions tests (success, redis error)
- [x] Write TTL refresh tests
- [x] Write factory tests (success, no redis)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement ChatMessage Pydantic model
- [x] Implement ConversationMemory class -- pass tests
- [x] Implement add_message with sliding window
- [x] Implement get_history with optional limit
- [x] Implement clear_session
- [x] Implement list_sessions
- [x] Implement TTL refresh on interactions
- [x] Implement make_conversation_memory factory
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire ConversationMemoryDep into src/dependency.py
- [x] Export from src/services/chat/__init__.py
- [x] Run lint (ruff check)
- [x] Run full test suite (801 passed, 9 skipped)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: notebooks/specs/S7.1_chat_memory.ipynb
- [x] Update roadmap.md status to "done"
