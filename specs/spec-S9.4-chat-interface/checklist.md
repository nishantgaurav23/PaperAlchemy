# Checklist -- Spec S9.4: Chat Interface (Streaming)

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & navigation) is "done"
- [x] Create target files/directories (`frontend/src/app/chat/`, `frontend/src/components/chat/`, `frontend/src/lib/api/chat.ts`, `frontend/src/types/chat.ts`)

## Phase 2: Tests First (TDD)
- [x] Create test files for all chat components
- [x] Write failing tests for FR-1 (Message Input)
- [x] Write failing tests for FR-2 (Message History)
- [x] Write failing tests for FR-3 (Streaming Response Display)
- [x] Write failing tests for FR-4 (Citation Rendering)
- [x] Write failing tests for FR-5 (Session Management)
- [x] Write failing tests for FR-6 (Welcome State)
- [x] Write failing tests for FR-7 (Error Handling)
- [x] Write failing tests for FR-8 (Mock SSE Layer)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement chat types (`frontend/src/types/chat.ts`)
- [x] Implement mock SSE API layer (`frontend/src/lib/api/chat.ts`) -- pass FR-8 tests
- [x] Implement MessageInput component -- pass FR-1 tests
- [x] Implement MessageBubble component -- pass FR-2 tests
- [x] Implement streaming display + typing indicator -- pass FR-3 tests
- [x] Implement CitationBadge + SourceCard components -- pass FR-4 tests
- [x] Implement session management (useChat hook) -- pass FR-5 tests
- [x] Implement WelcomeState component -- pass FR-6 tests
- [x] Implement error handling + retry -- pass FR-7 tests
- [x] Implement chat page (`frontend/src/app/chat/page.tsx`) -- integrate all
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add chat route to sidebar navigation (nav-items.ts) — already present
- [x] Verify page accessible at `/chat`
- [x] Run lint (`pnpm lint`) — clean
- [x] Run full test suite (`pnpm test`) — 26 files, 157 tests passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] ArXiv links open in new tab with rel="noopener noreferrer"
- [x] Responsive on mobile and desktop (Tailwind responsive classes)
- [x] Update roadmap.md status to "done"
