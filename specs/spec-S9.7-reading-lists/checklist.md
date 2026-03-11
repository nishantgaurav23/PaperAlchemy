# Checklist -- Spec S9.7: Reading Lists & Collections

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & navigation) is "done"
- [x] Create target directories: `frontend/src/app/collections/`, `frontend/src/app/collections/[id]/`, `frontend/src/components/collections/`

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/types/collection.ts` — Collection types
- [x] Create `frontend/src/lib/collections.test.ts` — localStorage CRUD tests (24 tests)
- [x] Create `frontend/src/components/collections/collection-card.test.tsx` — card component tests (7 tests)
- [x] Create `frontend/src/components/collections/add-to-collection.test.tsx` — add popover tests (5 tests)
- [x] Create `frontend/src/components/collections/paper-list.test.tsx` — drag-and-drop paper list tests (8 tests)
- [x] Create `frontend/src/app/collections/page.test.tsx` — collections list page tests (5 tests)
- [x] Create `frontend/src/app/collections/[id]/page.test.tsx` — collection detail page tests (8 tests)
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement `frontend/src/lib/collections.ts` — localStorage storage layer
- [x] Implement `frontend/src/components/collections/collection-card.tsx` — collection card
- [x] Implement `frontend/src/components/collections/create-collection-dialog.tsx` — create/edit dialog
- [x] Implement `frontend/src/components/collections/add-to-collection.tsx` — add paper popover
- [x] Implement `frontend/src/components/collections/paper-list.tsx` — paper list with drag-and-drop
- [x] Implement `frontend/src/components/collections/index.ts` — barrel export
- [x] Implement `frontend/src/app/collections/page.tsx` — collections list page
- [x] Implement `frontend/src/app/collections/[id]/page.tsx` — collection detail page
- [x] Run tests — expect pass (Green) — 287/287 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify `/collections` nav link works from sidebar (already in nav-items.ts)
- [x] Run lint (`pnpm lint`) — clean
- [x] Run full test suite (`pnpm test`) — 42 files, 287 tests pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
