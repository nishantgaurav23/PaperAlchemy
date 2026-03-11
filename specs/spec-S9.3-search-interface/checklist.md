# Checklist -- Spec S9.3: Search Interface

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (layout & navigation) is "done"
- [x] Create target directories: `frontend/src/app/search/`, `frontend/src/components/search/`, `frontend/src/types/`
- [x] Install any needed shadcn/ui components (Input, Select, Badge, Skeleton, etc.)

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/types/paper.ts` — Paper type definitions
- [x] Create `frontend/src/lib/api/search.ts` — Search API function (mock-ready)
- [x] Create test files for all components
- [x] Write failing tests for FR-1 through FR-7
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement Paper types (`types/paper.ts`)
- [x] Implement search API client (`lib/api/search.ts`)
- [x] Implement SearchBar component — pass tests (FR-1)
- [x] Implement CategoryFilter component — pass tests (FR-2)
- [x] Implement SortSelect component — pass tests (FR-3)
- [x] Implement PaperCard component — pass tests (FR-4)
- [x] Implement Pagination component — pass tests (FR-5)
- [x] Implement loading/empty/error states — pass tests (FR-6)
- [x] Implement search page with URL state — pass tests (FR-7)
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire search page into app router (`app/search/page.tsx`)
- [x] Verify navigation from sidebar works
- [x] Run lint (`pnpm lint`)
- [x] Run full test suite (`pnpm test`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Responsive layout verified
- [x] Update roadmap.md status to "done"
