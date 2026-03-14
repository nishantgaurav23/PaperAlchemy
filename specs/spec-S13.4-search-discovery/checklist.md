# Checklist -- Spec S13.4: Search & Discovery Polish

## Phase 1: Setup & Dependencies
- [x] Verify S9.3 (Search Interface) is "done"
- [x] Review existing search components in `frontend/src/components/search/`
- [x] Identify new files to create and existing files to modify

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/lib/hooks/use-recent-searches.test.ts` — recent searches hook tests (11 tests)
- [x] Create `frontend/src/components/search/filter-pills.test.tsx` — filter pills component tests (8 tests)
- [x] Extend `frontend/src/components/search/paper-card.test.tsx` — category chips, bookmark, hover preview (15 tests total)
- [x] Extend `frontend/src/components/search/search-bar.test.tsx` — recent searches dropdown (12 tests total)
- [x] Extend `frontend/src/components/search/search-results.test.tsx` — shimmer skeletons, staggered animation, enhanced empty state (11 tests total)
- [x] Run tests — expect failures (Red) → then pass (Green)

## Phase 3: Implementation
- [x] Implement `useRecentSearches` hook (localStorage, max 10, LIFO) — pass tests
- [x] Implement `FilterPills` component (active filters as removable pills) — pass tests
- [x] Upgrade `PaperCard` — category color chips, bookmark icon, hover abstract preview — pass tests
- [x] Upgrade `SearchBar` — recent searches dropdown on focus — pass tests
- [x] Upgrade `SearchResults` — shimmer skeletons, staggered fade-in, enhanced empty state — pass tests
- [x] Add shimmer + fade-in-up animation CSS to globals.css
- [x] Run tests — all 75 pass (Green)
- [x] Refactor: clean code, consistent patterns

## Phase 4: Integration
- [x] Update search page to wire FilterPills and pass filter removal handlers
- [x] Update search barrel export (`index.ts`) with new FilterPills component
- [x] Run lint (`cd frontend && pnpm lint`) — 0 new errors (2 pre-existing in unrelated file)
- [x] Run full search test suite — 75/75 passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Existing search tests still pass
- [x] Update roadmap.md status to "done"
