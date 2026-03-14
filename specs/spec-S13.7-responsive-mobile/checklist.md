# Checklist -- Spec S13.7: Mobile-First Responsive Polish

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & navigation) is "done"
- [x] Audit existing mobile-nav.tsx and app-shell.tsx for current mobile handling
- [x] Plan component structure for new mobile components

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/components/layout/bottom-nav.test.tsx`
- [x] Create `frontend/src/lib/hooks/use-swipe.test.ts`
- [x] Create `frontend/src/components/layout/pull-to-refresh.test.tsx`
- [x] Create `frontend/src/components/search/mobile-filter-sheet.test.tsx`
- [x] Write failing tests for bottom nav (FR-1)
- [x] Write failing tests for swipe gesture hook (FR-2)
- [x] Write failing tests for pull-to-refresh (FR-3)
- [x] Write failing tests for mobile filter sheet (FR-6)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement bottom-nav.tsx (FR-1) -- pass tests
- [x] Implement useSwipe hook (FR-2) -- pass tests
- [x] Implement pull-to-refresh component (FR-3) -- pass tests
- [x] Apply touch-optimized hit targets (FR-4) -- CSS/Tailwind utilities
- [x] Update page layouts for adaptive grids (FR-5)
- [x] Implement mobile-filter-sheet.tsx (FR-6) -- pass tests
- [x] Add responsive typography with clamp() (FR-7)
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire bottom-nav into app-shell.tsx (replace mobile-nav on mobile)
- [x] Add bottom padding to main content area for bottom nav clearance
- [x] Wire pull-to-refresh into search page
- [x] Wire swipe gestures into chat page (useSwipe hook available)
- [x] Wire mobile-filter-sheet into search page
- [x] Update globals.css with fluid typography tokens
- [x] Run lint (`cd frontend && pnpm lint`) -- 0 errors
- [x] Run full test suite (`cd frontend && pnpm test`) -- all S13.7 tests pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] No horizontal overflow at 320px viewport width
- [x] All existing tests still pass
- [x] Update roadmap.md status to "done"
