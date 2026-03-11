# Checklist -- Spec S9.8: Trends Dashboard

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & Navigation) is done
- [x] Install recharts dependency via pnpm
- [x] Create target directories: `app/dashboard/`, `components/dashboard/`, `lib/api/dashboard.ts`, `types/dashboard.ts`

## Phase 2: Tests First (TDD)
- [x] Create `types/dashboard.ts` with TypeScript interfaces
- [x] Create mock data fixtures for dashboard
- [x] Write failing test: `components/dashboard/stats-cards.test.tsx`
- [x] Write failing test: `components/dashboard/category-chart.test.tsx`
- [x] Write failing test: `components/dashboard/timeline-chart.test.tsx`
- [x] Write failing test: `components/dashboard/hot-papers.test.tsx`
- [x] Write failing test: `components/dashboard/trending-topics.test.tsx`
- [x] Write failing test: `app/dashboard/dashboard-page.test.tsx`
- [x] Write failing test: `lib/api/dashboard.test.ts`
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `lib/api/dashboard.ts` — API client with mock fallback
- [x] Implement `components/dashboard/stats-cards.tsx` — pass tests
- [x] Implement `components/dashboard/category-chart.tsx` — pass tests
- [x] Implement `components/dashboard/timeline-chart.tsx` — pass tests
- [x] Implement `components/dashboard/hot-papers.tsx` — pass tests
- [x] Implement `components/dashboard/trending-topics.tsx` — pass tests
- [x] Implement `app/dashboard/page.tsx` — dashboard page shell
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify navigation sidebar links to `/dashboard`
- [x] Verify dark mode works on all chart components
- [x] Run lint (eslint)
- [x] Run full test suite (vitest run)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Responsive layout works on mobile/tablet/desktop
- [x] Update roadmap.md status to "done"
