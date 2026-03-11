# Checklist -- Spec S9.2: Layout & Navigation

## Phase 1: Setup & Dependencies
- [x] Verify S9.1 is "done"
- [x] Install additional shadcn/ui components if needed (sheet for mobile drawer) — used custom drawer instead
- [x] Create `frontend/src/components/layout/` directory

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/components/layout/app-shell.test.tsx`
- [x] Create `frontend/src/components/layout/sidebar.test.tsx`
- [x] Create `frontend/src/components/layout/header.test.tsx`
- [x] Create `frontend/src/components/layout/breadcrumbs.test.tsx`
- [x] Create `frontend/src/components/layout/mobile-nav.test.tsx`
- [x] Create `frontend/src/components/layout/sidebar-nav-item.test.tsx`
- [x] Write failing tests for all FRs (Red)
- [x] Run tests — expect failures

## Phase 3: Implementation
- [x] Implement `sidebar-nav-item.tsx` — pass tests
- [x] Implement `sidebar.tsx` with collapse toggle — pass tests
- [x] Implement `breadcrumbs.tsx` — pass tests
- [x] Implement `header.tsx` with theme toggle integration — pass tests
- [x] Implement `mobile-nav.tsx` (drawer) — pass tests
- [x] Implement `app-shell.tsx` combining all pieces — pass tests
- [x] Update `layout.tsx` to use AppShell
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed (fixed lint: useState initializer instead of useEffect)

## Phase 4: Integration
- [x] Verify all routes render inside the shell
- [x] Verify sidebar active state matches current route
- [x] Verify localStorage persistence for collapse state
- [x] Run lint (`pnpm lint`) — passes
- [x] Run full test suite (`pnpm test`) — 53 tests pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Responsive behavior verified (mobile, tablet, desktop)
- [x] Accessibility checked (keyboard nav, aria labels)
- [x] Update roadmap.md status to "done"
