# Checklist — Spec S9b.3: Frontend Infrastructure Dependencies

## Phase 1: Setup & Dependencies
- [x] Verify S9.1 (Next.js project scaffold) is "done"
- [x] Audit current package.json for existing deps (avoid duplicates)

## Phase 2: Tests First (TDD)
- [x] Create test file: `frontend/src/lib/infra-deps.test.tsx`
- [x] Write import smoke tests for all new packages
- [x] Write react-markdown render test
- [x] Write zustand store creation test
- [x] Write react-hook-form + zod integration test
- [x] Write framer-motion mock test
- [x] Write sonner mock test
- [x] Run tests — expect failures (Red) ✅ confirmed

## Phase 3: Implementation
- [x] Install markdown deps: react-markdown, remark-gfm, rehype-highlight
- [x] Install framer-motion
- [x] Install zustand
- [x] Install sonner
- [x] Install cmdk
- [x] Install react-hook-form, @hookform/resolvers
- [x] Install zod
- [x] Add jsdom polyfills (ResizeObserver, scrollIntoView) in test setup
- [x] Run tests — expect pass (Green) ✅ 16/16 passing
- [x] Verify `pnpm install` is clean

## Phase 4: Integration
- [x] Run existing test suite — no regressions (47 passed files, 355 tests; 6 pre-existing failures unchanged)
- [x] Run lint (`pnpm lint`) — 0 errors, 1 pre-existing warning
- [x] Verify `pnpm build` — pre-existing type error in papers/page.tsx (not caused by S9b.3)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
