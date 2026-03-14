# Checklist -- Spec S13.6: Animations & Micro-Interactions

## Phase 1: Setup & Dependencies
- [x] Verify S9.1 (Next.js setup) is done
- [x] Verify S9b.3 (framer-motion, sonner installed) is done
- [x] Create animation components directory: `frontend/src/components/animations/`

## Phase 2: Tests First (TDD)
- [x] Create test files for all animation components
- [x] Write failing tests for FR-1 (PageTransition)
- [x] Write failing tests for FR-2 (Skeleton shimmer loaders)
- [x] Write failing tests for FR-3 (Button press feedback)
- [x] Write failing tests for FR-4 (HoverCardPreview)
- [x] Write failing tests for FR-5 (Toast notifications)
- [x] Write failing tests for FR-6 (ProgressIndicator)
- [x] Write failing tests for FR-7 (ScrollFadeIn)
- [x] Write failing tests for FR-8 (AnimatedCounter)
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement FR-1 (PageTransition) — pass tests
- [x] Implement FR-2 (Skeleton loaders) — pass tests
- [x] Implement FR-3 (Button press feedback) — pass tests
- [x] Implement FR-4 (HoverCardPreview) — pass tests
- [x] Implement FR-5 (Toast notifications) — pass tests
- [x] Implement FR-6 (ProgressIndicator) — pass tests
- [x] Implement FR-7 (ScrollFadeIn) — pass tests
- [x] Implement FR-8 (AnimatedCounter) — pass tests
- [x] Run all tests — expect pass (Green) — 57/57 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add PageTransition wrapper to landing page
- [x] Skeleton loaders available (SkeletonCard, SkeletonText, SkeletonChart, SkeletonList)
- [x] PressableButton component available for interactive buttons
- [x] Add hover card previews component (HoverCardPreview)
- [x] Configure toast provider in AppShell layout
- [x] ProgressIndicator component available for upload/search
- [x] Add scroll fade-in to dashboard cards (staggered 100ms delays)
- [x] Add animated counters to landing page stats (replaced useCountUp)
- [x] Run lint (`cd frontend && pnpm lint`) — 0 errors
- [x] Run full test suite — all animation + dashboard + landing tests pass
- [x] Added IntersectionObserver polyfill to test setup

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] All animations respect prefers-reduced-motion (via globals.css media query)
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to docs/spec-summaries.md
