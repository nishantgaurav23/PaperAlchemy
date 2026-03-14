# Checklist — Spec S13.2: Design System Overhaul

## Phase 1: Setup & Dependencies
- [x] Verify S9.1 (Next.js setup) is "done"
- [x] Add Inter + Plus Jakarta Sans fonts to Next.js layout
- [x] Identify all existing CSS custom properties in globals.css

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/app/design-system.test.tsx`
- [x] Write tests for color palette tokens (light + dark)
- [x] Write tests for glassmorphism utility classes
- [x] Write tests for typography (font families, scale)
- [x] Write tests for gradient utilities
- [x] Write tests for elevation/shadow classes
- [x] Write tests for focus ring styles
- [x] Write tests for transition classes
- [x] Write tests for prefers-reduced-motion
- [x] Write regression tests for existing Button, Card, Input
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] FR-1: Update color palette in globals.css (indigo/violet primary, light + dark)
- [x] FR-2: Add glassmorphism utility classes
- [x] FR-3: Document 8px spacing grid usage (Tailwind defaults align)
- [x] FR-4: Configure Inter + Plus Jakarta Sans typography scale
- [x] FR-5: Add gradient accent utilities
- [x] FR-6: Define elevation/shadow system
- [x] FR-7: Add focus ring + 200ms transition defaults + prefers-reduced-motion
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify all existing pages render with updated design tokens
- [x] Run lint: `cd frontend && pnpm lint` — 0 errors
- [x] Run full test suite: `cd frontend && pnpm test` — 28/28 pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
