# Checklist — Spec S13.3: Chat UX Polish

## Phase 1: Setup & Dependencies
- [x] Verify S9.4 (chat interface) is done
- [x] Verify S9b.3 (frontend infra deps — react-markdown, framer-motion) is done
- [x] Confirm react-markdown, remark-gfm, rehype-highlight, framer-motion are installed

## Phase 2: Tests First (TDD)
- [x] Create/update test files for enhanced components
- [x] Write failing tests for FR-1 (markdown rendering)
- [x] Write failing tests for FR-2 (code block copy button)
- [x] Write failing tests for FR-3 (rich citation cards)
- [x] Write failing tests for FR-4 (follow-up suggestion chips)
- [x] Write failing tests for FR-5 (typing indicator animation)
- [x] Write failing tests for FR-6 (message timestamps)
- [x] Write failing tests for FR-7 (scroll-to-bottom)
- [x] Write failing tests for FR-8 (empty state)
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] FR-1: Enhance message-bubble with react-markdown + remark-gfm + rehype-highlight
- [x] FR-2: Add CodeBlock component with syntax highlighting + copy button
- [x] FR-3: Upgrade source-card with rich citation card design
- [x] FR-4: Add FollowUpChips component
- [x] FR-5: Enhance typing-indicator with framer-motion animations
- [x] FR-6: Add timestamp display to message-bubble (data-testid, relative formatting)
- [x] FR-7: Enhance scroll-to-bottom with framer-motion AnimatePresence animation
- [x] FR-8: Upgrade welcome-state with gradient empty state + example prompts
- [x] Run tests — expect pass (Green) — 60/60 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire enhanced components into chat page (FollowUpChips, AnimatePresence)
- [x] Update component barrel exports (index.ts)
- [x] Run lint (pnpm lint) — no errors in chat files
- [x] Run full test suite (pnpm vitest run src/components/chat/) — 9 files, 60 tests pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to docs/spec-summaries.md
