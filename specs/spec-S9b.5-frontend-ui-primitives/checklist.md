# Checklist — Spec S9b.5: Missing UI Primitives

## Phase 1: Setup & Dependencies
- [x] Verify S9b.3 is "done"
- [x] Install Radix UI packages: `@radix-ui/react-dialog`, `@radix-ui/react-dropdown-menu`, `@radix-ui/react-popover`, `@radix-ui/react-tabs`, `@radix-ui/react-checkbox`, `@radix-ui/react-tooltip`, `@radix-ui/react-avatar`
- [x] Verify `cmdk` and `sonner` already installed

## Phase 2: Tests First (TDD)
- [x] Create `dialog.test.tsx` — trigger, open, close, accessibility
- [x] Create `dropdown-menu.test.tsx` — trigger, items, keyboard nav
- [x] Create `popover.test.tsx` — trigger, toggle content
- [x] Create `tabs.test.tsx` — tab switching, content panels
- [x] Create `textarea.test.tsx` — render, disabled, placeholder
- [x] Create `checkbox.test.tsx` — toggle, disabled, indeterminate
- [x] Create `tooltip.test.tsx` — trigger, show on hover/focus
- [x] Create `avatar.test.tsx` — image, fallback
- [x] Create `sonner.test.tsx` — Toaster renders
- [x] Create `command.test.tsx` — input, items, filtering
- [x] Create `sheet.test.tsx` — trigger, open, close, slide direction
- [x] Run tests — expect failures (Red) ✅ All 11 failed

## Phase 3: Implementation
- [x] Implement `dialog.tsx` — pass tests
- [x] Implement `dropdown-menu.tsx` — pass tests
- [x] Implement `popover.tsx` — pass tests
- [x] Implement `tabs.tsx` — pass tests
- [x] Implement `textarea.tsx` — pass tests
- [x] Implement `checkbox.tsx` — pass tests
- [x] Implement `tooltip.tsx` — pass tests
- [x] Implement `avatar.tsx` — pass tests
- [x] Implement `sonner.tsx` — pass tests
- [x] Implement `command.tsx` — pass tests
- [x] Implement `sheet.tsx` — pass tests
- [x] Run tests — expect pass (Green) ✅ 43 tests pass
- [x] Refactor if needed — clean, no refactoring needed

## Phase 4: Integration
- [x] Verify all components export correctly (import in a test)
- [x] Run lint: `cd frontend && pnpm lint` ✅ 0 errors
- [x] Run full test suite: `cd frontend && pnpm test` ✅ All UI tests pass

## Phase 5: Verification
- [x] All 11 components created with co-located tests
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
