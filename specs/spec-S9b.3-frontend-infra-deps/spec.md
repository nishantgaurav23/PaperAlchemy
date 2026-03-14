# Spec S9b.3 — Frontend Infrastructure Dependencies

## Overview
Add frontend npm packages required by P13–P23 feature phases. These are foundational libraries that multiple downstream specs depend on: markdown rendering for chat, animations, global state management, toast notifications, command palette, and form validation. This spec also ensures all new dependencies have proper Vitest mocks so downstream specs can test without side effects.

## Dependencies
- **S9.1** (Next.js project scaffold) — `done` ✅

## Target Location
- `frontend/package.json` — new dependencies
- `frontend/src/test/` — Vitest mocks/setup for new deps
- `frontend/vitest.config.ts` — updated mock config if needed

## Functional Requirements

### FR-1: Markdown Rendering Dependencies
- **What**: Install `react-markdown`, `remark-gfm`, `rehype-highlight` for rendering markdown in chat messages (needed by S13.3 chat UX polish)
- **Inputs**: N/A (package installation)
- **Outputs**: Packages available in `node_modules`, importable in components
- **Edge cases**: Ensure compatibility with React 19 and Next.js 16

### FR-2: Animation Library
- **What**: Install `framer-motion` for page transitions, micro-interactions, and animated components (needed by S13.6 micro-interactions)
- **Inputs**: N/A
- **Outputs**: `framer-motion` importable, works with React 19 + Next.js 16 App Router
- **Edge cases**: Server component compatibility (framer-motion requires "use client")

### FR-3: Global State Management
- **What**: Install `zustand` for lightweight global state (auth state, notifications, UI preferences) (needed by S9b.4 auth infra, S13.5 navigation)
- **Inputs**: N/A
- **Outputs**: `zustand` importable, can create stores
- **Edge cases**: SSR hydration — zustand stores must handle server/client mismatch

### FR-4: Toast Notifications
- **What**: Install `sonner` for toast notifications (success/error/info feedback) (needed by S9b.4 auth, S13.6 micro-interactions)
- **Inputs**: N/A
- **Outputs**: `sonner` importable, `<Toaster />` component available
- **Edge cases**: Must work with Next.js App Router (client component)

### FR-5: Command Palette
- **What**: Install `cmdk` for Cmd+K command palette (needed by S13.5 sidebar navigation)
- **Inputs**: N/A
- **Outputs**: `cmdk` importable, `<Command />` component available
- **Edge cases**: Keyboard event handling, focus management, accessibility

### FR-6: Form Validation
- **What**: Install `react-hook-form` and `zod` (+ `@hookform/resolvers`) for type-safe form validation (needed by S9b.4 auth forms, P14 social features)
- **Inputs**: N/A
- **Outputs**: `useForm`, `zodResolver` importable
- **Edge cases**: Zod may already be a dependency — check before adding; ensure resolver compatibility

### FR-7: Vitest Mock Setup
- **What**: Create Vitest mocks for new dependencies so downstream tests don't break on import
- **Inputs**: New packages that need mocking (framer-motion, sonner, cmdk)
- **Outputs**: Mock files in `frontend/src/test/mocks/` or vitest setup, tests can import components using these deps without errors
- **Edge cases**: react-markdown and rehype/remark plugins need proper mock (they transform children)

## Tangible Outcomes
- [ ] `react-markdown`, `remark-gfm`, `rehype-highlight` in package.json dependencies
- [ ] `framer-motion` in package.json dependencies
- [ ] `zustand` in package.json dependencies
- [ ] `sonner` in package.json dependencies
- [ ] `cmdk` in package.json dependencies
- [ ] `react-hook-form`, `@hookform/resolvers` in package.json dependencies
- [ ] `zod` in package.json dependencies (if not already present)
- [ ] All packages install cleanly (`pnpm install` succeeds)
- [ ] Vitest mocks exist for framer-motion, sonner, cmdk
- [ ] A smoke test file verifies all packages are importable
- [ ] `pnpm test` passes with no regressions

## Test-Driven Requirements

### Tests to Write First
1. `test_deps_importable`: Verify each new package can be imported without errors
2. `test_react_markdown_renders`: Verify react-markdown renders basic markdown to HTML
3. `test_zustand_store_creates`: Verify a zustand store can be created and read
4. `test_framer_motion_mock`: Verify framer-motion mock works in test environment
5. `test_sonner_mock`: Verify sonner toast mock works in test environment
6. `test_react_hook_form_with_zod`: Verify useForm + zodResolver integration works
7. `test_existing_tests_pass`: Verify no regressions in existing test suite

### Mocking Strategy
- `framer-motion`: Mock `motion` components as passthrough divs, mock `AnimatePresence`
- `sonner`: Mock `toast` function and `<Toaster />` component
- `cmdk`: Mock `<Command />` components as passthrough elements
- `react-markdown`: Can render normally in jsdom (text-based), or mock if heavy

### Coverage
- All new packages verified importable
- Integration between react-hook-form + zod verified
- Existing tests unbroken
