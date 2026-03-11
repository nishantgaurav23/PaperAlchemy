# Checklist -- Spec S9.1: Next.js Project Setup

## Phase 1: Setup & Dependencies
- [x] Verify no dependency specs required (S9.1 has no deps)
- [x] Create `frontend/` directory with Next.js 15 scaffold

## Phase 2: Tests First (TDD)
- [x] Configure Vitest + React Testing Library
- [x] Write failing test: home page renders
- [x] Write failing test: theme toggle
- [x] Write failing test: Button component variants
- [x] Write failing tests: API client (GET, POST, error handling)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] FR-1: Next.js 15 project init (App Router, TypeScript, Tailwind, pnpm, src/)
- [x] FR-2: Tailwind CSS v4 + shadcn/ui setup (new-york style, Button component)
- [x] FR-3: Dark mode (next-themes, ThemeProvider, suppressHydrationWarning)
- [x] FR-4: ESLint configuration (Next.js recommended)
- [x] FR-5: Vitest setup (jsdom, path aliases, React Testing Library)
- [x] FR-6: API client (typed fetch wrapper, error handling, base URL config)
- [x] FR-7: TypeScript strict mode (strict: true, path aliases @/)
- [x] Run tests -- expect pass (Green) -- 16/16 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify `pnpm dev` starts on port 3000
- [x] Verify `pnpm build` completes without errors
- [x] Verify `pnpm lint` passes with zero errors
- [x] Verify `pnpm test` passes all tests (16/16)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets (API URL from env var)
- [x] Update roadmap.md status to "done"
