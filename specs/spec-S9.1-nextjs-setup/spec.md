# Spec S9.1 -- Next.js Project Setup

## Overview
Initialize the Next.js 15 frontend application with TypeScript, Tailwind CSS v4, and shadcn/ui component library. This is the foundation for all frontend specs in Phase 9. The setup includes App Router, pnpm package manager, ESLint, Vitest for testing, and dark mode support via next-themes.

## Dependencies
- None (can start in parallel from P3+)

## Target Location
- `frontend/` — entire Next.js application directory

## Functional Requirements

### FR-1: Next.js 15 Project Initialization
- **What**: Create a Next.js 15 project using `create-next-app` with TypeScript, Tailwind CSS, App Router, and `src/` directory
- **Inputs**: None (project scaffold)
- **Outputs**: Working Next.js 15 app in `frontend/` directory
- **Edge cases**: Ensure pnpm is used (not npm/yarn), ensure `src/` directory structure

### FR-2: Tailwind CSS v4 + shadcn/ui Setup
- **What**: Configure Tailwind CSS v4 with CSS variables for theming. Install and configure shadcn/ui with the "new-york" style variant.
- **Inputs**: None (configuration)
- **Outputs**: Working Tailwind + shadcn/ui with at least one component (Button) installed as proof
- **Edge cases**: Ensure CSS variables are set up for light/dark themes

### FR-3: Dark Mode Support
- **What**: Implement dark mode toggle using `next-themes` provider with system preference detection
- **Inputs**: User preference (light/dark/system)
- **Outputs**: Theme persists across page loads, smooth transitions
- **Edge cases**: SSR hydration mismatch prevention (suppressHydrationWarning)

### FR-4: ESLint Configuration
- **What**: Configure ESLint with Next.js recommended rules + Prettier integration
- **Inputs**: None (configuration)
- **Outputs**: `pnpm lint` runs successfully with zero errors
- **Edge cases**: Ensure no conflicts between ESLint and Prettier rules

### FR-5: Vitest Testing Setup
- **What**: Configure Vitest with React Testing Library for component testing, jsdom environment
- **Inputs**: None (configuration)
- **Outputs**: `pnpm test` runs successfully, sample test passes
- **Edge cases**: Ensure path aliases (@/) work in tests

### FR-6: API Client Foundation
- **What**: Create a typed API client utility for communicating with the FastAPI backend
- **Inputs**: Base URL (from env var `NEXT_PUBLIC_API_URL`, default `http://localhost:8000`)
- **Outputs**: `frontend/src/lib/api-client.ts` with typed fetch wrapper
- **Edge cases**: Error handling, timeout configuration, base URL configuration

### FR-7: TypeScript Strict Mode
- **What**: Enable strict TypeScript configuration
- **Inputs**: None (tsconfig)
- **Outputs**: `strict: true` in tsconfig.json, all files type-safe
- **Edge cases**: Path aliases configured (@/ -> src/)

## Tangible Outcomes
- [ ] `frontend/` directory exists with Next.js 15 app
- [ ] `pnpm dev` starts dev server on port 3000
- [ ] `pnpm build` completes without errors
- [ ] `pnpm lint` passes with zero errors
- [ ] `pnpm test` passes with at least one test
- [ ] Dark mode toggle works (light/dark/system)
- [ ] shadcn/ui Button component renders correctly
- [ ] API client exists at `frontend/src/lib/api-client.ts`
- [ ] TypeScript strict mode enabled
- [ ] Vitest configured with React Testing Library

## Test-Driven Requirements

### Tests to Write First
1. `test_home_page_renders`: Home page renders with expected heading/content
2. `test_theme_toggle`: Theme provider wraps the app, dark mode class toggleable
3. `test_button_component`: shadcn/ui Button renders with correct variants
4. `test_api_client_get`: API client makes GET requests with correct base URL
5. `test_api_client_post`: API client makes POST requests with JSON body
6. `test_api_client_error_handling`: API client handles HTTP errors gracefully

### Mocking Strategy
- Mock `fetch` globally for API client tests
- Use React Testing Library `render` for component tests
- Mock `next-themes` for theme toggle tests

### Coverage
- All public functions in api-client.ts tested
- Home page render test
- Theme toggle functionality tested
- Button component variant tests
