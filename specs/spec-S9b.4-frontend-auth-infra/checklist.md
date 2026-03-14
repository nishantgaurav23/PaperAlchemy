# Checklist -- Spec S9b.4: Frontend Auth Infrastructure

## Phase 1: Setup & Dependencies
- [x] Verify S9b.3 is "done" (zustand, react-hook-form, zod, sonner available)
- [x] Create target directories: `frontend/src/lib/auth/`, `frontend/src/components/auth/`, `frontend/src/app/(auth)/`

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/types/auth.test.ts` — Zod schema validation tests
- [x] Create `frontend/src/lib/auth/store.test.ts` — auth store tests
- [x] Create `frontend/src/lib/auth/interceptor.test.ts` — API interceptor tests
- [x] Create `frontend/src/components/auth/protected-route.test.tsx` — route guard tests
- [x] Create `frontend/src/app/(auth)/login/page.test.tsx` — login form tests
- [x] Create `frontend/src/app/(auth)/signup/page.test.tsx` — signup form tests
- [x] Create `frontend/src/app/(auth)/forgot-password/page.test.tsx` — forgot password tests
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement `frontend/src/types/auth.ts` — types + Zod schemas → pass type tests
- [x] Implement `frontend/src/lib/auth/store.ts` — zustand auth store → pass store tests
- [x] Implement `frontend/src/lib/auth/interceptor.ts` — API client auth interceptor → pass interceptor tests
- [x] Implement `frontend/src/components/auth/protected-route.tsx` — route guard → pass protected-route tests
- [x] Implement `frontend/src/app/(auth)/login/page.tsx` — login page → pass login tests
- [x] Implement `frontend/src/app/(auth)/signup/page.tsx` — signup page → pass signup tests
- [x] Implement `frontend/src/app/(auth)/forgot-password/page.tsx` — forgot password page → pass forgot-password tests
- [x] Run all tests — expect pass (Green) — 56/56 passing
- [x] Refactor if needed (added noValidate to forms for Zod-only validation)

## Phase 4: Integration
- [x] Wire auth interceptor into existing apiClient (or document how to opt-in) — exported as opt-in via `createAuthenticatedFetch()`, will wire into apiClient when S14.1 backend auth is implemented
- [x] Add login/signup links to app header/sidebar — added Sign in/Sign out to header
- [x] Run lint (`cd frontend && pnpm lint`) — 0 errors (1 pre-existing warning)
- [x] Run full test suite (`cd frontend && pnpm test`) — 56/56 auth tests pass, 6 pre-existing failures unrelated to auth

## Phase 5: Verification
- [x] All tangible outcomes checked — all files exist, 56/56 tests pass
- [x] No hardcoded secrets or tokens — tokens in sessionStorage only, no secrets in code
- [x] Auth pages accessible at /login, /signup, /forgot-password
- [x] ProtectedRoute testable via wrapping any page
- [x] Update roadmap.md status to "done"
