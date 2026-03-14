# Spec S9b.4 -- Frontend Auth Infrastructure

## Overview
Frontend authentication infrastructure providing auth context, token management, API client integration, route protection, and page shells for login/signup. This is the client-side foundation that all authenticated features (P14 community, P15 annotations, P17 collaboration, etc.) will build upon.

**Note**: This spec builds the **frontend auth infrastructure only**. The actual backend auth API (S14.1) is a separate spec. This spec uses typed stubs and mock endpoints — the backend will be wired in when S14.1 is implemented.

## Dependencies
- **S9b.3** (Frontend infrastructure dependencies) — provides zustand, react-hook-form, zod, sonner

## Target Location
- `frontend/src/lib/auth/` — auth store, token helpers, API interceptor
- `frontend/src/components/auth/` — ProtectedRoute, auth form components
- `frontend/src/app/(auth)/` — login, signup, forgot-password page shells
- `frontend/src/types/auth.ts` — auth types

## Functional Requirements

### FR-1: Auth Types
- **What**: TypeScript types for auth state and API payloads
- **Types**:
  - `User` — `id`, `email`, `name`, `avatar_url?`, `affiliation?`, `created_at`
  - `AuthState` — `user: User | null`, `token: string | null`, `isAuthenticated: boolean`, `isLoading: boolean`
  - `LoginRequest` — `email`, `password`
  - `LoginResponse` — `access_token`, `token_type`, `user: User`
  - `SignupRequest` — `email`, `password`, `name`, `affiliation?`
  - `SignupResponse` — same as LoginResponse
  - `ForgotPasswordRequest` — `email`
- **Edge cases**: All types must be exported; Zod schemas for runtime validation of API responses

### FR-2: Auth Store (Zustand)
- **What**: Global auth state using zustand with persistence
- **Store shape**: `AuthState` + actions: `login(email, password)`, `signup(data)`, `logout()`, `refreshUser()`, `setToken(token)`, `clearAuth()`
- **Persistence**: Token stored in `sessionStorage` (SSR-safe — only accessed client-side)
- **Outputs**: `useAuthStore` hook
- **Edge cases**: Handle SSR (no window), handle expired tokens, handle concurrent login attempts

### FR-3: API Client Auth Interceptor
- **What**: Enhance existing `apiClient` to auto-attach Bearer token and handle 401 responses
- **Behavior**:
  - Before each request: read token from auth store, attach `Authorization: Bearer <token>` header
  - On 401 response: clear auth state, redirect to `/login`
  - On 403 response: do NOT clear auth — user is authenticated but lacks permission
- **Inputs**: Existing `apiClient` from `lib/api-client.ts`
- **Outputs**: Modified request pipeline with auth header injection
- **Edge cases**: No token = no header (anonymous requests still work), avoid infinite redirect loop on /login 401

### FR-4: ProtectedRoute Component
- **What**: Wrapper component that redirects unauthenticated users to `/login`
- **Behavior**:
  - If `isLoading`: show skeleton/spinner
  - If `!isAuthenticated`: redirect to `/login?redirect={currentPath}`
  - If `isAuthenticated`: render children
- **Props**: `children: ReactNode`, `fallback?: ReactNode`
- **Edge cases**: SSR-safe (use `useEffect` for redirect, not immediate), preserve intended URL for post-login redirect

### FR-5: Login Page Shell (`/login`)
- **What**: Login form page using react-hook-form + zod validation
- **Fields**: email (required, valid email), password (required, min 8 chars)
- **Behavior**: Submit calls `authStore.login()`, success redirects to `redirect` query param or `/`, error shows toast
- **UI**: Centered card layout, link to signup, link to forgot password
- **Edge cases**: Disable submit while loading, show field-level validation errors

### FR-6: Signup Page Shell (`/signup`)
- **What**: Registration form page
- **Fields**: name (required), email (required, valid), password (required, min 8, must include number), affiliation (optional)
- **Behavior**: Submit calls `authStore.signup()`, success redirects to `/`, error shows toast
- **UI**: Centered card layout, link to login
- **Edge cases**: Password confirmation field, disable submit while loading

### FR-7: Forgot Password Page Shell (`/forgot-password`)
- **What**: Simple email-only form to request password reset
- **Fields**: email (required, valid)
- **Behavior**: Submit shows success message (backend not wired yet), link back to login
- **Edge cases**: Rate limiting messaging (client-side), always show success (don't leak user existence)

## Tangible Outcomes
- [ ] `frontend/src/types/auth.ts` exports all auth types + Zod schemas
- [ ] `frontend/src/lib/auth/store.ts` exports `useAuthStore` zustand hook with login/signup/logout actions
- [ ] `frontend/src/lib/auth/interceptor.ts` exports auth interceptor that enhances apiClient
- [ ] `frontend/src/components/auth/protected-route.tsx` redirects unauthenticated users
- [ ] `frontend/src/app/(auth)/login/page.tsx` renders login form with validation
- [ ] `frontend/src/app/(auth)/signup/page.tsx` renders signup form with validation
- [ ] `frontend/src/app/(auth)/forgot-password/page.tsx` renders forgot password form
- [ ] All auth types validated with Zod schemas at runtime
- [ ] API client auto-attaches Bearer token when authenticated
- [ ] 401 responses trigger automatic logout + redirect to /login
- [ ] All components have Vitest tests
- [ ] `pnpm test` passes with all new tests green

## Test-Driven Requirements

### Tests to Write First
1. `auth.test.ts` (types): Zod schemas validate correct/incorrect LoginResponse, SignupResponse, User
2. `store.test.ts`: login sets user+token, logout clears state, signup sets user+token, handles errors
3. `interceptor.test.ts`: attaches Bearer header when token present, skips when absent, clears auth on 401, no clear on 403
4. `protected-route.test.tsx`: renders children when authenticated, redirects when not, shows loading state
5. `login/page.test.tsx`: renders form fields, validates email/password, calls login on submit, shows errors
6. `signup/page.test.tsx`: renders form fields, validates all fields, calls signup on submit, shows errors
7. `forgot-password/page.test.tsx`: renders email field, shows success message on submit

### Mocking Strategy
- Mock `useAuthStore` with zustand for component tests
- Mock `fetch` / `apiClient` for store tests
- Mock `next/navigation` (`useRouter`, `useSearchParams`, `usePathname`) for redirect tests
- Use `@testing-library/react` + `@testing-library/user-event` for form interaction

### Coverage
- All public functions and components tested
- Zod schema validation (valid + invalid inputs)
- Auth state transitions (logged out → login → logged in → logout)
- Error paths (invalid credentials, network error, 401/403 handling)
- Form validation (field-level errors, submit disabled states)
