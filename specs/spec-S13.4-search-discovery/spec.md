# Spec S13.4 -- Search & Discovery Polish

## Overview
Premium search & discovery experience for the PaperAlchemy frontend. Upgrades the existing S9.3 search interface with autocomplete/typeahead, rich paper cards, staggered animations, active filter pills, polished empty states, and skeleton shimmer loading.

## Dependencies
- **S9.3** (Search Interface) — `done` — provides base search page, paper cards, filters, pagination

## Target Location
`frontend/src/components/search/` — enhanced search components
`frontend/src/app/search/page.tsx` — updated search page

## Functional Requirements

### FR-1: Autocomplete / Typeahead with Recent Searches
- **What**: Search bar shows a dropdown with recent searches (stored in localStorage) and typeahead suggestions as the user types
- **Inputs**: User keystrokes in the search input
- **Outputs**: Dropdown list of recent searches (max 5) and typeahead matches
- **Edge cases**: No recent searches (show nothing), localStorage unavailable (graceful fallback), empty input (show recent only)
- **Storage**: `localStorage` key `paperalchemy:recent-searches` — array of strings (max 10)

### FR-2: Rich Paper Cards
- **What**: Upgraded paper cards with: abstract preview on hover (tooltip/popover), category color chips (mapped colors per arXiv category), bookmark icon (visual only for now — no backend)
- **Inputs**: `Paper` object with title, authors, abstract, categories, arxiv_id
- **Outputs**: Glassmorphism-styled card with hover abstract preview, colored category badges, bookmark toggle
- **Edge cases**: Missing abstract (hide preview), no categories (hide chips), long titles (truncate with ellipsis)

### FR-3: Staggered Fade-In Animations
- **What**: Search results appear with staggered fade-in animation (each card delayed by 50ms * index)
- **Inputs**: Array of paper results
- **Outputs**: CSS animation — opacity 0→1, translateY 8px→0, staggered per card
- **Edge cases**: Reduced motion preference (skip animations), large result sets (cap stagger at 10 items)

### FR-4: Active Filter Pills with Remove Button
- **What**: Active search filters (category, sort) displayed as removable pill badges above results
- **Inputs**: Current filter state (category, sort, query)
- **Outputs**: Pill badges with × remove button; clicking removes the filter
- **Edge cases**: No active filters (hide pills section), "relevance" sort is default — don't show pill for it

### FR-5: Enhanced "No Results" State
- **What**: Polished empty state with gradient illustration, search suggestions, and "try these" quick links
- **Inputs**: Empty search results after a query
- **Outputs**: Illustration + "No papers found for X" + suggestion text + quick search links
- **Edge cases**: Very long query (truncate in display)

### FR-6: Skeleton Shimmer Loading States
- **What**: Upgraded loading skeletons with shimmer animation effect (gradient sweep) instead of plain pulse
- **Inputs**: `isLoading` state
- **Outputs**: Shimmer-animated skeleton cards matching the rich paper card layout
- **Edge cases**: Reduced motion (fallback to static placeholder)

## Tangible Outcomes
- [ ] Search bar shows recent searches dropdown on focus (stored in localStorage)
- [ ] Recent searches persist across page reloads (max 10, LIFO)
- [ ] Paper cards display colored category chips (at least 5 distinct colors for different arXiv prefixes)
- [ ] Paper cards show abstract preview on hover via tooltip/popover
- [ ] Paper cards have a bookmark icon toggle (visual only, no backend persistence)
- [ ] Search results animate in with staggered fade-in (respects prefers-reduced-motion)
- [ ] Active filters shown as removable pill badges
- [ ] Removing a filter pill updates URL params and re-fetches results
- [ ] "No results" state shows gradient illustration + search suggestions
- [ ] Loading state uses shimmer animation instead of plain pulse
- [ ] All new components have Vitest tests
- [ ] All existing search tests still pass

## Test-Driven Requirements

### Tests to Write First
1. `search-bar.test.tsx` — extended: recent searches dropdown renders on focus, stores/retrieves from localStorage, max 10 items, clear individual recent search
2. `paper-card.test.tsx` — extended: category color chips render, bookmark icon toggles, abstract preview on hover
3. `filter-pills.test.tsx` — new: renders active filters as pills, removes filter on click, hides when no filters active
4. `search-results.test.tsx` — extended: staggered animation classes applied, shimmer skeletons render, enhanced empty state renders with suggestions
5. `recent-searches.test.ts` — new: localStorage hook/utility — save, load, remove, max capacity

### Mocking Strategy
- Mock `localStorage` via `vi.stubGlobal` or JSDOM built-in
- Mock `useRouter` and `useSearchParams` from next/navigation
- Use `@testing-library/react` for component interaction tests
- No external API mocks needed (search API already mocked in existing tests)

### Coverage
- All public components tested
- localStorage edge cases (full, unavailable)
- Animation class presence (not visual rendering)
- Filter pill interactions
- Reduced motion media query handling
