# Spec S9.3 -- Search Interface

## Overview
Search page for PaperAlchemy's Next.js frontend. Provides a search bar with category filters, sort options, pagination, and paper result cards linking to arXiv. This is the primary discovery interface for the knowledge base.

Since the backend search API (S4.4 hybrid search) is not yet implemented, this spec builds the full UI with a mock API layer that can be swapped for real endpoints later. The page must be fully functional with mock data and ready to wire to the real `/api/v1/search` endpoint.

## Dependencies
- S9.2 (Layout & navigation) — **done**

## Target Location
- `frontend/src/app/search/page.tsx` — Search page (Server Component wrapper)
- `frontend/src/components/search/` — Search components
- `frontend/src/lib/api/search.ts` — Search API functions
- `frontend/src/types/paper.ts` — Paper type definitions

## Functional Requirements

### FR-1: Search Bar
- **What**: Full-width search input with submit button at the top of the page
- **Inputs**: Query string (text), Enter key or click to submit
- **Outputs**: Updates URL search params (`?q=...`), triggers search
- **Edge cases**: Empty query shows recent/all papers, whitespace-only treated as empty
- **Details**:
  - Debounced input (300ms) for URL param sync
  - Search icon in input, clear button when non-empty
  - Placeholder: "Search papers by title, author, or topic..."

### FR-2: Category Filter
- **What**: Dropdown or multi-select for arXiv category filtering
- **Inputs**: Category selection (e.g., cs.AI, cs.CL, cs.LG, stat.ML, etc.)
- **Outputs**: Adds `?category=cs.AI` to URL params, filters results
- **Edge cases**: No category selected = all categories, multiple categories supported
- **Details**:
  - Common CS/ML categories pre-populated
  - URL param: `category` (comma-separated for multiple)

### FR-3: Sort Options
- **What**: Sort results by relevance, date (newest/oldest), or citation count
- **Inputs**: Sort selection
- **Outputs**: Adds `?sort=relevance|date_desc|date_asc` to URL params
- **Edge cases**: Default sort = relevance when query present, date_desc when no query
- **Details**:
  - Options: Relevance, Newest First, Oldest First
  - URL param: `sort`

### FR-4: Paper Result Cards
- **What**: Card display for each search result showing paper metadata
- **Inputs**: Paper data from API response
- **Outputs**: Rendered card with title, authors, abstract (truncated), date, categories, arXiv link
- **Edge cases**: Missing authors (show "Unknown"), long titles (truncate with ellipsis), no abstract
- **Details**:
  - Title as link to `/papers/[id]` (internal detail page)
  - External arXiv link button (opens in new tab)
  - Authors list (truncated to first 3 + "et al." if more)
  - Abstract preview (first 200 chars with "..." if longer)
  - Published date (formatted: "Mar 11, 2026")
  - Category badges (colored)
  - Responsive: stack on mobile, side-by-side metadata on desktop

### FR-5: Pagination
- **What**: Page-based navigation for results
- **Inputs**: Page number click or next/prev
- **Outputs**: Updates `?page=N` in URL, fetches new page
- **Edge cases**: First page (disable prev), last page (disable next), no results
- **Details**:
  - Show total results count and current page range ("Showing 1-20 of 142 results")
  - Page size: 20 results
  - URL param: `page` (1-indexed)
  - Show max 5 page numbers with ellipsis for large result sets

### FR-6: Empty & Loading States
- **What**: Appropriate UI states for loading, empty results, and errors
- **Inputs**: API state (loading, success, error, empty)
- **Outputs**: Skeleton loaders during fetch, empty state with suggestions, error with retry
- **Edge cases**: Network timeout, API unavailable
- **Details**:
  - Loading: Skeleton cards (3-4 placeholder cards)
  - Empty: "No papers found" with search suggestions
  - Error: Error message with retry button
  - Initial state (no query): Show prompt to search or browse recent papers

### FR-7: URL-Driven State
- **What**: All search state persisted in URL search params
- **Inputs**: URL params (q, category, sort, page)
- **Outputs**: Page state restored from URL on load/navigation
- **Edge cases**: Invalid page number, unknown category, missing params
- **Details**:
  - Browser back/forward works correctly
  - Shareable URLs (copy URL → same results)
  - Use `useSearchParams` + `useRouter` for state management

## Tangible Outcomes
- [ ] Search page renders at `/search` route
- [ ] Search input updates URL params on submit
- [ ] Category filter adds/removes `category` URL param
- [ ] Sort dropdown changes `sort` URL param
- [ ] Paper cards display title, authors, abstract, date, categories, arXiv link
- [ ] Pagination controls navigate between pages via URL params
- [ ] Loading skeletons shown during fetch
- [ ] Empty state shown when no results
- [ ] Error state shown with retry button
- [ ] All state driven by URL (shareable, back/forward works)
- [ ] Responsive layout (mobile + desktop)
- [ ] All components tested with Vitest + React Testing Library

## Test-Driven Requirements

### Tests to Write First
1. `test_search_page_renders`: Page renders with search input, filters, and content area
2. `test_search_input_updates_url`: Typing and submitting updates URL search params
3. `test_category_filter`: Selecting categories updates URL and filters
4. `test_sort_dropdown`: Changing sort updates URL param
5. `test_paper_card_display`: Paper card shows all metadata fields correctly
6. `test_paper_card_truncation`: Long titles/authors/abstracts are truncated
7. `test_paper_card_arxiv_link`: ArXiv link opens in new tab
8. `test_pagination_controls`: Page buttons update URL, disable at boundaries
9. `test_pagination_display`: Shows correct result count and page range
10. `test_loading_state`: Skeleton loaders shown during fetch
11. `test_empty_state`: Empty message shown when no results
12. `test_error_state`: Error message + retry button shown on API error
13. `test_url_state_restoration`: Page state restored from URL params on load
14. `test_responsive_layout`: Components render correctly at different breakpoints

### Mocking Strategy
- Mock `fetch` / API client for search responses
- Mock `next/navigation` (useSearchParams, useRouter, usePathname)
- Use mock paper data fixtures
- No real API calls in tests

### Coverage
- All public components tested
- Edge cases covered (empty query, missing fields, pagination boundaries)
- Error paths tested (API failure, timeout)
