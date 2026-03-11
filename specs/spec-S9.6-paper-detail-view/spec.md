# Spec S9.6 -- Paper Detail View

## Overview
Full paper detail page accessible at `/papers/[id]`. Displays complete paper metadata (title, authors, abstract, categories, dates), provides links to arXiv and PDF, shows paper sections/highlights when available, and lists related papers. This is the canonical view for any paper in the system — search results, chat citations, and upload results all link here.

## Dependencies
- S9.2 (Layout & navigation) — done

## Target Location
- `frontend/src/app/papers/[id]/page.tsx` — dynamic route page
- `frontend/src/components/paper/` — reusable paper detail components
- `frontend/src/lib/api/papers.ts` — paper API client
- `frontend/src/types/paper.ts` — extended paper types (update existing)

## Functional Requirements

### FR-1: Dynamic Route & Data Fetching
- **What**: `/papers/[id]` page fetches paper by ID from backend API
- **Inputs**: Paper ID from URL params (`params.id`)
- **Outputs**: Full paper data rendered on page
- **Edge cases**: Paper not found (404 page), API error (error state with retry), loading state (skeleton)
- **API**: `GET /api/v1/papers/{id}` — returns full paper object
- **Mock mode**: Returns mock paper data when API unavailable (dev workflow)

### FR-2: Paper Metadata Display
- **What**: Header section showing complete paper metadata
- **Inputs**: Paper object with title, authors, abstract, categories, dates
- **Outputs**: Rendered metadata section
- **Details**:
  - Title (large heading)
  - Authors list (full, not truncated)
  - Published date (formatted: "March 15, 2024")
  - Categories as badges
  - Abstract (full text, expandable if very long)
  - ArXiv link (external, opens in new tab)
  - PDF link (external, opens in new tab)
  - Copy citation button (copies BibTeX-style reference)

### FR-3: Paper Sections Display
- **What**: Show paper sections/content when available (from parsed PDF)
- **Inputs**: Paper sections array (title + content per section)
- **Outputs**: Collapsible section list
- **Edge cases**: No sections available (show "Sections not yet parsed" message)
- **Details**:
  - Section headings (Introduction, Methods, Results, etc.)
  - Section content (rendered as paragraphs)
  - Collapsible/expandable sections (default: first 2 expanded)

### FR-4: Paper Highlights & Analysis
- **What**: Show AI-generated summary, highlights, methodology if available
- **Inputs**: Optional summary, highlights, methodology analysis objects
- **Outputs**: Tabbed or card-based display (reuse patterns from upload analysis)
- **Edge cases**: Analysis not available (show "Request Analysis" CTA button)
- **Details**:
  - Summary card: objective, method, key findings, contribution, limitations
  - Highlights card: novel contributions, important findings, practical implications
  - Methodology card: approach, datasets, baselines, results

### FR-5: Related Papers
- **What**: Show related papers section based on similarity
- **Inputs**: Array of related paper objects (from API)
- **Outputs**: Horizontal scrollable list of paper cards
- **Edge cases**: No related papers (hide section), API error (hide section gracefully)
- **API**: `GET /api/v1/papers/{id}/related` — returns list of related papers
- **Details**:
  - Show up to 6 related papers
  - Each card: title, authors (truncated), categories, link to detail page
  - "View All" link if more than 6

### FR-6: Back Navigation & Breadcrumbs
- **What**: Navigation aids for returning to previous page
- **Inputs**: None (uses router history)
- **Outputs**: Back button + breadcrumb trail
- **Details**:
  - Back arrow button (router.back())
  - Breadcrumb: Papers > {Paper Title (truncated)}

### FR-7: Loading & Error States
- **What**: Proper UX for async states
- **Loading**: Skeleton placeholders matching layout structure
- **Error**: Error message with retry button
- **Not Found**: Custom 404 message ("Paper not found")

## Tangible Outcomes
- [ ] `/papers/[id]` route renders full paper detail page
- [ ] Paper metadata (title, authors, abstract, categories, dates) displayed correctly
- [ ] ArXiv external link works (opens in new tab)
- [ ] PDF external link works (opens in new tab)
- [ ] Paper sections displayed when available (collapsible)
- [ ] Analysis tabs (summary, highlights, methodology) shown when available
- [ ] Related papers section renders with links to other paper detail pages
- [ ] Loading skeleton shown while fetching
- [ ] Error state with retry button shown on API failure
- [ ] Not-found state shown for invalid paper IDs
- [ ] Back navigation button works
- [ ] Copy citation button copies formatted reference
- [ ] PaperCard in search results links to `/papers/{id}`
- [ ] All Vitest tests pass

## Test-Driven Requirements

### Tests to Write First
1. `paper-detail-page.test.tsx`: Page renders with mock paper data, shows metadata
2. `paper-detail-page.test.tsx`: Loading state shows skeleton
3. `paper-detail-page.test.tsx`: Error state shows retry button
4. `paper-detail-page.test.tsx`: Not-found state renders correctly
5. `paper-header.test.tsx`: Renders title, authors, date, categories, abstract
6. `paper-header.test.tsx`: ArXiv and PDF links have correct hrefs and target="_blank"
7. `paper-header.test.tsx`: Copy citation button triggers clipboard write
8. `paper-sections.test.tsx`: Renders section headings and content
9. `paper-sections.test.tsx`: Sections are collapsible/expandable
10. `paper-sections.test.tsx`: Empty sections shows fallback message
11. `paper-analysis.test.tsx`: Renders summary, highlights, methodology tabs
12. `paper-analysis.test.tsx`: Shows CTA when analysis not available
13. `related-papers.test.tsx`: Renders related paper cards with links
14. `related-papers.test.tsx`: Hides section when no related papers
15. `papers-api.test.ts`: API client fetches paper by ID
16. `papers-api.test.ts`: API client fetches related papers

### Mocking Strategy
- Mock `fetch` for API calls (paper detail, related papers)
- Mock `navigator.clipboard.writeText` for copy citation
- Mock `next/navigation` (useParams, useRouter)
- Use mock paper data fixtures for consistent test data

### Coverage
- All components tested (header, sections, analysis, related papers)
- All states tested (loading, error, not-found, success)
- Edge cases tested (missing optional fields, empty arrays)
- User interactions tested (expand/collapse, copy, navigate)
