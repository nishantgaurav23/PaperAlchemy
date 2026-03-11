# Spec S9.8 -- Trends Dashboard

## Overview
Research trends and analytics dashboard for PaperAlchemy. Displays trending topics, hot papers, category breakdown, and publication timeline charts. Uses **recharts** for data visualization, following the existing Next.js 15 + TypeScript + Tailwind + shadcn/ui patterns.

The dashboard provides at-a-glance insights into the indexed paper collection: what categories are most represented, which papers are recent/popular, and how publication volume trends over time.

## Dependencies
- S9.2 (Layout & Navigation) — done

## Target Location
- `frontend/src/app/dashboard/` — Dashboard page
- `frontend/src/components/dashboard/` — Dashboard sub-components (charts, cards, widgets)
- `frontend/src/lib/api/dashboard.ts` — API client functions for dashboard data
- `frontend/src/types/dashboard.ts` — TypeScript types for dashboard data

## Functional Requirements

### FR-1: Dashboard Page Shell
- **What**: Main dashboard page at `/dashboard` showing a grid of analytics widgets
- **Inputs**: None (loads on mount)
- **Outputs**: Responsive grid layout with stat cards, charts, and paper lists
- **Edge cases**: API unavailable → show skeleton/error state; empty data → show "No papers indexed yet" message

### FR-2: Stats Overview Cards
- **What**: Top-level stat cards showing key metrics
- **Inputs**: Stats API response
- **Outputs**: Cards showing: Total Papers, Papers This Week, Total Categories, Most Active Category
- **Edge cases**: Zero values display "0", loading state shows skeleton

### FR-3: Category Breakdown Chart
- **What**: Pie/donut chart showing distribution of papers across arXiv categories
- **Inputs**: Category counts from API
- **Outputs**: Interactive donut chart with legend, top 8 categories + "Other" grouping
- **Edge cases**: Single category → full circle; no data → empty state message

### FR-4: Publication Timeline Chart
- **What**: Area/line chart showing publication volume over time (monthly)
- **Inputs**: Time-series data from API (month, count)
- **Outputs**: Responsive area chart with tooltips, X-axis (months), Y-axis (paper count)
- **Edge cases**: Single month → show as bar; no data → empty state

### FR-5: Trending/Hot Papers List
- **What**: List of recently indexed or most-viewed papers
- **Inputs**: Papers list from API (sorted by recency or relevance)
- **Outputs**: Paper cards with title, authors, categories (badges), published date, arXiv link
- **Edge cases**: No papers → "No papers found" message; long titles → truncate with ellipsis

### FR-6: Trending Topics Widget
- **What**: Tag cloud or list of trending research topics extracted from paper titles/abstracts
- **Inputs**: Topic list with frequency counts from API
- **Outputs**: Styled tag list with relative sizing or frequency badge
- **Edge cases**: No topics → placeholder; very long topic → truncate

### FR-7: API Integration with Mock Fallback
- **What**: Dashboard API client functions that call backend endpoints, with mock data fallback when backend is unavailable
- **Inputs**: API calls to `/api/v1/dashboard/stats`, `/api/v1/dashboard/trends`
- **Outputs**: Typed dashboard data
- **Edge cases**: Backend returns 404/500 → use mock data, show "Using sample data" indicator

## Tangible Outcomes
- [ ] Dashboard page renders at `/dashboard` with responsive grid layout
- [ ] Stats overview cards display 4 key metrics with loading skeletons
- [ ] Category donut chart renders with recharts, shows top categories
- [ ] Publication timeline area chart renders monthly paper counts
- [ ] Hot papers list shows recent papers with arXiv links
- [ ] Trending topics widget displays topic tags with frequency
- [ ] Mock data fallback works when backend API is unavailable
- [ ] Dark mode support for all charts and widgets
- [ ] All components have Vitest tests
- [ ] Navigation sidebar links to dashboard correctly

## Test-Driven Requirements

### Tests to Write First
1. `dashboard-page.test.tsx`: Dashboard page renders all widget sections
2. `stats-cards.test.tsx`: Stats cards display metrics, handle loading/error states
3. `category-chart.test.tsx`: Category chart renders with mock data, handles empty state
4. `timeline-chart.test.tsx`: Timeline chart renders, handles single/no data points
5. `hot-papers.test.tsx`: Paper list renders, links to arXiv, handles empty state
6. `trending-topics.test.tsx`: Topic tags render with frequency, handle empty state
7. `dashboard-api.test.ts`: API client functions handle success, error, and mock fallback

### Mocking Strategy
- Mock `fetch` / API client for all dashboard API calls
- Use mock dashboard data fixtures for consistent test data
- Mock recharts components in unit tests (they don't render in jsdom)
- Use React Testing Library for component rendering and assertions

### Coverage
- All dashboard components tested (render, loading, error, empty states)
- API client functions tested (success, error, fallback)
- Dark mode class application tested
- Responsive layout breakpoints tested where feasible
