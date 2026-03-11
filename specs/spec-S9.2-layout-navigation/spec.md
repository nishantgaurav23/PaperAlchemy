# Spec S9.2 -- Layout & Navigation

## Overview
Build the application shell for PaperAlchemy's Next.js frontend: a responsive layout with a collapsible sidebar, top header with theme toggle, and breadcrumb navigation. This establishes the consistent UI frame that all page-level specs (S9.3-S9.9) will render inside.

## Dependencies
- **S9.1** (Next.js project setup) — done

## Target Location
- `frontend/src/app/layout.tsx` (update existing)
- `frontend/src/components/layout/sidebar.tsx`
- `frontend/src/components/layout/header.tsx`
- `frontend/src/components/layout/breadcrumbs.tsx`
- `frontend/src/components/layout/app-shell.tsx`
- `frontend/src/components/layout/sidebar-nav-item.tsx`
- `frontend/src/components/layout/mobile-nav.tsx`

## Functional Requirements

### FR-1: App Shell
- **What**: A root layout wrapper that combines sidebar + header + main content area
- **Inputs**: `children` (React nodes from page routes)
- **Outputs**: Rendered shell with sidebar on left, header on top, content area filling remaining space
- **Edge cases**: Content overflow (scroll), empty content area

### FR-2: Collapsible Sidebar
- **What**: Left sidebar with navigation links, collapsible to icon-only mode on desktop; off-canvas drawer on mobile
- **Inputs**: Collapse toggle click, current route (for active state)
- **Outputs**: Sidebar renders nav items; collapsed state persisted via localStorage; active route highlighted
- **Navigation items**:
  - Search (magnifying glass icon) — `/search`
  - Chat (message icon) — `/chat`
  - Upload (upload icon) — `/upload`
  - Papers (file icon) — `/papers`
  - Collections (bookmark icon) — `/collections`
  - Dashboard (bar-chart icon) — `/dashboard`
- **Edge cases**: Window resize between mobile/desktop breakpoints, no JS (SSR fallback)

### FR-3: Header
- **What**: Top header bar with app title/logo, breadcrumbs, and theme toggle
- **Inputs**: Current route (for breadcrumbs), theme state
- **Outputs**: Rendered header with breadcrumb trail and dark/light toggle button
- **Edge cases**: Very long breadcrumb paths, mobile viewport (compact mode)

### FR-4: Breadcrumbs
- **What**: Route-aware breadcrumb component showing navigation hierarchy
- **Inputs**: Current pathname from `usePathname()`
- **Outputs**: Clickable breadcrumb links (e.g., Home > Papers > Detail)
- **Edge cases**: Root path (show nothing or just "Home"), dynamic segments like `[id]`

### FR-5: Theme Toggle (existing — integrate into header)
- **What**: Move the existing `theme-toggle.tsx` into the header bar
- **Inputs**: Click event
- **Outputs**: Toggles dark/light mode via next-themes
- **Edge cases**: System preference changes, SSR hydration mismatch (already handled by next-themes)

### FR-6: Mobile Navigation
- **What**: On screens < 768px, sidebar becomes a slide-out drawer triggered by a hamburger menu button in the header
- **Inputs**: Hamburger button click, route change (auto-close)
- **Outputs**: Animated drawer overlay with same nav items as sidebar
- **Edge cases**: Scroll lock when drawer open, focus trap for accessibility

### FR-7: Responsive Behavior
- **What**: Layout adapts to viewport width
- **Breakpoints**:
  - `< 768px` (mobile): No sidebar visible, hamburger menu in header, drawer nav
  - `>= 768px` (tablet): Collapsed sidebar (icons only)
  - `>= 1024px` (desktop): Expanded sidebar (icons + labels), collapsible on click
- **Edge cases**: Orientation change on tablet, very narrow desktop windows

## Tangible Outcomes
- [ ] App shell renders sidebar + header + content area on all routes
- [ ] Sidebar shows 6 navigation items with correct icons and routes
- [ ] Clicking a nav item navigates to the correct route and highlights active item
- [ ] Sidebar collapses to icon-only mode on toggle click (desktop)
- [ ] Collapsed state persists across page navigations (localStorage)
- [ ] Mobile viewport shows hamburger menu instead of sidebar
- [ ] Mobile drawer opens/closes with animation
- [ ] Breadcrumbs reflect current route hierarchy
- [ ] Theme toggle in header switches dark/light mode
- [ ] All components pass accessibility checks (keyboard nav, aria labels)
- [ ] All tests pass (Vitest + React Testing Library)

## Test-Driven Requirements

### Tests to Write First
1. `app-shell.test.tsx`: Renders sidebar, header, and children content
2. `sidebar.test.tsx`: Renders all nav items, highlights active route, toggles collapse
3. `header.test.tsx`: Renders app title, breadcrumbs, theme toggle
4. `breadcrumbs.test.tsx`: Generates correct breadcrumb links from pathname
5. `mobile-nav.test.tsx`: Opens/closes drawer, renders nav items
6. `sidebar-nav-item.test.tsx`: Renders icon + label, active state styling

### Mocking Strategy
- Mock `next/navigation` (`usePathname`, `useRouter`) for route-dependent tests
- Mock `next-themes` (`useTheme`) for theme toggle tests
- Mock `localStorage` for collapse state persistence tests
- Use React Testing Library `render` + `screen` + `userEvent` for interaction tests

### Coverage
- All public components tested
- Active route detection tested for each nav item path
- Collapse/expand toggle tested
- Mobile breakpoint behavior tested (via mocked media queries or responsive utils)
- Breadcrumb generation tested for various path depths
- Accessibility: aria-labels, keyboard navigation tested
