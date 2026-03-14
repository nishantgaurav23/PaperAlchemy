# Spec S13.7 -- Mobile-First Responsive Polish

## Overview
Complete mobile-first responsive overhaul for PaperAlchemy. Replace the existing sidebar-based mobile navigation with a bottom navigation bar, add touch-friendly interactions (swipe gestures, pull-to-refresh), ensure all touch targets meet accessibility minimums (44px), implement adaptive layouts that transition from single-column on mobile to multi-column on desktop, use sheet/drawer patterns for mobile filters, and apply responsive typography with CSS clamp-based fluid sizing.

## Dependencies
- **S9.2** (Layout & navigation) — done

## Target Location
`frontend/src/` — affects layout components, page layouts, global styles, and search components

## Functional Requirements

### FR-1: Bottom Navigation Bar (Mobile)
- **What**: On screens < 768px (md breakpoint), replace the sidebar with a fixed bottom navigation bar showing the 6 nav items (Search, Chat, Upload, Papers, Collections, Dashboard) as icon+label buttons
- **Inputs**: NAV_ITEMS array, current route (usePathname)
- **Outputs**: Fixed bottom bar with active state indicator, hidden on md+ screens
- **Edge cases**: More than 5 items may need a "More" overflow menu; active state must highlight correctly; bar must not overlap page content (add bottom padding to main)

### FR-2: Swipe Gestures for Chat
- **What**: Enable horizontal swipe gestures on the chat page to navigate between conversations or dismiss messages
- **Inputs**: Touch events (touchstart, touchmove, touchend)
- **Outputs**: Swipe left/right detection with visual feedback; configurable threshold (50px minimum swipe distance)
- **Edge cases**: Must not interfere with scrolling; disabled when keyboard is open; should work with mouse drag on desktop for testing

### FR-3: Pull-to-Refresh on Search
- **What**: Add pull-to-refresh gesture on the search results page to re-execute the current search query
- **Inputs**: Touch pull-down gesture on search results container
- **Outputs**: Visual pull indicator (spinner + "Pull to refresh" text), triggers search refresh callback
- **Edge cases**: Only activate when scrolled to top; must not conflict with normal scrolling; show loading state during refresh; disabled on desktop

### FR-4: Touch-Optimized Hit Targets
- **What**: Ensure all interactive elements (buttons, links, nav items, filter chips) have a minimum touch target of 44×44px as per WCAG 2.5.5
- **Inputs**: Existing interactive components
- **Outputs**: CSS utility class or Tailwind config for minimum touch targets; audit of existing components
- **Edge cases**: Small icons should have expanded tap areas via padding; closely spaced items need adequate spacing

### FR-5: Adaptive Layouts (Single → Multi-Column)
- **What**: Pages transition from single-column on mobile to multi-column on larger screens. Dashboard uses 1-col → 2-col → 3-col grid. Search results use full-width cards on mobile, grid on desktop. Paper detail uses stacked sections on mobile, sidebar layout on desktop.
- **Inputs**: Screen width via Tailwind responsive prefixes
- **Outputs**: Responsive grid layouts using Tailwind's grid system (grid-cols-1 md:grid-cols-2 lg:grid-cols-3)
- **Edge cases**: Content must not overflow horizontally on any screen size; images and charts must scale proportionally

### FR-6: Sheet/Drawer for Mobile Filters
- **What**: On mobile, search filters (category, sort, date range) appear in a bottom sheet/drawer instead of inline. Use the existing shadcn Sheet component with `side="bottom"`.
- **Inputs**: Filter state, open/close toggle
- **Outputs**: Bottom sheet with filter controls, "Apply" button, swipe-to-dismiss
- **Edge cases**: Sheet must not exceed 80vh height; scrollable if content overflows; filters persist when sheet closes

### FR-7: Responsive Typography (Clamp-Based Fluid Sizing)
- **What**: Implement fluid typography using CSS `clamp()` for headings and body text that smoothly scales between mobile and desktop sizes
- **Inputs**: Design tokens for min/max font sizes
- **Outputs**: CSS custom properties or Tailwind utilities for fluid text sizing
- **Edge cases**: Must respect user's browser font size preferences; minimum readable size is 14px on mobile

## Tangible Outcomes
- [ ] Bottom nav bar visible on mobile (< 768px), hidden on desktop
- [ ] Bottom nav highlights active route
- [ ] Existing sidebar hidden on mobile, bottom nav replaces it
- [ ] Swipe gesture hook (`useSwipe`) works with configurable threshold
- [ ] Pull-to-refresh component shows visual feedback and triggers callback
- [ ] All interactive elements have min 44px touch targets on mobile
- [ ] Dashboard page uses responsive grid (1 → 2 → 3 columns)
- [ ] Search page uses responsive card layout
- [ ] Search filters use bottom sheet on mobile
- [ ] Typography scales fluidly with clamp() between mobile and desktop
- [ ] No horizontal overflow on any page at 320px viewport width
- [ ] All existing tests still pass

## Test-Driven Requirements

### Tests to Write First
1. `bottom-nav.test.tsx`: Renders all nav items, highlights active route, hidden on md+ screens
2. `use-swipe.test.ts`: Detects left/right swipe, respects threshold, fires callbacks
3. `pull-to-refresh.test.tsx`: Shows indicator on pull, triggers refresh callback, only activates at scroll top
4. `mobile-filter-sheet.test.tsx`: Opens/closes sheet, renders filter controls, applies filters
5. `responsive-layout.test.tsx`: Verifies grid column classes are applied correctly
6. `touch-target.test.tsx`: Verifies minimum 44px dimensions on interactive elements

### Mocking Strategy
- Use `@testing-library/react` with `fireEvent.touchStart/touchMove/touchEnd` for gesture tests
- Mock `usePathname` from `next/navigation` for active route testing
- Use `matchMedia` mock for responsive breakpoint tests
- Mock `window.scrollY` for pull-to-refresh scroll position detection

### Coverage
- All new components and hooks tested
- Touch gesture edge cases covered (short swipe ignored, vertical scroll not triggered)
- Responsive breakpoint behavior validated
- Accessibility: touch targets meet 44px minimum
