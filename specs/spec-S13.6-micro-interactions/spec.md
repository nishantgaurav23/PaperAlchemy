# Spec S13.6 -- Animations & Micro-Interactions

## Overview
Add polish-level animations and micro-interactions across the frontend to create a premium, responsive feel. Uses framer-motion (already installed via S9b.3) for page transitions, scroll-triggered animations, and interactive feedback. Covers skeleton loaders, button feedback, hover previews, toast animations, progress indicators, and counting animations.

## Dependencies
- **S9.1** (Next.js 15 setup) — done
- **S9b.3** (Frontend infra deps — framer-motion, sonner) — done

## Target Location
`frontend/src/` — shared animation utilities, component enhancements, and page-level transitions

## Functional Requirements

### FR-1: Page Transition Animations
- **What**: Smooth fade/slide transitions when navigating between pages using framer-motion's `AnimatePresence` and `motion` components
- **Inputs**: Page route changes via Next.js App Router
- **Outputs**: Fade-in with subtle Y-axis slide (opacity 0→1, translateY 8→0) on page mount
- **Edge cases**: No animation on initial page load (SSR), reduced-motion preference respected via `prefers-reduced-motion`

### FR-2: Skeleton Shimmer Loaders
- **What**: Animated skeleton placeholders shown while data is loading on search, dashboard, chat, and papers pages
- **Inputs**: Loading state (boolean or suspense boundary)
- **Outputs**: Pulsing shimmer effect matching content layout shape (cards, text lines, charts)
- **Edge cases**: Skeleton disappears instantly when data arrives (no flash), works in both light and dark mode

### FR-3: Button Press Feedback
- **What**: Scale-down effect (scale-95) on button press for tactile feedback
- **Inputs**: Mouse/touch press events on interactive buttons
- **Outputs**: `transform: scale(0.95)` on press with 100ms transition, return to scale(1) on release
- **Edge cases**: Disabled buttons have no feedback, keyboard activation (Enter/Space) also triggers effect

### FR-4: Hover Card Previews for Paper Links
- **What**: Popover card showing paper abstract preview on hover over paper title links
- **Inputs**: Hover/focus event on paper title links
- **Outputs**: Animated popover (fade + scale from 95% to 100%) with title, authors, year, abstract snippet (first 150 chars)
- **Edge cases**: Dismiss on mouse leave with 200ms delay (prevent flicker), position auto-adjusts to stay in viewport, touch devices use tap instead

### FR-5: Toast Notifications with Slide-In Animation
- **What**: Toast notifications (via sonner) with slide-in-from-bottom animation for success, error, info, and warning states
- **Inputs**: Toast trigger events (upload complete, error, copy-to-clipboard, etc.)
- **Outputs**: Slide-in from bottom-right, auto-dismiss after 4s, manual dismiss with swipe/click
- **Edge cases**: Multiple toasts stack vertically, max 3 visible at once

### FR-6: Progress Indicators for Long Operations
- **What**: Animated progress bars/spinners for PDF upload, paper ingestion, and search operations
- **Inputs**: Operation start/progress/complete events
- **Outputs**: Indeterminate progress bar with gradient animation, or determinate bar when percentage is known
- **Edge cases**: Progress resets on error, shows error state with red color, handles 0% and 100% edge cases

### FR-7: Scroll-Triggered Fade-In for Dashboard Cards
- **What**: Cards and sections fade in as they enter the viewport during scroll
- **Inputs**: Scroll position relative to dashboard card elements
- **Outputs**: Staggered fade-in (opacity 0→1, translateY 20→0) with 100ms delay between siblings
- **Edge cases**: Elements already in viewport on load animate immediately, no re-animation on scroll back up (animate once), reduced-motion disables animations

### FR-8: Smooth Number Counting Animations for Stats
- **What**: Animated counter that counts up from 0 to target value when stats become visible
- **Inputs**: Target number, duration (default 1.5s), trigger (intersection observer)
- **Outputs**: Smooth easing count from 0 to target with locale-formatted numbers (e.g., 1,234)
- **Edge cases**: Handles 0 (no animation), very large numbers (abbreviate: 1.2K, 3.4M), decimal values, negative values

## Tangible Outcomes
- [ ] Page transitions animate on route change (fade + slide)
- [ ] Skeleton loaders render on search, dashboard, chat, and papers pages during loading
- [ ] Buttons show scale-95 press feedback
- [ ] Paper title hover shows preview popover with abstract snippet
- [ ] Toast notifications slide in from bottom with stacking
- [ ] Progress indicators animate during upload and search operations
- [ ] Dashboard cards fade in on scroll with staggered timing
- [ ] Stats counters animate from 0 to target value
- [ ] All animations respect `prefers-reduced-motion` media query
- [ ] All components have Vitest tests

## Test-Driven Requirements

### Tests to Write First
1. `test_page_transition`: PageTransition component renders children, applies motion props
2. `test_skeleton_shimmer`: Skeleton components render with animate-pulse class, match expected shapes
3. `test_button_press`: Button shows scale-95 on press, scale-100 on release
4. `test_hover_card_preview`: HoverCard shows/hides on hover with paper data
5. `test_toast_notifications`: Toast wrapper renders sonner Toaster with correct position config
6. `test_progress_indicator`: Progress bar renders with correct width%, animates indeterminate state
7. `test_scroll_fade_in`: ScrollFadeIn component renders children, applies intersection observer
8. `test_counter_animation`: AnimatedCounter counts from 0 to target, formats numbers

### Mocking Strategy
- Mock `framer-motion` in Vitest (motion components render as plain divs with data attributes)
- Mock `IntersectionObserver` API for scroll-triggered tests
- Mock `window.matchMedia` for reduced-motion preference tests
- Use `@testing-library/react` for user event simulation (hover, press)

### Coverage
- All 8 FR components tested
- Reduced-motion behavior tested
- Edge cases (empty data, 0 values, disabled state) covered
- Dark mode visual compatibility verified via class checks
