# Spec S13.5 -- Premium Navigation & Command Palette

## Overview
Upgrade the existing sidebar navigation and header into a premium, polished navigation experience. This includes: a gradient logo mark, active item accent indicators (left border + subtle bg), smooth collapse animation with icon tooltips, keyboard shortcut hints in nav items, a global command palette (Cmd+K) with fuzzy search across papers/collections/actions, breadcrumb trail with dropdown navigation, and a notification bell with badge count.

## Dependencies
- **S9.2** (Layout & Navigation) — provides base `Sidebar`, `Header`, `AppShell`, `Breadcrumbs`, `MobileNav`, `SidebarNavItem` components
- **S9b.5** (Frontend UI Primitives) — provides `Command`, `CommandDialog`, `Tooltip`, `Dialog`, `DropdownMenu` primitives

## Target Location
`frontend/src/components/layout/` (enhance existing components) + new `frontend/src/components/layout/command-palette.tsx`

## Functional Requirements

### FR-1: Enhanced Sidebar with Gradient Logo Mark
- **What**: Replace plain `FlaskConical` icon with a gradient-styled logo mark (CSS gradient background on icon container). Add smooth width transition animation (not just CSS transition — use proper easing).
- **Inputs**: Sidebar collapsed state (boolean)
- **Outputs**: Sidebar renders with gradient logo, smooth collapse/expand animation
- **Edge cases**: SSR hydration mismatch with localStorage (already handled), very long nav labels

### FR-2: Active Item Accent Indicator
- **What**: Active nav items show a left border accent (3px primary-colored left border) + subtle background highlight. Replace current simple `bg-accent` with a more distinctive indicator.
- **Inputs**: Current pathname, nav item href
- **Outputs**: Active item has left border accent + subtle bg; inactive items have no border
- **Edge cases**: Nested routes (e.g., `/papers/123` should activate `/papers`), root path

### FR-3: Collapsed Sidebar Icon Tooltips
- **What**: When sidebar is collapsed, hovering over a nav icon shows a tooltip with the item label + keyboard shortcut hint (if any). Uses the `Tooltip` primitive from S9b.5.
- **Inputs**: Collapsed state, nav item label, optional keyboard shortcut
- **Outputs**: Tooltip appears on hover when collapsed; no tooltip when expanded (label is visible)
- **Edge cases**: Tooltip positioning near screen edges (handled by Radix)

### FR-4: Keyboard Shortcut Hints
- **What**: Each nav item displays a subtle keyboard shortcut hint (e.g., `⌘1` for Search, `⌘2` for Chat). Shortcuts are displayed as a right-aligned kbd element when sidebar is expanded. Shortcuts actually navigate when pressed.
- **Inputs**: Keyboard events (Cmd/Ctrl + number)
- **Outputs**: Navigation to corresponding page on shortcut press; visual hint in nav items
- **Edge cases**: Input fields should not trigger shortcuts; Ctrl on Windows/Linux vs Cmd on Mac

### FR-5: Global Command Palette (Cmd+K)
- **What**: A command palette dialog triggered by `Cmd+K` (or `Ctrl+K`). Uses `CommandDialog` from S9b.5. Provides fuzzy search across: navigation pages, recent papers (mocked), collections (mocked), and quick actions (new chat, upload paper, etc.).
- **Inputs**: User query string, keyboard shortcut (Cmd+K)
- **Outputs**: Filtered list of matching items grouped by category; selecting an item navigates or executes the action
- **Edge cases**: Empty query shows recent/suggested items; no results shows empty state; dialog closes on selection; Escape closes dialog

### FR-6: Breadcrumb Trail with Dropdown
- **What**: Enhance existing `Breadcrumbs` component to show a dropdown on intermediate segments, listing sibling pages for quick navigation. Uses `DropdownMenu` primitive.
- **Inputs**: Current pathname segments
- **Outputs**: Clickable breadcrumb segments with dropdown showing sibling routes
- **Edge cases**: Root path (no breadcrumbs), single-segment paths, dynamic segments (e.g., paper IDs)

### FR-7: Notification Bell with Badge Count
- **What**: Add a notification bell icon in the header with an unread count badge. Initially shows a static count (UI-only, no backend integration yet). Clicking opens a dropdown with placeholder notification items.
- **Inputs**: Notification count (static/mocked)
- **Outputs**: Bell icon with red badge showing count; dropdown with notification items on click
- **Edge cases**: Zero notifications (no badge), count > 99 shows "99+"

## Tangible Outcomes
- [ ] Sidebar shows gradient logo mark (gradient bg on icon container)
- [ ] Active nav items have a 3px left border accent + subtle bg highlight
- [ ] Collapsed sidebar shows tooltips with label + shortcut on icon hover
- [ ] Nav items display keyboard shortcut hints (⌘1, ⌘2, etc.) when expanded
- [ ] Cmd/Ctrl+1-6 navigate to corresponding pages
- [ ] Cmd/Ctrl+K opens command palette with fuzzy search
- [ ] Command palette groups results: Pages, Papers, Collections, Actions
- [ ] Selecting a command palette item navigates/executes and closes dialog
- [ ] Breadcrumb segments show dropdown with sibling routes
- [ ] Notification bell in header with badge count
- [ ] All existing layout tests still pass
- [ ] New components have ≥90% test coverage

## Test-Driven Requirements

### Tests to Write First
1. `sidebar.test.tsx` (enhance existing): Test gradient logo mark renders, collapse animation classes
2. `sidebar-nav-item.test.tsx` (enhance existing): Test left border accent on active items, keyboard shortcut hint display, tooltip on collapsed hover
3. `command-palette.test.tsx` (new): Test Cmd+K opens dialog, fuzzy search filters items, item selection navigates, Escape closes, empty state renders
4. `breadcrumbs.test.tsx` (enhance existing): Test dropdown renders on segment click, sibling routes listed
5. `notification-bell.test.tsx` (new): Test bell renders with badge count, zero hides badge, 99+ truncation, dropdown opens on click
6. `keyboard-shortcuts.test.tsx` (new): Test Cmd+1-6 navigate to correct pages, shortcuts disabled in input fields

### Mocking Strategy
- Mock `next/navigation` (`usePathname`, `useRouter`)
- Mock `localStorage` for sidebar state
- No external services to mock (pure frontend spec)

### Coverage
- All new components and enhanced components tested
- Keyboard shortcut edge cases (input focus, modifier keys)
- Accessibility: aria-labels, keyboard navigation, screen reader support
