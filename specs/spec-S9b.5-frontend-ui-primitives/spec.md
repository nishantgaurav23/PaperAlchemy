# Spec S9b.5 — Missing UI Primitives

## Overview
Add shadcn/ui (base-nova style) components needed by P13–P23 feature phases. The project already has Button, Input, Badge, Skeleton, and Select. This spec fills the remaining gaps: Dialog/Modal, Dropdown Menu, Popover, Tabs, Textarea, Checkbox, Tooltip, Avatar, Toast (Sonner), Command (cmdk for Cmd+K palette), and Sheet (mobile drawers).

All components follow the existing pattern: `"use client"` directive, Radix UI / Base UI primitives underneath, `cn()` utility for className merging, `data-slot` attributes, and CVA variants where applicable.

## Dependencies
- **S9b.3** (Frontend infrastructure dependencies) — `done` — provides `cmdk`, `sonner`, `framer-motion`, and the base shadcn/ui setup

## Target Location
`frontend/src/components/ui/`

## Functional Requirements

### FR-1: Dialog / Modal
- **What**: Accessible modal dialog with overlay, title, description, close button, and footer actions
- **Primitives**: `@radix-ui/react-dialog`
- **Exports**: Dialog, DialogTrigger, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription, DialogClose
- **Edge cases**: Escape key closes, click-outside closes, focus trap, scroll lock on body

### FR-2: Dropdown Menu
- **What**: Context/action menu with items, separators, checkboxes, radio groups, and sub-menus
- **Primitives**: `@radix-ui/react-dropdown-menu`
- **Exports**: DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuCheckboxItem, DropdownMenuRadioItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuGroup, DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuSubContent
- **Edge cases**: Keyboard navigation (arrow keys, Enter, Escape), nested sub-menus

### FR-3: Popover
- **What**: Floating content anchored to a trigger element
- **Primitives**: `@radix-ui/react-popover`
- **Exports**: Popover, PopoverTrigger, PopoverContent, PopoverAnchor
- **Edge cases**: Auto-placement when near viewport edges, close on outside click

### FR-4: Tabs
- **What**: Tabbed content panels with keyboard navigation
- **Primitives**: `@radix-ui/react-tabs`
- **Exports**: Tabs, TabsList, TabsTrigger, TabsContent
- **Edge cases**: Arrow key navigation between tabs, disabled tabs, controlled/uncontrolled modes

### FR-5: Textarea
- **What**: Multi-line text input with consistent styling
- **Primitives**: Native `<textarea>` (no Radix needed)
- **Exports**: Textarea
- **Edge cases**: Disabled state, placeholder, auto-resize (optional), aria-invalid

### FR-6: Checkbox
- **What**: Accessible checkbox with checked, unchecked, and indeterminate states
- **Primitives**: `@radix-ui/react-checkbox`
- **Exports**: Checkbox
- **Edge cases**: Indeterminate state, disabled, form integration with label

### FR-7: Tooltip
- **What**: Hover/focus tooltip for supplementary information
- **Primitives**: `@radix-ui/react-tooltip`
- **Exports**: TooltipProvider, Tooltip, TooltipTrigger, TooltipContent
- **Edge cases**: Delay before show, keyboard focusable triggers, collision avoidance

### FR-8: Avatar
- **What**: User/author avatar with image and fallback (initials or icon)
- **Primitives**: `@radix-ui/react-avatar`
- **Exports**: Avatar, AvatarImage, AvatarFallback
- **Edge cases**: Image load failure → fallback, different sizes

### FR-9: Toast (Sonner)
- **What**: Toast notification container using Sonner (already installed)
- **Primitives**: `sonner` package
- **Exports**: Toaster component (wraps Sonner's `<Toaster />`)
- **Edge cases**: Multiple simultaneous toasts, different types (success, error, info), auto-dismiss

### FR-10: Command (cmdk)
- **What**: Command palette / combobox for Cmd+K search across papers, actions, navigation
- **Primitives**: `cmdk` package (already installed)
- **Exports**: Command, CommandDialog, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem, CommandSeparator
- **Edge cases**: Empty state, keyboard navigation, search filtering, modal vs inline modes

### FR-11: Sheet (Mobile Drawer)
- **What**: Slide-in panel from edge of screen for mobile navigation/filters
- **Primitives**: `@radix-ui/react-dialog` (reuses dialog primitive with sheet styling)
- **Exports**: Sheet, SheetTrigger, SheetContent, SheetHeader, SheetFooter, SheetTitle, SheetDescription, SheetClose
- **Edge cases**: Slide direction (top, right, bottom, left), overlay, scroll within sheet

## Tangible Outcomes
- [ ] 11 new component files in `frontend/src/components/ui/`
- [ ] Each component has a co-located Vitest test file (`*.test.tsx`)
- [ ] All tests pass: `cd frontend && pnpm test`
- [ ] All components export from their respective files with proper TypeScript types
- [ ] Radix UI packages installed: `@radix-ui/react-dialog`, `@radix-ui/react-dropdown-menu`, `@radix-ui/react-popover`, `@radix-ui/react-tabs`, `@radix-ui/react-checkbox`, `@radix-ui/react-tooltip`, `@radix-ui/react-avatar`
- [ ] Lint passes: `cd frontend && pnpm lint`
- [ ] Components follow existing pattern (cn utility, data-slot, "use client")

## Test-Driven Requirements

### Tests to Write First
1. **test_dialog**: renders trigger, opens on click, shows content, closes on Escape
2. **test_dropdown_menu**: renders trigger, opens menu, shows items, keyboard navigation
3. **test_popover**: renders trigger, toggles content visibility
4. **test_tabs**: renders tab list, switches content on tab click, keyboard nav
5. **test_textarea**: renders with placeholder, handles disabled state, applies className
6. **test_checkbox**: renders, toggles checked state on click, disabled state
7. **test_tooltip**: renders trigger, shows content on hover/focus (may need async)
8. **test_avatar**: renders image, shows fallback when image fails
9. **test_toast**: Toaster component renders, toast function triggers notification
10. **test_command**: renders input, shows items, filters on type, keyboard selection
11. **test_sheet**: renders trigger, opens sheet, shows content, closes

### Mocking Strategy
- No external services to mock — these are pure UI components
- Use `@testing-library/react` + `@testing-library/user-event` for interactions
- Mock `window.matchMedia` if needed for responsive components
- Sonner toast tests may need to verify DOM updates with `waitFor`

### Coverage
- All exported components rendered
- Interactive states tested (open/close, checked/unchecked)
- Accessibility roles verified (dialog, menu, tab, checkbox)
- Disabled states tested where applicable
