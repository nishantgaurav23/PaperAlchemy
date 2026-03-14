# Spec S13.2 — Design System Overhaul

## Overview
Overhaul the default shadcn/ui design system with a refined, premium aesthetic for PaperAlchemy. This includes a rich indigo/violet primary color palette, glassmorphism card effects, an 8px spacing grid, a polished typography scale (Inter + Plus Jakarta Sans), gradient accents on primary surfaces, an elevation system (shadow-sm → shadow-2xl), accessible focus rings, and smooth 200ms transitions on all interactive elements.

## Dependencies
- **S9.1** (Next.js 15 + TS + Tailwind + shadcn/ui setup) — **done**

## Target Location
- `frontend/src/app/globals.css` — CSS custom properties (color palette, spacing, typography, elevation)
- `frontend/src/components/ui/` — updated shadcn/ui primitives with new design tokens

## Functional Requirements

### FR-1: Color Palette (Indigo/Violet Primary)
- **What**: Replace the default neutral (oklch grayscale) palette with a rich indigo/violet primary. Define semantic color tokens for: primary, secondary, accent, muted, destructive, success, warning. Both light and dark mode.
- **Tokens**:
  - `--primary`: indigo-600 equivalent (oklch)
  - `--primary-foreground`: white
  - `--accent`: violet-500 equivalent
  - `--secondary`: slate with slight violet tint
  - `--muted`: light lavender (light) / dark slate (dark)
  - `--success`: emerald-500
  - `--warning`: amber-500
  - All sidebar tokens updated to match
- **Edge cases**: Contrast ratio ≥ 4.5:1 for text on colored backgrounds (WCAG AA)

### FR-2: Glassmorphism Card Styles
- **What**: Add glassmorphism utility classes for cards: `backdrop-blur`, semi-transparent backgrounds, subtle border opacity.
- **Classes**:
  - `.glass-card`: `backdrop-blur-xl bg-white/70 dark:bg-slate-900/70 border border-white/20 dark:border-white/10`
  - `.glass-card-elevated`: Same + stronger shadow
- **Edge cases**: Fallback for browsers without backdrop-filter support

### FR-3: 8px Spacing Grid
- **What**: Ensure all spacing uses multiples of 8px. Define custom spacing tokens if needed. Document the spacing scale.
- **Scale**: 0, 0.5 (4px), 1 (8px), 1.5 (12px), 2 (16px), 3 (24px), 4 (32px), 5 (40px), 6 (48px), 8 (64px), 10 (80px), 12 (96px)
- **Note**: Tailwind's default spacing already follows a 4px base; we enforce 8px multiples in component patterns.

### FR-4: Typography Scale
- **What**: Configure Inter as the primary sans-serif and Plus Jakarta Sans as the display/heading font. Define a type scale with consistent sizing.
- **Scale**:
  - `--font-sans`: Inter (body text)
  - `--font-display`: Plus Jakarta Sans (headings, hero text)
  - Heading sizes: h1 (2.25rem/36px), h2 (1.875rem/30px), h3 (1.5rem/24px), h4 (1.25rem/20px)
  - Body: base (1rem/16px), sm (0.875rem/14px), xs (0.75rem/12px)
  - Line heights: headings (1.2), body (1.6)
- **Edge cases**: Font loading — use `font-display: swap` to avoid FOIT

### FR-5: Gradient Accents
- **What**: Define reusable gradient CSS custom properties for primary surfaces (buttons, hero sections, highlights).
- **Gradients**:
  - `--gradient-primary`: indigo-600 → violet-500
  - `--gradient-accent`: violet-500 → purple-400
  - `--gradient-surface`: subtle background gradient for cards/sections
- **Utilities**: `.bg-gradient-primary`, `.bg-gradient-accent`, `.text-gradient` (gradient text via background-clip)

### FR-6: Elevation System
- **What**: Define a consistent shadow/elevation scale from subtle to dramatic.
- **Scale**:
  - `--shadow-xs`: `0 1px 2px rgba(0,0,0,0.05)`
  - `--shadow-sm`: `0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06)`
  - `--shadow-md`: `0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06)`
  - `--shadow-lg`: `0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05)`
  - `--shadow-xl`: `0 20px 25px rgba(0,0,0,0.1), 0 8px 10px rgba(0,0,0,0.04)`
  - `--shadow-2xl`: `0 25px 50px rgba(0,0,0,0.25)`
  - Dark mode: Use oklch-based shadows with lower opacity
- **Utilities**: Map to Tailwind shadow classes

### FR-7: Focus Rings & Transitions
- **What**: Standardize focus ring appearance (indigo/violet offset ring) and apply 200ms transitions to all interactive elements.
- **Focus**: `outline: 2px solid var(--ring)` with `outline-offset: 2px`, ring color = primary
- **Transitions**: `transition-all duration-200 ease-in-out` on buttons, links, cards, inputs
- **Edge cases**: `prefers-reduced-motion` — disable transitions for users who request it

## Tangible Outcomes
- [ ] `globals.css` has indigo/violet color palette for both light and dark mode
- [ ] Glassmorphism `.glass-card` class works with backdrop-blur
- [ ] Typography uses Inter (body) + Plus Jakarta Sans (headings) with correct scale
- [ ] Gradient utilities (`.bg-gradient-primary`, `.text-gradient`) are defined
- [ ] Elevation shadow scale (xs → 2xl) is defined
- [ ] Focus rings use primary color with offset
- [ ] 200ms transitions on all interactive elements
- [ ] `prefers-reduced-motion` respected
- [ ] WCAG AA contrast ratios met for primary text on backgrounds
- [ ] All existing shadcn/ui components render correctly with new tokens

## Test-Driven Requirements

### Tests to Write First
1. `design-system.test.ts`: Validate CSS custom properties exist in the DOM after rendering
2. `design-system.test.ts`: Validate glassmorphism class applies correct styles
3. `design-system.test.ts`: Validate font families are set correctly
4. `design-system.test.ts`: Validate gradient utilities render
5. `design-system.test.ts`: Validate focus ring styles on interactive elements
6. `design-system.test.ts`: Validate dark mode token switching
7. `design-system.test.ts`: Validate prefers-reduced-motion disables transitions
8. `design-system.test.ts`: Visual regression — existing Button, Card, Input components render without errors

### Mocking Strategy
- No external services to mock (pure frontend CSS/styling)
- Use React Testing Library to render components and inspect computed styles
- Use JSDOM for CSS custom property testing (limited — test class application, not computed values)
- Snapshot tests for component visual regression

### Coverage
- All new CSS custom properties have corresponding tests
- All utility classes tested for correct class application
- Dark mode toggle tested
- Existing components regression tested
