# Spec S13.1 -- Premium Landing Page

## Overview
Build a premium, animated landing page for PaperAlchemy that showcases the platform's capabilities as an AI Research Assistant. The page features an animated hero section with mesh gradient background, a feature showcase grid (6 cards with icons and hover effects), a live stats counter (papers indexed, queries answered), prominent CTA buttons, a testimonial/use-case section, and a footer with links.

## Dependencies
- S9.2 (Layout & Navigation) — **done**

## Target Location
- `frontend/src/app/page.tsx` — Main landing page (replace current placeholder)
- `frontend/src/components/landing/` — Landing page components

## Functional Requirements

### FR-1: Hero Section with Animated Mesh Gradient
- **What**: Full-width hero section with animated mesh gradient background, headline, subtext, and CTA buttons
- **Inputs**: None (static content)
- **Outputs**: Rendered hero with smooth gradient animation via CSS
- **Details**:
  - Headline: "Transform How You Read Research Papers" (or similar impactful copy)
  - Subtext: Brief description of PaperAlchemy's value proposition
  - Two CTA buttons: "Get Started" (primary, links to /chat) and "Explore Papers" (secondary, links to /search)
  - Mesh gradient uses CSS `@keyframes` animation (no JS animation library required)
  - Responsive: stacks vertically on mobile

### FR-2: Feature Showcase Grid
- **What**: 6-card grid showcasing key features with icons and hover lift effect
- **Inputs**: Static feature data array
- **Outputs**: Responsive grid (3×2 on desktop, 2×3 on tablet, 1×6 on mobile)
- **Details**:
  - Features: AI Chat, Paper Search, PDF Upload & Analysis, Citation-Backed Answers, Paper Comparison, Research Dashboard
  - Each card: icon (Lucide), title, short description
  - Hover effect: subtle lift (translateY) + shadow increase, 200ms transition
  - Cards use existing shadcn Card component

### FR-3: Live Stats Counter
- **What**: Animated counter section showing platform statistics
- **Inputs**: Stats fetched from `/api/v1/stats` endpoint (or fallback defaults)
- **Outputs**: Animated number counters with labels
- **Details**:
  - Stats: Papers Indexed, Questions Answered, Citations Generated (minimum 3)
  - Numbers animate from 0 to target value on scroll into view (Intersection Observer)
  - Fallback values if API unavailable: show reasonable defaults
  - Counter animation: ~2s duration, easing function

### FR-4: Use Cases / Testimonial Section
- **What**: Section showcasing 3 use cases for the platform
- **Inputs**: Static use case data
- **Outputs**: Styled cards or blocks with use case descriptions
- **Details**:
  - Use cases: "Literature Review", "Stay Current", "Deep Dive Analysis"
  - Each with icon, title, and 1-2 sentence description
  - Clean layout with subtle background differentiation

### FR-5: Footer
- **What**: Page footer with navigation links and branding
- **Inputs**: Static link data
- **Outputs**: Responsive footer
- **Details**:
  - Sections: Product (features links), Resources (docs, API), Connect (GitHub)
  - PaperAlchemy branding + copyright
  - Responsive: multi-column on desktop, stacked on mobile

## Tangible Outcomes
- [ ] Landing page renders at `/` with hero, features, stats, use cases, and footer sections
- [ ] Hero section has animated mesh gradient background (CSS keyframes)
- [ ] Feature grid shows 6 cards with hover lift effect
- [ ] Stats counter animates numbers on scroll into view
- [ ] CTA buttons navigate to `/chat` and `/search`
- [ ] Page is fully responsive (mobile, tablet, desktop)
- [ ] All components have unit tests
- [ ] No hardcoded API URLs (uses environment-based config)

## Test-Driven Requirements

### Tests to Write First
1. `test_hero_renders`: Hero section renders headline, subtext, and CTA buttons
2. `test_cta_links`: CTA buttons have correct href to /chat and /search
3. `test_feature_grid_renders_6_cards`: Feature grid renders exactly 6 feature cards
4. `test_feature_cards_have_content`: Each feature card has icon, title, and description
5. `test_stats_section_renders`: Stats section renders with counter elements
6. `test_use_cases_render`: Use case section renders 3 use cases
7. `test_footer_renders`: Footer renders with navigation links
8. `test_responsive_classes`: Components use responsive Tailwind classes

### Mocking Strategy
- Mock `fetch` for stats API call (or use MSW)
- No external service dependencies — page is mostly static
- Use React Testing Library for component rendering tests

### Coverage
- All landing page components tested
- Hover/animation classes verified via className assertions
- Link targets verified
- Responsive class presence verified
