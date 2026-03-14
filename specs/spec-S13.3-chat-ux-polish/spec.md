# Spec S13.3 — Chat UX Polish

## Overview
Premium chat experience upgrade for the RAG chatbot interface. Enhances message rendering with markdown support, code syntax highlighting, rich citation cards, suggested follow-up questions, animated typing indicator, message timestamps, smooth scroll animations, and an empty state with gradient illustration.

## Dependencies
- **S9.4** (Chat interface with streaming) — done
- **S9b.3** (Frontend infrastructure dependencies) — done (provides react-markdown, remark-gfm, rehype-highlight, framer-motion)

## Target Location
`frontend/src/components/chat/`

## Functional Requirements

### FR-1: Markdown Rendering in Messages
- **What**: Render assistant messages as rich markdown using react-markdown with remark-gfm (tables, strikethrough) and rehype-highlight (syntax highlighting)
- **Inputs**: Message content string (may contain markdown, code blocks, lists, tables)
- **Outputs**: Rendered HTML with proper styling for headings, lists, code blocks, tables, blockquotes
- **Edge cases**: Empty content, malformed markdown, very long code blocks, nested formatting

### FR-2: Code Block Enhancement
- **What**: Code blocks get syntax highlighting (via rehype-highlight or shiki), a language label, and a copy-to-clipboard button
- **Inputs**: Fenced code block with optional language identifier (```python ... ```)
- **Outputs**: Highlighted code with copy button that shows "Copied!" feedback
- **Edge cases**: No language specified (fallback to plain text), very long code, clipboard API unavailable

### FR-3: Rich Citation Cards
- **What**: Replace plain citation links with rich cards showing paper title, authors, year, and arXiv link with visual styling
- **Inputs**: Citation data from chat API response (sources array with title, authors, arxiv_id, year)
- **Outputs**: Visually distinct citation cards with hover effects, clickable arXiv links
- **Edge cases**: Missing fields (no authors, no year), many citations (5+), very long titles

### FR-4: Suggested Follow-up Chips
- **What**: After each assistant response, display 2-3 suggested follow-up question chips that the user can click to ask
- **Inputs**: Suggested questions from API response (or generated client-side from context)
- **Outputs**: Clickable chip buttons that populate the input field and optionally auto-send
- **Edge cases**: No suggestions available, very long suggestion text, rapid clicking

### FR-5: Enhanced Typing Indicator
- **What**: Animated typing indicator with smooth dot animation while waiting for assistant response
- **Inputs**: Loading/streaming state boolean
- **Outputs**: Animated dots indicator with smooth entrance/exit transitions (framer-motion)
- **Edge cases**: Very fast responses (indicator should not flash), connection errors during streaming

### FR-6: Message Timestamps
- **What**: Display timestamps on messages showing when they were sent/received
- **Inputs**: Message timestamp (ISO string or Date)
- **Outputs**: Formatted relative time ("just now", "2 min ago") or absolute time for older messages
- **Edge cases**: Missing timestamp, timezone differences, messages from previous sessions

### FR-7: Smooth Scroll Animations
- **What**: Auto-scroll to latest message with smooth animation, with a "scroll to bottom" button when user scrolls up
- **Inputs**: New message events, user scroll position
- **Outputs**: Smooth scroll behavior, floating "scroll to bottom" FAB when not at bottom
- **Edge cases**: Very long messages, rapid message sequence, user actively reading older messages

### FR-8: Empty State with Gradient Illustration
- **What**: When no messages exist, show an engaging empty state with gradient illustration, welcome text, and example prompts
- **Inputs**: Empty message array
- **Outputs**: Styled empty state with gradient background, icon/illustration, welcome message, and 3-4 clickable example research questions
- **Edge cases**: Transition from empty to first message should be smooth

## Tangible Outcomes
- [ ] Assistant messages render markdown (headings, lists, bold, italic, links, tables)
- [ ] Code blocks have syntax highlighting and a working copy button
- [ ] Citation sources display as rich cards with title, authors, year, arXiv link
- [ ] Follow-up suggestion chips appear after assistant messages and are clickable
- [ ] Typing indicator animates smoothly with framer-motion entrance/exit
- [ ] Messages show relative timestamps
- [ ] Chat auto-scrolls to new messages with smooth animation
- [ ] Scroll-to-bottom button appears when user scrolls up
- [ ] Empty state shows gradient illustration with clickable example prompts
- [ ] All components have Vitest tests
- [ ] No accessibility regressions (ARIA labels, keyboard navigation)

## Test-Driven Requirements

### Tests to Write First
1. `test_markdown_rendering`: Verify headings, lists, code blocks, links render correctly
2. `test_code_copy_button`: Verify copy button appears on code blocks and copies content
3. `test_citation_cards`: Verify rich citation cards render with title, authors, year, link
4. `test_citation_card_missing_fields`: Verify graceful handling of missing citation data
5. `test_followup_chips`: Verify suggestion chips render and trigger input population on click
6. `test_typing_indicator_animation`: Verify typing indicator shows/hides with animation classes
7. `test_message_timestamps`: Verify relative time formatting ("just now", "2 min ago", etc.)
8. `test_scroll_to_bottom`: Verify scroll-to-bottom button visibility based on scroll position
9. `test_empty_state`: Verify empty state renders with example prompts and gradient styling
10. `test_empty_state_click`: Verify clicking example prompt populates input

### Mocking Strategy
- Mock `navigator.clipboard.writeText` for copy-to-clipboard tests
- Mock `Date.now()` for consistent timestamp testing
- Mock `IntersectionObserver` for scroll detection
- Use React Testing Library for component rendering and interaction
- Mock framer-motion animations for deterministic testing

### Coverage
- All public components tested
- Edge cases (missing data, empty states) covered
- User interactions (click, scroll) tested
- Accessibility attributes verified
