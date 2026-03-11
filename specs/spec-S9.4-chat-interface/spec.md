# Spec S9.4 -- Chat Interface (Streaming)

## Overview
RAG chatbot interface for PaperAlchemy's Next.js frontend. Provides a conversational UI with SSE streaming, follow-up Q&A, message history, and citation rendering with clickable arXiv links. This is the primary research interaction interface.

Since the backend chat API (S7.3) is not yet implemented, this spec builds the full UI with a mock API layer that simulates SSE streaming responses. The page must be fully functional with mock data and ready to wire to the real `/api/v1/chat` endpoint.

## Dependencies
- S9.2 (Layout & navigation) — **done**

## Target Location
- `frontend/src/app/chat/page.tsx` — Chat page
- `frontend/src/components/chat/` — Chat components
- `frontend/src/lib/api/chat.ts` — Chat API functions (mock + real)
- `frontend/src/types/chat.ts` — Chat type definitions

## Functional Requirements

### FR-1: Message Input
- **What**: Fixed-bottom text input with send button for composing research questions
- **Inputs**: Message text (string), Enter key or click to submit, Shift+Enter for newline
- **Outputs**: Appends user message to history, triggers API call
- **Edge cases**: Empty message (disable send), very long message (character limit 2000), whitespace-only treated as empty
- **Details**:
  - Auto-growing textarea (1-5 lines)
  - Send button (icon), disabled when empty or streaming
  - Keyboard: Enter to send, Shift+Enter for newline
  - Placeholder: "Ask a research question..."
  - Disabled during streaming response

### FR-2: Message History
- **What**: Scrollable chat area displaying user and assistant messages
- **Inputs**: Array of messages (role: user | assistant, content, sources)
- **Outputs**: Rendered message bubbles with role-appropriate styling
- **Edge cases**: No messages (show welcome prompt), single message, very long messages
- **Details**:
  - User messages: right-aligned, primary color background
  - Assistant messages: left-aligned, muted background, wider max-width
  - Auto-scroll to bottom on new messages
  - Scroll-to-bottom button when scrolled up
  - Timestamps (relative: "just now", "2m ago")
  - Message ordering: chronological, newest at bottom

### FR-3: Streaming Response Display
- **What**: Token-by-token rendering of assistant responses via SSE
- **Inputs**: SSE event stream from API (or mock)
- **Outputs**: Progressive text display with typing indicator, then final rendered message
- **Edge cases**: Stream interruption (show error + retry), empty response, connection timeout
- **Details**:
  - Typing indicator (animated dots) before first token
  - Tokens appended progressively (no flicker)
  - Markdown rendering in final message (bold, italic, lists, code blocks)
  - "Stop generating" button during stream
  - Sources appended after stream completes

### FR-4: Citation Rendering
- **What**: Inline citation references [1], [2] rendered as clickable links, with source list at bottom of message
- **Inputs**: Message content with inline `[1]`, `[2]` references + sources array
- **Outputs**: Clickable citation badges that scroll to/highlight source, source cards with arXiv links
- **Edge cases**: No citations (render plain text), broken references (show as plain text), missing source for reference number
- **Details**:
  - Inline `[1]` rendered as superscript badges
  - Clicking badge scrolls to source in source list
  - Source list at bottom of assistant message:
    - Paper title (clickable → arXiv)
    - Authors (truncated to first 3)
    - Year
    - arXiv link (opens in new tab)
  - Source cards have subtle border and academic styling

### FR-5: Session Management
- **What**: Chat session persistence with new chat / clear history actions
- **Inputs**: Session ID (auto-generated UUID), user actions (new chat, clear)
- **Outputs**: Session state persisted in memory (not localStorage for now), session ID sent with API calls
- **Edge cases**: First visit (new session), browser refresh (messages lost — acceptable for now)
- **Details**:
  - "New Chat" button in header/sidebar clears messages and generates new session ID
  - Session ID included in API requests for backend conversation memory
  - No localStorage persistence in this spec (future enhancement)
  - Welcome message shown on new session

### FR-6: Welcome State
- **What**: Initial empty state with greeting and suggested questions
- **Inputs**: No messages in session
- **Outputs**: Welcome card with description and 3-4 clickable suggested questions
- **Edge cases**: After clearing history, return to welcome state
- **Details**:
  - Title: "PaperAlchemy Research Assistant"
  - Subtitle: "Ask me anything about research papers in the knowledge base"
  - Suggested questions (clickable, auto-fill input):
    - "What are the latest advances in transformer architectures?"
    - "Explain the key contributions of attention mechanisms"
    - "Compare BERT and GPT approaches to NLP"
    - "What methods are used for efficient fine-tuning of LLMs?"

### FR-7: Error Handling
- **What**: Graceful error handling for API failures
- **Inputs**: API error response, network failure, stream interruption
- **Outputs**: Error message inline in chat with retry button
- **Edge cases**: Multiple consecutive errors, timeout during streaming
- **Details**:
  - Error shown as a system message (distinct styling from user/assistant)
  - "Retry" button re-sends the last user message
  - Connection status indicator (optional: subtle dot in header)

### FR-8: Mock SSE Layer
- **What**: Mock API that simulates SSE streaming for development without backend
- **Inputs**: User message
- **Outputs**: Simulated token-by-token stream with mock citations and sources
- **Edge cases**: Simulated errors (10% chance for testing), simulated slow responses
- **Details**:
  - Mock responses include academic-sounding text with inline [1], [2] citations
  - Mock sources with real-looking paper metadata (titles, authors, arXiv IDs)
  - Configurable delay between tokens (30-50ms)
  - Switch to real API via environment variable `NEXT_PUBLIC_API_URL`

## Tangible Outcomes
- [ ] Chat page renders at `/chat` route
- [ ] Message input accepts text and submits on Enter
- [ ] User messages appear right-aligned in chat history
- [ ] Assistant messages stream token-by-token with typing indicator
- [ ] "Stop generating" button halts streaming
- [ ] Inline citations [1], [2] rendered as clickable superscript badges
- [ ] Source list with paper title, authors, year, arXiv link at end of assistant messages
- [ ] ArXiv links open in new tab
- [ ] "New Chat" button clears history and shows welcome state
- [ ] Welcome state shows suggested questions that auto-fill input
- [ ] Error messages shown inline with retry button
- [ ] Auto-scroll to bottom on new messages
- [ ] Scroll-to-bottom button when scrolled up
- [ ] Responsive layout (mobile + desktop)
- [ ] Mock SSE layer provides realistic streaming responses
- [ ] All components tested with Vitest + React Testing Library

## Test-Driven Requirements

### Tests to Write First
1. `test_chat_page_renders`: Page renders with message input and chat area
2. `test_message_input_submit`: Enter key submits message, Shift+Enter adds newline
3. `test_message_input_disabled_when_empty`: Send button disabled for empty/whitespace input
4. `test_message_input_disabled_during_streaming`: Input and send disabled while streaming
5. `test_user_message_display`: User messages render right-aligned with correct content
6. `test_assistant_message_display`: Assistant messages render left-aligned with markdown
7. `test_streaming_typing_indicator`: Typing indicator shown before first token
8. `test_streaming_progressive_display`: Tokens appear progressively during stream
9. `test_stop_generating`: Stop button halts stream and shows partial response
10. `test_citation_inline_rendering`: `[1]` in text rendered as superscript badge
11. `test_citation_source_list`: Source cards render with title, authors, year, arXiv link
12. `test_citation_arxiv_link`: ArXiv links have target="_blank" and rel="noopener"
13. `test_new_chat_button`: Clears messages and shows welcome state
14. `test_welcome_state`: Welcome message and suggested questions shown on empty session
15. `test_suggested_question_click`: Clicking suggestion fills input and submits
16. `test_error_display`: Error messages shown with distinct styling and retry button
17. `test_retry_button`: Retry re-sends last user message
18. `test_auto_scroll`: Chat scrolls to bottom on new message
19. `test_scroll_to_bottom_button`: Button appears when scrolled up, scrolls down on click
20. `test_session_id_generation`: New session gets UUID, new chat generates new UUID

### Mocking Strategy
- Mock SSE/EventSource for streaming responses
- Mock `next/navigation` (useRouter, usePathname)
- Use mock message data fixtures with citations and sources
- No real API calls in tests
- Mock `crypto.randomUUID()` for deterministic session IDs

### Coverage
- All public components tested
- Edge cases covered (empty input, stream interruption, no citations)
- Error paths tested (API failure, network error)
- Streaming lifecycle tested (start → tokens → complete)
