# Spec S9.7 -- Reading Lists & Collections

## Overview
Save and organize papers into named collections (reading lists). Users can create, rename, delete collections, add/remove papers, reorder papers via drag-and-drop, and generate shareable links. Collections are persisted in `localStorage` (no backend required for MVP). The collections page at `/collections` shows all collections with their paper counts, and individual collection views display the papers within.

## Dependencies
- S9.2 (Layout & navigation) — done

## Target Location
- `frontend/src/app/collections/page.tsx` — collections list page
- `frontend/src/app/collections/[id]/page.tsx` — single collection detail page
- `frontend/src/components/collections/` — collection components
- `frontend/src/lib/collections.ts` — collection storage & logic (localStorage)
- `frontend/src/types/collection.ts` — collection types

## Functional Requirements

### FR-1: Collection CRUD
- **What**: Create, read, update (rename), and delete collections
- **Inputs**: Collection name (string, 1-100 chars), collection ID (UUID)
- **Outputs**: Collection object with id, name, description, paperIds, createdAt, updatedAt
- **Edge cases**: Duplicate name (allow, distinguish by ID), empty name (reject), delete collection with papers (confirm dialog)
- **Storage**: `localStorage` key `paperalchemy-collections` — JSON array of Collection objects
- **Details**:
  - Create: modal/dialog with name + optional description
  - Rename: inline edit or dialog
  - Delete: confirmation dialog ("This will remove the collection but not the papers")
  - List: sorted by updatedAt descending (most recently modified first)

### FR-2: Add/Remove Papers to Collection
- **What**: Add papers to a collection from search results, paper detail, or collection view; remove papers from collection view
- **Inputs**: Paper object (or paper ID + minimal metadata), collection ID
- **Outputs**: Updated collection with paper added/removed
- **Edge cases**: Paper already in collection (no-op, show toast), adding to non-existent collection (error)
- **Details**:
  - "Add to Collection" button/dropdown on paper cards (search results, paper detail)
  - Shows list of existing collections + "Create New" option
  - Remove button (X icon) on papers within collection view
  - Toast notification on add/remove

### FR-3: Collection Detail View
- **What**: Display papers within a collection at `/collections/[id]`
- **Inputs**: Collection ID from URL params
- **Outputs**: Collection header (name, description, paper count, dates) + paper list
- **Edge cases**: Collection not found (404-like message), empty collection (empty state with CTA)
- **Details**:
  - Collection name as heading (editable inline)
  - Paper count + creation date
  - Paper cards in list format (title, authors, abstract preview, arXiv link)
  - Remove paper button per card
  - Back to collections link

### FR-4: Drag-and-Drop Reordering
- **What**: Reorder papers within a collection via drag-and-drop
- **Inputs**: Drag source index, drop target index
- **Outputs**: Updated paper order persisted to localStorage
- **Edge cases**: Single paper (no drag handles), empty collection (no drag targets)
- **Details**:
  - Use HTML5 drag-and-drop API (no extra dependencies)
  - Visual drag indicator (grab cursor, drop zone highlight)
  - Order persisted immediately to localStorage

### FR-5: Share Collection Link
- **What**: Generate a shareable link encoding the collection's paper IDs
- **Inputs**: Collection object
- **Outputs**: URL with encoded collection data (query params or hash)
- **Edge cases**: Very large collections (URL length limit — truncate with warning if >50 papers)
- **Details**:
  - "Share" button on collection detail view
  - Generates URL like `/collections/shared?data=<base64-encoded-json>`
  - Shared link loads a read-only view of the collection
  - Copy-to-clipboard with toast confirmation

### FR-6: Collections List Page
- **What**: `/collections` page showing all user collections
- **Inputs**: None (reads from localStorage)
- **Outputs**: Grid/list of collection cards
- **Edge cases**: No collections (empty state with "Create your first collection" CTA)
- **Details**:
  - Collection card: name, description preview, paper count, last updated
  - Create new collection button (prominent)
  - Delete collection button (with confirmation)
  - Click card → navigate to `/collections/[id]`

## Tangible Outcomes
- [ ] `/collections` page renders with empty state when no collections exist
- [ ] User can create a new collection with name and optional description
- [ ] User can rename and delete collections
- [ ] User can add papers to collections from a dropdown/popover
- [ ] Papers appear in collection detail view at `/collections/[id]`
- [ ] User can remove papers from a collection
- [ ] User can reorder papers via drag-and-drop
- [ ] Share button generates a copyable link encoding the collection
- [ ] Collections persist across page refreshes (localStorage)
- [ ] All components have co-located Vitest tests

## Test-Driven Requirements

### Tests to Write First
1. test_collection_storage: localStorage CRUD operations (create, read, update, delete)
2. test_add_remove_papers: Adding/removing papers updates collection correctly
3. test_reorder_papers: Drag-and-drop reorder updates paper order
4. test_share_link: Share link generation and parsing
5. test_collections_page: Collections list page renders empty state and collection cards
6. test_collection_detail_page: Collection detail page renders papers and handles missing collection
7. test_add_to_collection_popover: "Add to Collection" UI component
8. test_collection_card: Collection card displays correct metadata

### Mocking Strategy
- Mock `localStorage` with in-memory store in tests
- Mock `navigator.clipboard` for share link copy
- Mock `useRouter` / `useSearchParams` from next/navigation
- No backend API mocking needed (localStorage-only for MVP)

### Coverage
- All public functions in `collections.ts` tested
- All components render correctly with various states (empty, populated, error)
- Edge cases: duplicate names, empty names, large collections
