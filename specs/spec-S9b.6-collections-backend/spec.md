# Spec S9b.6 — Collections Backend API

## Overview
Migrate collections from frontend localStorage to a proper backend API. This provides a `Collection` ORM model with a many-to-many relationship to `Paper`, a `CollectionRepository` for CRUD operations, Pydantic schemas, and REST API endpoints under `/api/v1/collections`. The `user_id` field is nullable for now (no auth yet — S14.1 will add real user ownership). Required by P15 (Annotations), P17 (Shared Collections).

## Dependencies
- **S2.4** (Dependency injection) — `done` — provides FastAPI `Depends()` pattern
- **S3.1** (Paper model & repository) — `done` — provides `Paper` model and `PaperRepository`

## Target Locations
- `src/models/collection.py` — Collection ORM model + association table
- `src/repositories/collection.py` — CollectionRepository (async CRUD)
- `src/schemas/collection.py` — Pydantic request/response schemas
- `src/routers/collections.py` — REST API endpoints
- `src/dependency.py` — CollectionRepoDep injection alias
- `src/main.py` — router registration
- `tests/unit/test_collections.py` — unit tests

## Functional Requirements

### FR-1: Collection Model
- **Table**: `collections`
- **Fields**:
  - `id` — UUID primary key, auto-generated
  - `name` — String(255), required, not null
  - `description` — Text, nullable
  - `user_id` — UUID, nullable (no auth yet; will be FK to users table after S14.1)
  - `created_at` — DateTime with timezone, server default `now()`
  - `updated_at` — DateTime with timezone, auto-updated on change
- **Association table**: `collection_papers` (M2M)
  - `collection_id` — FK to `collections.id`, ON DELETE CASCADE
  - `paper_id` — FK to `papers.id`, ON DELETE CASCADE
  - Composite primary key `(collection_id, paper_id)`
  - `added_at` — DateTime, server default `now()`
- **Relationships**: `papers` relationship on Collection (lazy="selectin")

### FR-2: Collection Repository
- `create(data: CollectionCreate) -> Collection` — insert new collection
- `get_by_id(collection_id: UUID) -> Collection | None` — fetch with papers eagerly loaded
- `list_all(user_id: UUID | None, limit, offset) -> list[Collection]` — list collections, optionally filtered by user_id
- `update(collection_id: UUID, data: CollectionUpdate) -> Collection | None` — partial update
- `delete(collection_id: UUID) -> bool` — delete collection (cascade removes M2M links)
- `add_paper(collection_id: UUID, paper_id: UUID) -> bool` — add paper to collection (idempotent)
- `remove_paper(collection_id: UUID, paper_id: UUID) -> bool` — remove paper from collection
- `get_collection_papers(collection_id: UUID) -> list[Paper]` — get all papers in a collection
- `count(user_id: UUID | None) -> int` — count collections

### FR-3: Pydantic Schemas
- `CollectionCreate` — name (required), description (optional)
- `CollectionUpdate` — name (optional), description (optional)
- `CollectionResponse` — id, name, description, user_id, paper_count, created_at, updated_at
- `CollectionDetailResponse` — extends CollectionResponse with list of PaperResponse
- `CollectionPaperAction` — paper_id (UUID) for add/remove operations

### FR-4: REST API Endpoints
| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/v1/collections` | `list_collections` | List all collections (optional `?user_id=` filter, pagination) |
| POST | `/api/v1/collections` | `create_collection` | Create new collection |
| GET | `/api/v1/collections/{id}` | `get_collection` | Get collection with papers |
| PUT | `/api/v1/collections/{id}` | `update_collection` | Update collection name/description |
| DELETE | `/api/v1/collections/{id}` | `delete_collection` | Delete collection |
| POST | `/api/v1/collections/{id}/papers` | `add_paper_to_collection` | Add paper to collection |
| DELETE | `/api/v1/collections/{id}/papers/{paper_id}` | `remove_paper_from_collection` | Remove paper from collection |

### FR-5: Error Handling
- 404 when collection not found
- 404 when paper not found (for add_paper)
- 409 if adding paper already in collection (or silently succeed — idempotent)
- Standard error response format

## Non-Functional Requirements
- All database operations async
- SQLite-compatible for tests (no PostgreSQL-specific features)
- Follows existing patterns in `PaperRepository` and `papers` router
- Cascade delete on collection removes M2M links, not the papers themselves

## TDD Notes
- Write tests FIRST against SQLite in-memory database
- Test all CRUD operations on repository
- Test API endpoints with FastAPI TestClient
- Mock nothing — use real async SQLite session (same pattern as `test_ingest_router.py`)
- Test edge cases: duplicate add, remove non-existent paper, delete non-existent collection

## Out of Scope
- User authentication (S14.1)
- Shared collections with team features (S17.1)
- Frontend migration to backend API (separate frontend spec)
