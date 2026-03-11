# Spec S3.1 -- Paper ORM Model & Repository

## Overview
Define the core Paper SQLAlchemy ORM model and an async repository layer for CRUD operations. The Paper model stores arXiv paper metadata (title, authors, abstract, categories, dates), PDF content (parsed text, sections), and parsing status. The repository provides async methods for create, read, update, delete, upsert, bulk upsert, and filtered queries (by date range, category, parsing status).

## Dependencies
- **S2.2** (Database layer) — async engine, sessions, Base class

## Target Location
- `src/models/__init__.py`
- `src/models/paper.py`
- `src/repositories/__init__.py`
- `src/repositories/paper.py`

## Functional Requirements

### FR-1: Paper ORM Model
- **What**: SQLAlchemy 2.0 mapped class with UUID primary key, arXiv metadata fields, PDF content fields, parsing status, and timestamps
- **Inputs**: N/A (model definition)
- **Outputs**: `Paper` class inheriting from `Base`
- **Fields**:
  - `id`: UUID primary key (server-default `gen_random_uuid()`)
  - `arxiv_id`: str, unique, indexed, not null
  - `title`: str (Text), not null
  - `authors`: list[str] (JSONB), not null
  - `abstract`: str (Text), not null
  - `categories`: list[str] (JSONB), not null
  - `published_date`: datetime (timezone-aware), not null
  - `updated_date`: datetime (timezone-aware), nullable
  - `pdf_url`: str, not null
  - `pdf_content`: str (Text), nullable (populated after parsing)
  - `sections`: list[dict] (JSONB), nullable (populated after parsing)
  - `parsing_status`: str, not null, default "pending" (enum: pending/processing/success/failed)
  - `parsing_error`: str (Text), nullable
  - `created_at`: datetime, server-default now()
  - `updated_at`: datetime, server-default now(), onupdate now()
- **Indexes**: arxiv_id (unique), published_date, parsing_status, categories (GIN)
- **Edge cases**: Duplicate arxiv_id insertion should raise IntegrityError

### FR-2: Paper Repository (Async)
- **What**: Async repository class accepting `AsyncSession`, providing CRUD + query methods
- **Methods**:
  - `create(data: PaperCreate) -> Paper` — insert new paper, flush to get ID
  - `get_by_id(paper_id: UUID) -> Paper | None` — lookup by UUID
  - `get_by_arxiv_id(arxiv_id: str) -> Paper | None` — lookup by arXiv ID
  - `exists(arxiv_id: str) -> bool` — check existence
  - `update(arxiv_id: str, data: PaperUpdate) -> Paper | None` — partial update
  - `update_parsing_status(arxiv_id, status, pdf_content?, sections?, error?) -> Paper | None`
  - `delete(arxiv_id: str) -> bool` — delete by arXiv ID
  - `upsert(data: PaperCreate) -> Paper` — insert or update on conflict (arxiv_id)
  - `bulk_upsert(papers: list[PaperCreate]) -> int` — batch upsert, return count
  - `get_by_date_range(from_date, to_date, limit, offset) -> list[Paper]`
  - `get_by_category(category, limit, offset) -> list[Paper]`
  - `get_pending_parsing(limit) -> list[Paper]` — papers with status="pending"
  - `search(query?, category?, from_date?, to_date?, parsing_status?, limit, offset) -> list[Paper]`
  - `count(parsing_status?) -> int` — count papers, optionally filtered
- **Edge cases**: Non-existent IDs return None/False, empty bulk list returns 0

### FR-3: Pydantic Schemas
- **What**: Pydantic models for API input/output validation
- **Models**:
  - `PaperCreate` — all required fields for creating a paper
  - `PaperUpdate` — all fields optional (partial update)
  - `PaperResponse` — full paper representation for API responses (with `model_config = ConfigDict(from_attributes=True)`)

## Tangible Outcomes
- [ ] `src/models/paper.py` exists with `Paper` ORM class using UUID PK
- [ ] `src/models/__init__.py` re-exports `Paper`
- [ ] `src/repositories/paper.py` exists with async `PaperRepository`
- [ ] `src/repositories/__init__.py` re-exports `PaperRepository`
- [ ] `src/schemas/paper.py` exists with `PaperCreate`, `PaperUpdate`, `PaperResponse`
- [ ] All repository methods are async (`async def`)
- [ ] Upsert uses PostgreSQL `ON CONFLICT DO UPDATE`
- [ ] Tests cover all CRUD operations + edge cases
- [ ] Paper model registers with `Base.metadata`

## Test-Driven Requirements

### Tests to Write First
1. `test_paper_model_columns`: Verify all columns exist with correct types
2. `test_paper_model_table_name`: Table name is "papers"
3. `test_paper_model_indexes`: Required indexes exist
4. `test_paper_create`: Insert a paper, verify fields
5. `test_paper_get_by_id`: Retrieve by UUID
6. `test_paper_get_by_arxiv_id`: Retrieve by arXiv ID
7. `test_paper_exists`: Check existence
8. `test_paper_update`: Partial update
9. `test_paper_update_parsing_status`: Update status + content
10. `test_paper_delete`: Delete and verify gone
11. `test_paper_upsert_insert`: Upsert new paper
12. `test_paper_upsert_update`: Upsert existing paper (update fields)
13. `test_paper_bulk_upsert`: Bulk insert multiple papers
14. `test_paper_get_by_date_range`: Query by date range
15. `test_paper_get_by_category`: Query by arXiv category
16. `test_paper_get_pending_parsing`: Query pending papers
17. `test_paper_search_by_title`: Text search in title
18. `test_paper_search_multi_filter`: Combined filters
19. `test_paper_count`: Count with/without filter
20. `test_paper_not_found`: Get non-existent returns None
21. `test_pydantic_paper_create_validation`: PaperCreate validates input
22. `test_pydantic_paper_response_from_orm`: PaperResponse.model_validate(paper)

### Mocking Strategy
- Use an in-memory SQLite async engine OR a real PostgreSQL testcontainer
- For unit tests: use `aiosqlite` with `create_async_engine("sqlite+aiosqlite:///:memory:")`
- Note: JSONB and GIN indexes are PostgreSQL-specific; unit tests may need to handle this gracefully (use JSON type fallback for SQLite)

### Coverage
- All public methods tested
- Edge cases: empty results, duplicate inserts, None returns
- Error paths: IntegrityError on duplicate arxiv_id
