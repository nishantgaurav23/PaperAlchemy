# Checklist -- Spec S4.3: Embedding Service (Jina AI)

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (Environment Config) is "done"
- [x] Create `src/services/embeddings/` directory
- [x] Create `src/schemas/embeddings.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_embedding_schemas.py`
- [x] Create `tests/unit/test_embedding_client.py`
- [x] Create `tests/unit/test_embedding_factory.py`
- [x] Write failing tests for FR-1 (schemas)
- [x] Write failing tests for FR-2 (embed_passages)
- [x] Write failing tests for FR-3 (embed_query)
- [x] Write failing tests for FR-4 (context manager)
- [x] Write failing tests for FR-5 (factory)
- [x] Write failing tests for FR-6 (error handling)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `src/schemas/embeddings.py` — request/response models
- [x] Implement `src/services/embeddings/client.py` — JinaEmbeddingsClient
- [x] Implement `src/services/embeddings/factory.py` — make_embeddings_client
- [x] Implement `src/services/embeddings/__init__.py` — exports
- [x] Add `EmbeddingError` to `src/exceptions.py` (already existed as EmbeddingServiceError)
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire into dependency injection (`src/dependency.py`)
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets (API key from settings only)
- [x] Create notebook: `notebooks/specs/S4.3_embeddings.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
