# Checklist -- Spec S3.4: Ingestion Pipeline (Airflow DAG)

## Phase 1: Setup & Dependencies
- [x] Verify S3.1 (Paper model) is "done"
- [x] Verify S3.2 (ArXiv client) is "done"
- [x] Verify S3.3 (PDF parser) is "done"
- [x] Create `airflow/dags/` directory structure
- [x] Create `src/routers/ingest.py`
- [x] Create `src/schemas/api/ingest.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_ingestion_dag.py` — DAG config + task tests
- [x] Create `tests/unit/test_ingest_router.py` — API endpoint tests
- [x] Write failing tests for DAG loading, task count, dependencies, schedule
- [x] Write failing tests for task functions (setup, fetch, report)
- [x] Write failing tests for ingest endpoint (success, idempotent, no papers, failures)
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement `airflow/dags/arxiv_ingestion/common.py` — constants
- [x] Implement `airflow/dags/arxiv_ingestion/setup.py` — health check task
- [x] Implement `airflow/dags/arxiv_ingestion/fetching.py` — fetch task
- [x] Implement `airflow/dags/arxiv_ingestion/reporting.py` — report task
- [x] Implement `airflow/dags/arxiv_paper_ingestion.py` — DAG definition
- [x] Implement `src/schemas/api/ingest.py` — request/response schemas
- [x] Implement `src/routers/ingest.py` — POST /api/v1/ingest/fetch endpoint
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Register ingest router in `src/main.py`
- [x] Add ingest router to `src/routers/__init__.py`
- [x] Run lint (`ruff check src/ airflow/ tests/`)
- [x] Run full test suite (`pytest`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S3.4_ingestion.ipynb`
- [x] Update roadmap.md status to "done"
