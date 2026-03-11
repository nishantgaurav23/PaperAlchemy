# Spec S3.4 -- Ingestion Pipeline (Airflow DAG)

## Overview
Airflow DAG that orchestrates the daily paper ingestion pipeline: fetch arXiv papers, download PDFs, parse them, store metadata in PostgreSQL, and prepare for indexing. Runs Mon-Fri at 6am UTC. Tasks communicate via XCom. All write operations delegate to the FastAPI REST API (HTTP calls) to avoid SQLAlchemy version conflicts between Airflow and the app. Idempotent, with 2 retries and 30-minute retry delay.

## Dependencies
- **S3.1** (Paper model & repository) — Paper ORM model, PaperRepository CRUD
- **S3.2** (ArXiv client) — ArxivClient for fetching papers and downloading PDFs
- **S3.3** (PDF parser) — PDFParserService for section-aware PDF parsing

## Target Location
- `airflow/dags/arxiv_paper_ingestion.py` — Main DAG definition
- `airflow/dags/arxiv_ingestion/__init__.py` — Package marker
- `airflow/dags/arxiv_ingestion/common.py` — Constants (API URLs, timeouts)
- `airflow/dags/arxiv_ingestion/setup.py` — Health check task
- `airflow/dags/arxiv_ingestion/fetching.py` — Fetch + parse task
- `airflow/dags/arxiv_ingestion/reporting.py` — Daily report task
- `src/routers/ingest.py` — FastAPI ingestion endpoints (POST /api/v1/ingest/fetch)

## Functional Requirements

### FR-1: DAG Configuration
- **What**: Airflow DAG with proper scheduling, retries, and concurrency control
- **Inputs**: Cron schedule, default_args
- **Outputs**: Compiled DAG object registered with Airflow
- **Config**:
  - Schedule: `0 6 * * 1-5` (Mon-Fri 6am UTC)
  - Retries: 2, retry delay: 30 minutes
  - `catchup=False` (no backfill)
  - `max_active_runs=1` (prevent concurrent runs)
  - `depends_on_past=False`
  - Tags: `["paperalchemy", "ingestion", "production"]`
- **Edge cases**: DAG parse errors, Airflow scheduler not running

### FR-2: Setup Task (Health Check)
- **What**: Validate all required services are healthy before proceeding
- **Inputs**: API health endpoint URL
- **Outputs**: Pass (services healthy) or RuntimeError (service down)
- **Checks**: GET `/api/v1/health` — verify API, PostgreSQL are healthy
- **Edge cases**: API unreachable, partial service failure

### FR-3: Fetch Daily Papers Task
- **What**: Call API to fetch yesterday's arXiv papers, download PDFs, parse, and store
- **Inputs**: execution_date from Airflow context
- **Outputs**: XCom push with fetch results (papers_fetched, pdfs_downloaded, pdfs_parsed, arxiv_ids, errors)
- **API call**: POST `/api/v1/ingest/fetch` with `{"target_date": "YYYYMMDD"}`
- **Date logic**: `target_date = execution_date - 1 day`
- **Timeout**: 1800s (30 min, PDF parsing is slow)
- **Edge cases**: No papers for date, API timeout, partial PDF failures

### FR-4: Daily Report Task
- **What**: Aggregate stats from fetch task and log a structured report
- **Inputs**: XCom pull from fetch task, health endpoint
- **Outputs**: Logged report with paper counts, success/failure stats
- **Edge cases**: Missing XCom data (defensive defaults)

### FR-5: Ingestion API Endpoint
- **What**: FastAPI endpoint that orchestrates fetch → parse → store for a given date
- **Endpoint**: POST `/api/v1/ingest/fetch`
- **Inputs**: `{"target_date": "YYYYMMDD"}` (optional, defaults to yesterday)
- **Process**:
  1. Fetch papers from arXiv API for target_date
  2. For each paper: upsert metadata to DB
  3. Download PDFs to local cache
  4. Parse PDFs with Docling
  5. Update paper records with parsed content
- **Outputs**: JSON with papers_fetched, pdfs_downloaded, pdfs_parsed, papers_stored, arxiv_ids, errors, processing_time
- **Idempotency**: Upserts prevent duplicate papers; re-parsing overwrites
- **Edge cases**: arXiv API rate limits, PDF download failures, parse timeouts, DB connection issues

### FR-6: Cleanup Task
- **What**: Remove cached PDFs older than 30 days
- **Operator**: BashOperator
- **Command**: `find /tmp/paperalchemy_pdfs -mtime +30 -delete || true`
- **Edge cases**: Directory doesn't exist, no old files

## Tangible Outcomes
- [ ] Airflow DAG file loads without parse errors
- [ ] DAG has 4 tasks in linear chain: setup → fetch → report → cleanup
- [ ] Health check task validates API is reachable
- [ ] Fetch task calls ingestion API endpoint and pushes results to XCom
- [ ] Report task aggregates and logs structured daily report
- [ ] Cleanup task removes old PDFs safely
- [ ] POST `/api/v1/ingest/fetch` endpoint fetches, parses, stores papers
- [ ] Ingestion endpoint is idempotent (re-run produces same DB state)
- [ ] All tasks handle errors gracefully with appropriate logging

## Test-Driven Requirements

### Tests to Write First
1. `test_dag_loads`: DAG file loads without import errors, has correct config
2. `test_dag_task_count`: DAG has exactly 4 tasks
3. `test_dag_task_dependencies`: Tasks are in correct linear order
4. `test_dag_schedule`: Schedule is Mon-Fri 6am UTC
5. `test_dag_default_args`: Retries=2, retry_delay=30min
6. `test_setup_task_healthy`: Health check passes when API returns healthy
7. `test_setup_task_unhealthy`: Health check raises RuntimeError on unhealthy service
8. `test_fetch_task_success`: Fetch task calls API and pushes XCom
9. `test_fetch_task_api_error`: Fetch task raises RuntimeError on API failure
10. `test_fetch_task_date_logic`: Target date is execution_date - 1 day
11. `test_report_task`: Report task logs structured output
12. `test_ingest_endpoint_success`: API endpoint fetches and stores papers
13. `test_ingest_endpoint_idempotent`: Re-running produces same result
14. `test_ingest_endpoint_no_papers`: Handles empty arXiv results
15. `test_ingest_endpoint_pdf_failure`: Handles PDF download/parse failures gracefully

### Mocking Strategy
- Mock `httpx` for DAG task HTTP calls (to API)
- Mock `ArxivClient` for ingestion endpoint tests
- Mock `PDFParserService` for ingestion endpoint tests
- Mock `PaperRepository` for DB operations
- Mock `AsyncSession` for database tests
- Use `unittest.mock.patch` for Airflow task function tests

### Coverage
- All DAG configuration validated
- All task functions tested in isolation
- API endpoint tested with mocked services
- Error paths tested (API failures, parse failures, empty results)
