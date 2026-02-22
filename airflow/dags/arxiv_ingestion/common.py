"""
Shared constants for the PaperAlchemy Airflow DAG tasks.

Why this is now just a constants file:
    The original version imported from src.* (SQLAlchemy 2.0) which conflicts
    with Airflow's bundled SQLAlchemy 1.4. The fix: DAG tasks call HTTP endpoints
    on the API container instead. The API container has all correct dependencies.

The API container is reachable at http://api:8000 from inside the Docker network.
"""

# Base URL of the PaperAlchemy API (Docker internal network hostname)
API_BASE_URL = "http://api:8000"

INGEST_FETCH_URL = f"{API_BASE_URL}/api/v1/ingest/fetch"
INGEST_INDEX_URL = f"{API_BASE_URL}/api/v1/ingest/index"
HEALTH_URL = f"{API_BASE_URL}/api/v1/health"

# Timeout for long-running ingest calls (PDF parsing can be slow)
FETCH_TIMEOUT = 1800  # 30 minutes — docling parses many PDFs
INDEX_TIMEOUT = 900   # 15 minutes — Jina embedding batches
DEFAULT_TIMEOUT = 30
