"""Constants for the arXiv ingestion DAG.

All API calls go through the FastAPI REST API (Docker internal hostname)
to avoid SQLAlchemy version conflicts between Airflow and the application.
"""

from __future__ import annotations

API_BASE_URL = "http://api:8000"

INGEST_FETCH_URL = f"{API_BASE_URL}/api/v1/ingest/fetch"
HEALTH_URL = f"{API_BASE_URL}/api/v1/ping"

FETCH_TIMEOUT = 1800  # 30 min — PDF parsing is slow
DEFAULT_TIMEOUT = 30  # Health checks
