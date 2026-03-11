"""Task 2: Fetch daily papers — call ingestion API endpoint."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from .common import FETCH_TIMEOUT, INGEST_FETCH_URL

logger = logging.getLogger(__name__)


def fetch_daily_papers(ti, execution_date: datetime | None = None, **context) -> None:
    """Fetch yesterday's arXiv papers via the ingestion API.

    Args:
        ti: Airflow TaskInstance (for XCom push).
        execution_date: Logical execution date from Airflow context.
        context: Additional Airflow context (unused).

    Raises:
        RuntimeError: On HTTP or connection failure (triggers Airflow retry).
    """
    if execution_date is None:
        execution_date = context.get("execution_date", datetime.utcnow())

    target_date = (execution_date - timedelta(days=1)).strftime("%Y%m%d")
    logger.info("Fetching papers for target_date=%s", target_date)

    payload = {"target_date": target_date}

    try:
        response = httpx.post(INGEST_FETCH_URL, json=payload, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Fetch endpoint error {e.response.status_code}: {e.response.text}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Fetch request failed: {e}") from e

    papers_fetched = result.get("papers_fetched", 0)
    arxiv_ids = result.get("arxiv_ids", [])
    logger.info("Fetched %d papers, %d arXiv IDs", papers_fetched, len(arxiv_ids))

    ti.xcom_push(key="fetch_result", value=result)
