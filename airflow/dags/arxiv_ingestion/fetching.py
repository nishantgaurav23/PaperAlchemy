"""
Task 2: fetch_daily_papers

Calls POST /api/v1/ingest/fetch on the API container.
The API handles arXiv fetching, PDF parsing, and PostgreSQL storage.
This task is just the orchestrator — it triggers and monitors the work.

XCom output (key="fetch_result"):
    {
        "target_date": "20260221",
        "papers_fetched": 42,
        "pdfs_downloaded": 38,
        "pdfs_parsed": 35,
        "papers_stored": 42,
        "arxiv_ids": ["2602.06039", ...],
        "errors": [],
        "processing_time": 183.4
    }
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from arxiv_ingestion.common import INGEST_FETCH_URL, FETCH_TIMEOUT

logger = logging.getLogger(__name__)


def fetch_daily_papers(ti, **context) -> dict:
    """
    Trigger paper fetch via POST /api/v1/ingest/fetch and push result to XCom.

    Airflow task function — called by PythonOperator.

    Args:
        ti: Airflow TaskInstance (injected by Airflow, used for XCom push)
        context: Airflow context dict (execution_date, etc.)

    Returns:
        Fetch statistics dict (also pushed to XCom).
    """
    # Determine target date: day before the DAG execution date
    execution_date = context.get("execution_date", datetime.now(timezone.utc))
    yesterday = execution_date - timedelta(days=1)
    target_date = yesterday.strftime("%Y%m%d")

    logger.info(f"Calling POST {INGEST_FETCH_URL} for date={target_date}")

    try:
        response = httpx.post(
            INGEST_FETCH_URL,
            json={"date": target_date, "max_results": 10, "process_pdfs": True},
            timeout=FETCH_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Fetch endpoint error {e.response.status_code}: {e.response.text}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Fetch request failed: {e}") from e

    logger.info(
        f"Fetch complete — fetched={result.get('papers_fetched', 0)}, "
        f"stored={result.get('papers_stored', 0)}, "
        f"errors={len(result.get('errors', []))}"
    )

    ti.xcom_push(key="fetch_result", value=result)
    return result
