"""Task 3: Daily report — aggregate stats and log a structured report."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import httpx

from .common import DEFAULT_TIMEOUT, HEALTH_URL

logger = logging.getLogger(__name__)


def generate_daily_report(ti, execution_date: datetime | None = None, **context) -> None:
    """Aggregate fetch results and log a structured daily report.

    This task is observational — it does NOT raise on failure so the DAG
    can still complete even if reporting encounters issues.

    Args:
        ti: Airflow TaskInstance (for XCom pull).
        execution_date: Logical execution date from Airflow context.
        context: Additional Airflow context (unused).
    """
    if execution_date is None:
        execution_date = context.get("execution_date", datetime.utcnow())

    fetch_result = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_result") or {}

    # Try to get system totals from health endpoint
    health_info = {}
    try:
        response = httpx.get(HEALTH_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        health_info = response.json()
    except Exception as e:
        logger.warning("Could not reach health endpoint for report: %s", e)

    report = {
        "execution_date": str(execution_date),
        "fetch": {
            "papers_fetched": fetch_result.get("papers_fetched", 0),
            "pdfs_downloaded": fetch_result.get("pdfs_downloaded", 0),
            "pdfs_parsed": fetch_result.get("pdfs_parsed", 0),
            "papers_stored": fetch_result.get("papers_stored", 0),
            "errors": fetch_result.get("errors", []),
            "processing_time": fetch_result.get("processing_time", 0),
        },
        "health": health_info,
        "status": "success" if not fetch_result.get("errors") else "partial",
    }

    logger.info("=== Daily Ingestion Report ===")
    logger.info(json.dumps(report, indent=2, default=str))
