"""
Task 4: generate_daily_report

Collects XCom stats from fetch + indexing tasks, queries the API
health endpoint for live DB and OpenSearch totals, and logs a
complete JSON summary report.

No XCom output — this is a terminal observability task.
"""

import json
import logging
from datetime import datetime, timezone

import httpx

from arxiv_ingestion.common import HEALTH_URL, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


def generate_daily_report(ti, **context) -> dict:
    """
    Generate and log a JSON daily pipeline report.

    Airflow task function — called by PythonOperator.

    Args:
        ti: Airflow TaskInstance (injected by Airflow)
        context: Airflow context dict

    Returns:
        Report dict (logged as JSON).
    """
    # Pull stats pushed by Tasks 2 and 3
    fetch_result = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_result") or {}
    index_result = ti.xcom_pull(task_ids="index_papers_hybrid", key="index_result") or {}

    # Query API health for live DB + OpenSearch totals
    opensearch_doc_count = 0
    try:
        response = httpx.get(HEALTH_URL, timeout=DEFAULT_TIMEOUT)
        health = response.json()
        opensearch_msg = health.get("services", {}).get("opensearch", {}).get("message", "")
        # Message format: "Index 'arxiv-papers-chunks' with 42 documents"
        if "with" in opensearch_msg and "documents" in opensearch_msg:
            opensearch_doc_count = int(opensearch_msg.split("with")[1].split("documents")[0].strip())
    except Exception as e:
        logger.warning(f"Could not read OpenSearch doc count from health: {e}")

    execution_date = context.get("execution_date", datetime.now(timezone.utc))

    report = {
        "execution_date": execution_date.isoformat(),
        "target_date": fetch_result.get("target_date", "unknown"),
        "fetch": {
            "papers_fetched": fetch_result.get("papers_fetched", 0),
            "papers_stored": fetch_result.get("papers_stored", 0),
            "pdfs_downloaded": fetch_result.get("pdfs_downloaded", 0),
            "pdfs_parsed": fetch_result.get("pdfs_parsed", 0),
            "errors": len(fetch_result.get("errors", [])),
            "processing_time_s": round(fetch_result.get("processing_time", 0.0), 1),
        },
        "indexing": {
            "papers_processed": index_result.get("papers_processed", 0),
            "chunks_created": index_result.get("chunks_created", 0),
            "chunks_indexed": index_result.get("chunks_indexed", 0),
            "errors": len(index_result.get("errors", [])),
        },
        "totals": {
            "chunks_in_opensearch": opensearch_doc_count,
        },
        "status": "success" if not fetch_result.get("errors") else "partial",
    }

    logger.info(f"Daily report:\n{json.dumps(report, indent=2)}")
    return report
