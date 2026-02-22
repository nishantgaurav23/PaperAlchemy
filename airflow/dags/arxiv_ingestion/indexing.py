"""
Task 3: index_papers_hybrid

Calls POST /api/v1/ingest/index on the API container.
Passes the arxiv_ids from Task 2's XCom result so only today's papers
are indexed. Falls back to indexing all papers from the last 24 hours
if no XCom result is available.

XCom output (key="index_result"):
    {
        "papers_processed": 42,
        "chunks_created": 310,
        "chunks_indexed": 308,
        "errors": []
    }
"""

import logging

import httpx

from arxiv_ingestion.common import INGEST_INDEX_URL, INDEX_TIMEOUT

logger = logging.getLogger(__name__)


def index_papers_hybrid(ti, **context) -> dict:
    """
    Trigger hybrid indexing via POST /api/v1/ingest/index and push result to XCom.

    Airflow task function — called by PythonOperator.

    Args:
        ti: Airflow TaskInstance (injected by Airflow)
        context: Airflow context dict

    Returns:
        Indexing statistics dict (also pushed to XCom).
    """
    # Pull arxiv_ids stored by Task 2 to index only today's papers
    fetch_result = ti.xcom_pull(task_ids="fetch_daily_papers", key="fetch_result") or {}
    arxiv_ids = fetch_result.get("arxiv_ids", [])

    payload = {"since_hours": 24}
    if arxiv_ids:
        payload["arxiv_ids"] = arxiv_ids
        logger.info(f"Indexing {len(arxiv_ids)} specific papers from XCom")
    else:
        logger.info("No XCom arxiv_ids — indexing all papers from last 24 hours")

    logger.info(f"Calling POST {INGEST_INDEX_URL}")

    try:
        response = httpx.post(
            INGEST_INDEX_URL,
            json=payload,
            timeout=INDEX_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Index endpoint error {e.response.status_code}: {e.response.text}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Index request failed: {e}") from e

    logger.info(
        f"Indexing complete — papers={result.get('papers_processed', 0)}, "
        f"chunks={result.get('chunks_indexed', 0)}"
    )

    ti.xcom_push(key="index_result", value=result)
    return result
