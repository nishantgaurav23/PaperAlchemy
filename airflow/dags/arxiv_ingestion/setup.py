"""
Task 1: setup_environment

Calls GET /api/v1/health to verify the API (and all its services:
PostgreSQL, OpenSearch) are healthy before committing to a full run.
Fails fast so the other 4 tasks don't run against a broken environment.
"""

import logging
import httpx

from arxiv_ingestion.common import HEALTH_URL, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


def setup_environment(**context) -> dict:
    """
    Verify all required services are healthy via the API health endpoint.

    Airflow task function — called by PythonOperator.

    Returns:
        dict with health check results for each service.

    Raises:
        RuntimeError: If the API or any downstream service is unreachable.
    """
    logger.info(f"Checking API health at {HEALTH_URL}")

    try:
        response = httpx.get(HEALTH_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        health = response.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f"API health check failed: {e}") from e

    # Verify each service reported as healthy
    services = health.get("services", {})
    for service_name, service_status in services.items():
        if service_status.get("status") != "healthy":
            raise RuntimeError(
                f"{service_name} is not healthy: {service_status.get('message')}"
            )

    logger.info(f"All services healthy: {list(services.keys())}")
    return health
