"""Task 1: Health check — validate all required services before proceeding."""

from __future__ import annotations

import logging

import httpx

from .common import DEFAULT_TIMEOUT, HEALTH_URL

logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """Validate API and downstream services are healthy.

    Raises RuntimeError if health check fails, triggering Airflow retry.
    """
    logger.info("Running health check: %s", HEALTH_URL)

    try:
        response = httpx.get(HEALTH_URL, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        health = response.json()
        logger.info("Health check passed: %s", health)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Health check failed: HTTP {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Health check failed: {e}") from e
