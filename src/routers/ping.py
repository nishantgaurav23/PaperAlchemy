"""
Health check router with per-service status reporting.

Why it's needed:
    Production systems need a detailed health endpoint that checks every
    dependency (database, search engine). Load balancers use the simple
    /health endpoint, while monitoring dashboards use /api/v1/health to
    see which specific service is down.

What it does:
    - GET /api/v1/health: Probes PostgreSQL (SELECT 1) and OpenSearch
      (cluster health + index stats). Returns structured JSON with
      per-service status ("healthy"/"unhealthy") and overall status
      ("ok"/"degraded").

How it helps:
    - Monitoring: Grafana/Datadog can parse the JSON to create alerts
    - Debugging: immediately see if OpenSearch is down vs database is down
    - Load balancers: route traffic away from degraded instances
    - On-call: first thing to check when users report issues
"""

import logging

from fastapi import APIRouter
from sqlalchemy import text

from src.dependency import DatabaseDep, OpenSearchDep, SettingsDep
from src.schemas.api.health import HealthResponse, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    settings: SettingsDep,
    database: DatabaseDep,
    opensearch_client: OpenSearchDep
) -> HealthResponse:
    """
    Comprehensive health check endpoint for monitoring and load balancer probes.

    Returns servie health status with version and connectivity checks.
    """
    services = {}
    overall_status = "ok"

    # Database check
    try:
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        services["database"] = ServiceStatus(
            status="healthy",
            message="Connected successfully"
        )
    except Exception as e:
        services["database"] = ServiceStatus(
            status="unhealthy",
            message=str(e)
        )
        overall_status = "degraded"

    # OpenSearch check
    try:
        if opensearch_client.health_check():
            stats = opensearch_client.get_index_stats()
            services["opensearch"] = ServiceStatus(
                status="healthy",
                message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents"
            )
        else:
            services["opensearch"] = ServiceStatus(
                status="unhealthy",
                message="Not responding"
            )
            overall_status = "degraded"

    except Exception as e:
        services["opensearch"] = ServiceStatus(
            status="unhealthy",
            message=str(e)
        )
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        environment="development",
        service_name="paperalchemy-api",
        services=services,
    )
