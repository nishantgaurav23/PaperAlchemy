"""Health check router with services status."""

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
