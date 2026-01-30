"""
Health check response schemas for the /api/v1/health endpoint.

Why it's needed:
    Structured health responses let monitoring tools (Grafana, Datadog)
    parse the JSON programmatically. Without typed schemas, the response
    shape could drift between code changes, breaking dashboards.

What it does:
    - ServiceStatus: Status of one service (e.g., database: healthy)
    - HealthResponse: Overall API status + map of service statuses.
      Status is "ok" when all services are healthy, "degraded" when
      any service is unhealthy.

How it helps:
    - FastAPI auto-generates OpenAPI docs from these models
    - Monitoring tools parse status field to trigger alerts
    - Frontend health dashboards render per-service status
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

class ServiceStatus(BaseModel):
    """Individual service health status."""

    status: str = Field(..., description="Servie status: healthy, unhealthy, degraded")
    message: Optional[str] = Field(None, description="Additional status information")

class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(..., description="Overall API status: ok, degraded, unhealthy")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment: development, staging, production")
    service_name: str = Field(..., description="Service name")
    services: Dict[str, ServiceStatus] = Field(
        default_factory=dict,
        description="Individual service health statuses"
    )