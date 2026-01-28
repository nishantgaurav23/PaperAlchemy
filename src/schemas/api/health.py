"""Health check Schemas."""

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