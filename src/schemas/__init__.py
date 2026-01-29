"""Pydantic schemas."""

from src.schemas.arxiv import (
    ArxivPaper,
    PaperCreate,
    PaperUpdate,
    PaperResponse,
    PDFContent,
    Section,
)

from src.schemas.api import (
    HealthResponse,
    ServiceStatus,
    SearchRequest,
    SearchResponse,
    SearchHit,
)

__all__ = [
    # arXiv schemas
    "ArxivPaper",
    "PaperCreate",
    "PaperUpdate",
    "PaperResponse",
    "PDFContent",
    "Section",
    # API schemas
    "HealthResponse",
    "ServiceStatus",
    "SearchRequest",
    "SearchResponse",
    "SearchHit",
]
