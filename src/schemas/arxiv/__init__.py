"""arXiv Schemas"""

from src.schemas.arxiv.paper import (
    ArxivPaper,
    PaperCreate,
    PaperUpdate,
    PaperResponse,
    PDFContent,
    Section,
)

__all__ = [
    "ArxivPaper",
    "PaperCreate",
    "PaperUpdate",
    "PaperResponse",
    "PDFContent",
    "Section",
]
