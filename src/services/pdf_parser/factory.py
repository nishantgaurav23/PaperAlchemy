"""PDF parser service factory."""

from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.services.pdf_parser.service import PDFParserService


@lru_cache(maxsize=1)
def make_pdf_parser_service() -> PDFParserService:
    """Create and cache a singleton PDFParserService from settings."""
    settings = get_settings()
    return PDFParserService(
        max_pages=settings.pdf_parser.max_pages,
        max_file_size_mb=settings.pdf_parser.max_file_size_mb,
        timeout=settings.pdf_parser.timeout,
        enable_docling_fallback=settings.pdf_parser.enable_docling_fallback,
    )


def reset_pdf_parser_cache() -> None:
    """Clear the cached service instance."""
    make_pdf_parser_service.cache_clear()
