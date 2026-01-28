"""PDF parser service factory"""

from functools import lru_cache

from src.config import get_settings
from src.services.pdf_parser.service import PDFParserService

@lru_cache(maxsize=1)
def make_pdf_parser_service() -> PDFParserService:
    """
    Create and cache PDF parser service instance.

    Returns:
        PDFParserService instance (singleton)
    """
    settings = get_settings()

    return PDFParserService(
        max_pages=settings.pdf_parser.max_pages,
        max_file_size_mb=settings.pdf_parser.max_file_size_mb,
        timeout=settings.pdf_parser.timeout,
    )

def reset_pdf_parser_cache() -> None:
    """Reset the cached service instane."""
    make_pdf_parser_service.cache_clear()