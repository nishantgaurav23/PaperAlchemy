"""PDF parser service module."""

from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.pdf_parser.service import PDFParserService

__all__ = [
    "PDFParserService",
    "make_pdf_parser_service",
]
