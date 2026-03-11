"""PDF parsing service using Docling."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.exceptions import PDFParsingError, PDFValidationError
from src.schemas.pdf import PDFContent, Section

logger = logging.getLogger(__name__)

# Lazy import guard — resolved at runtime inside _get_converter
DocumentConverter = None


class PDFParserService:
    """Section-aware PDF parser using Docling with async support and timeout protection."""

    def __init__(
        self,
        max_pages: int = 30,
        max_file_size_mb: int = 50,
        timeout: int = 120,
    ) -> None:
        self.max_pages = max_pages
        self.max_file_size_mb = max_file_size_mb
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._converter = None

    # ------------------------------------------------------------------
    # Lazy Docling init
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is None:
            try:
                global DocumentConverter
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
                logger.info("Docling converter initialized")
            except ImportError as err:
                raise PDFParsingError("Docling not installed. Run: uv add docling") from err
        return self._converter

    # ------------------------------------------------------------------
    # FR-1: File validation
    # ------------------------------------------------------------------

    def _validate_file(self, pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise PDFValidationError(f"File not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise PDFValidationError(f"Not a PDF file: {pdf_path}")

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise PDFValidationError(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")

        with open(pdf_path, "rb") as f:
            magic = f.read(5)
            if magic != b"%PDF-":
                raise PDFValidationError(f"Invalid PDF file: {pdf_path}")

    # ------------------------------------------------------------------
    # FR-2: Section-aware parsing (sync, runs in thread pool)
    # ------------------------------------------------------------------

    def _parse_sync(self, pdf_path: Path) -> PDFContent:
        start_time = time.time()

        converter = self._get_converter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Raw text
        raw_text = doc.export_to_text() if hasattr(doc, "export_to_text") else ""

        # Sections
        sections: list[Section] = []
        if hasattr(doc, "texts"):
            current_section: Section | None = None
            for item in doc.texts:
                label = str(getattr(item, "label", "")).lower()
                text = getattr(item, "text", "").strip()
                if "heading" in label or "header" in label:
                    if current_section:
                        sections.append(current_section)
                    current_section = Section(title=text, content="", level=1)
                elif current_section:
                    current_section.content += text + "\n"
                else:
                    current_section = Section(title="Introduction", content=text + "\n", level=1)
            if current_section:
                sections.append(current_section)

        # Tables
        tables: list[str] = []
        if hasattr(doc, "tables"):
            for table in doc.tables:
                if hasattr(table, "export_to_text"):
                    tables.append(table.export_to_text())
                elif hasattr(table, "text"):
                    tables.append(table.text)

        # Figure captions
        figures: list[str] = []
        if hasattr(doc, "pictures"):
            for pic in doc.pictures:
                if hasattr(pic, "caption") and pic.caption:
                    figures.append(pic.caption)

        parse_time = time.time() - start_time

        return PDFContent(
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            figures=figures,
            parser_used="docling",
            parser_time_seconds=parse_time,
        )

    # ------------------------------------------------------------------
    # FR-3: Async parsing with timeout
    # ------------------------------------------------------------------

    async def parse_pdf(self, pdf_path: Path) -> PDFContent:
        logger.info("Parsing PDF: %s", pdf_path.name)

        self._validate_file(pdf_path)

        loop = asyncio.get_running_loop()
        try:
            content = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._parse_sync, pdf_path),
                timeout=self.timeout,
            )
        except TimeoutError:
            raise PDFParsingError(f"PDF parsing timed out after {self.timeout}s: {pdf_path.name}") from None

        logger.info(
            "Parsed %s: %d sections, %d chars, %.1fs",
            pdf_path.name,
            len(content.sections),
            len(content.raw_text),
            content.parser_time_seconds,
        )
        return content

    # ------------------------------------------------------------------
    # FR-4: Batch parsing
    # ------------------------------------------------------------------

    async def parse_multiple(
        self,
        pdf_paths: list[Path],
        continue_on_error: bool = True,
    ) -> dict[str, PDFContent | None]:
        results: dict[str, PDFContent | None] = {}

        for pdf_path in pdf_paths:
            try:
                content = await self.parse_pdf(pdf_path)
                results[pdf_path.name] = content
            except Exception as e:
                logger.error("Failed to parse %s: %s", pdf_path.name, e)
                if continue_on_error:
                    results[pdf_path.name] = None
                else:
                    raise

        success_count = sum(1 for v in results.values() if v is not None)
        logger.info("Parsed %d/%d PDFs successfully", success_count, len(pdf_paths))
        return results

    # ------------------------------------------------------------------
    # FR-6: Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._converter = None
