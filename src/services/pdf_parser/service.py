"""PDF parsing service — PyMuPDF (fast) with Docling fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.exceptions import PDFParsingError, PDFValidationError
from src.schemas.pdf import PDFContent, Section

logger = logging.getLogger(__name__)

# Lazy import guards — resolved at runtime
DocumentConverter = None


def _sanitize_text(text: str) -> str:
    """Remove null bytes and other invalid characters that break PostgreSQL UTF-8 storage."""
    return text.replace("\x00", "")


class PDFParserService:
    """Two-tier PDF parser: PyMuPDF (primary, fast) → Docling (fallback, heavy).

    PyMuPDF handles ~95% of text-based PDFs in <2s.
    If PyMuPDF fails for any reason, Docling is used as a universal fallback.
    """

    def __init__(
        self,
        max_pages: int = 30,
        max_file_size_mb: int = 50,
        timeout: int = 120,
        enable_docling_fallback: bool = True,
    ) -> None:
        self.max_pages = max_pages
        self.max_file_size_mb = max_file_size_mb
        self.timeout = timeout
        self.enable_docling_fallback = enable_docling_fallback
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._docling_converter = None

    # ------------------------------------------------------------------
    # Lazy Docling init (only when fallback is needed)
    # ------------------------------------------------------------------

    def _get_docling_converter(self):
        if self._docling_converter is None:
            try:
                global DocumentConverter
                from docling.document_converter import DocumentConverter

                self._docling_converter = DocumentConverter()
                logger.info("Docling converter initialized (fallback)")
            except ImportError as err:
                raise PDFParsingError("Docling not installed. Run: uv add docling") from err
        return self._docling_converter

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
    # FR-2a: PyMuPDF parsing (primary — fast)
    # ------------------------------------------------------------------

    def _parse_with_pymupdf(self, pdf_path: Path) -> PDFContent:
        """Parse PDF using PyMuPDF (fitz). Fast and reliable for text-based PDFs."""
        import fitz

        start_time = time.time()
        doc = fitz.open(str(pdf_path))

        try:
            page_count = len(doc)

            # Extract raw text from all pages
            raw_text_parts: list[str] = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    raw_text_parts.append(text)
            raw_text = _sanitize_text("\n".join(raw_text_parts))

            # Extract sections by detecting headings via font size analysis
            sections = self._extract_sections_pymupdf(doc)

            # Extract tables (PyMuPDF built-in table finder, available since 1.23.0)
            tables: list[str] = []
            for page in doc:
                try:
                    page_tables = page.find_tables()
                    for table in page_tables:
                        df = table.to_pandas()
                        tables.append(df.to_string(index=False))
                except Exception:
                    pass  # Table extraction is best-effort

            # Extract figure captions (heuristic: lines starting with "Figure" or "Fig.")
            figures: list[str] = []
            for line in raw_text.split("\n"):
                stripped = line.strip()
                if stripped.lower().startswith(("figure ", "fig. ", "fig ")) and len(stripped) < 300:
                    figures.append(stripped)

            parse_time = time.time() - start_time

            return PDFContent(
                raw_text=raw_text,
                sections=sections,
                tables=tables,
                figures=figures,
                page_count=page_count,
                parser_used="pymupdf",
                parser_time_seconds=parse_time,
            )
        finally:
            doc.close()

    def _extract_sections_pymupdf(self, doc) -> list[Section]:
        """Extract sections by analyzing font sizes to detect headings."""
        import fitz

        # First pass: collect all text blocks with font info
        blocks: list[dict] = []
        for page in doc:
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    max_font_size = 0.0
                    is_bold = False
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 0)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or "black" in font_name:
                            is_bold = True
                    line_text = line_text.strip()
                    if line_text:
                        blocks.append({
                            "text": line_text,
                            "font_size": max_font_size,
                            "is_bold": is_bold,
                        })

        if not blocks:
            return []

        # Compute body font size (most common size)
        size_counts: dict[float, int] = {}
        for b in blocks:
            rounded = round(b["font_size"], 1)
            size_counts[rounded] = size_counts.get(rounded, 0) + 1
        body_size = max(size_counts, key=size_counts.get) if size_counts else 10.0

        # Second pass: classify headings vs body text
        sections: list[Section] = []
        current_section: Section | None = None

        for b in blocks:
            text = b["text"]
            font_size = b["font_size"]
            is_bold = b["is_bold"]

            # Heading heuristic: larger font or bold + short text + not all lowercase
            is_heading = (
                (font_size > body_size + 1.0 or (is_bold and font_size >= body_size))
                and len(text) < 150
                and not text[0].islower()
            )

            if is_heading:
                if current_section:
                    current_section.content = _sanitize_text(current_section.content)
                    sections.append(current_section)
                # Determine heading level from font size difference
                size_diff = font_size - body_size
                if size_diff > 4:
                    level = 1
                elif size_diff > 2:
                    level = 2
                else:
                    level = 3
                level = max(1, min(level, 6))
                current_section = Section(title=_sanitize_text(text), content="", level=level)
            elif current_section:
                current_section.content += text + "\n"
            else:
                current_section = Section(title="Introduction", content=text + "\n", level=1)

        if current_section:
            current_section.content = _sanitize_text(current_section.content)
            sections.append(current_section)

        # Post-process: merge orphaned section-number headings (e.g. "1", "2", "A", "B")
        # with the next named section. Academic PDFs often have section numbers on a
        # separate line in a larger font, which the heuristic above picks up as headings.
        merged: list[Section] = []
        i = 0
        while i < len(sections):
            title = sections[i].title.strip()
            # Detect standalone section numbers/letters: "1", "2", "10", "A", "B", "A.1", "3.2", etc.
            is_orphan = (
                len(title) <= 4
                and not sections[i].content.strip()
                and (title.replace(".", "").isdigit() or (len(title) == 1 and title.isalpha()))
            )
            if is_orphan and i + 1 < len(sections):
                # Merge: prepend the number/letter to the next section's title
                next_section = sections[i + 1]
                next_section.title = f"{title}. {next_section.title}" if next_section.title else title
                next_section.level = min(sections[i].level, next_section.level)
                i += 1  # skip the orphan, the next iteration picks up the merged section
            else:
                merged.append(sections[i])
                i += 1

        return merged

    # ------------------------------------------------------------------
    # FR-2b: Docling parsing (fallback — heavy but robust)
    # ------------------------------------------------------------------

    def _parse_with_docling(self, pdf_path: Path) -> PDFContent:
        """Parse PDF using Docling. Slower but handles complex layouts and scanned PDFs."""
        start_time = time.time()

        converter = self._get_docling_converter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Raw text
        raw_text = _sanitize_text(doc.export_to_text()) if hasattr(doc, "export_to_text") else ""

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
    # FR-2: Combined parsing — PyMuPDF first, Docling fallback
    # ------------------------------------------------------------------

    def _parse_sync(self, pdf_path: Path) -> PDFContent:
        """Try PyMuPDF first; fall back to Docling on any failure."""
        try:
            content = self._parse_with_pymupdf(pdf_path)
            logger.info(
                "PyMuPDF parsed %s: %d sections, %d chars in %.2fs",
                pdf_path.name,
                len(content.sections),
                len(content.raw_text),
                content.parser_time_seconds,
            )
            return content
        except Exception as pymupdf_err:
            logger.warning("PyMuPDF failed for %s: %s — falling back to Docling", pdf_path.name, pymupdf_err)

            if not self.enable_docling_fallback:
                raise PDFParsingError(f"PyMuPDF failed and Docling fallback is disabled: {pymupdf_err}") from pymupdf_err

            try:
                content = self._parse_with_docling(pdf_path)
                logger.info(
                    "Docling fallback parsed %s: %d sections, %d chars in %.2fs",
                    pdf_path.name,
                    len(content.sections),
                    len(content.raw_text),
                    content.parser_time_seconds,
                )
                return content
            except PDFParsingError:
                raise
            except Exception as docling_err:
                raise PDFParsingError(
                    f"Both parsers failed for {pdf_path.name}. "
                    f"PyMuPDF: {pymupdf_err}. Docling: {docling_err}"
                ) from docling_err

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
            "Parsed %s [%s]: %d sections, %d chars, %.1fs",
            pdf_path.name,
            content.parser_used,
            len(content.sections),
            len(content.raw_text),
            content.parser_time_seconds,
        )
        return content

    # ------------------------------------------------------------------
    # FR-4: Batch parsing (parallel with asyncio.gather)
    # ------------------------------------------------------------------

    async def parse_multiple(
        self,
        pdf_paths: list[Path],
        continue_on_error: bool = True,
    ) -> dict[str, PDFContent | None]:
        if not pdf_paths:
            return {}

        async def _parse_one(pdf_path: Path) -> tuple[str, PDFContent | None]:
            try:
                content = await self.parse_pdf(pdf_path)
                return pdf_path.name, content
            except Exception as e:
                logger.error("Failed to parse %s: %s", pdf_path.name, e)
                if continue_on_error:
                    return pdf_path.name, None
                raise

        tasks = [_parse_one(p) for p in pdf_paths]
        completed = await asyncio.gather(*tasks, return_exceptions=not continue_on_error)

        results: dict[str, PDFContent | None] = {}
        for item in completed:
            if isinstance(item, Exception):
                raise item
            name, content = item
            results[name] = content

        success_count = sum(1 for v in results.values() if v is not None)
        logger.info("Parsed %d/%d PDFs successfully", success_count, len(pdf_paths))
        return results

    # ------------------------------------------------------------------
    # FR-6: Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._docling_converter = None
