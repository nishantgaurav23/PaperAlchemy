"""Tests for PDF parser service (S3.3) — PyMuPDF primary + Docling fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from src.exceptions import PDFParsingError, PDFValidationError
from src.schemas.pdf import PDFContent, Section

# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestSectionModel:
    def test_section_defaults(self):
        s = Section(title="Abstract")
        assert s.title == "Abstract"
        assert s.content == ""
        assert s.level == 1

    def test_section_full(self):
        s = Section(title="Methods", content="We used X.", level=2)
        assert s.title == "Methods"
        assert s.content == "We used X."
        assert s.level == 2

    def test_section_level_bounds(self):
        with pytest.raises(ValidationError):
            Section(title="Bad", level=0)
        with pytest.raises(ValidationError):
            Section(title="Bad", level=7)


class TestPDFContentModel:
    def test_pdf_content_defaults(self):
        c = PDFContent()
        assert c.raw_text == ""
        assert c.sections == []
        assert c.tables == []
        assert c.figures == []
        assert c.page_count == 0
        assert c.parser_used == "docling"
        assert c.parser_time_seconds == 0.0

    def test_pdf_content_full(self):
        c = PDFContent(
            raw_text="hello",
            sections=[Section(title="Intro", content="text")],
            tables=["col1|col2"],
            figures=["Figure 1: something"],
            page_count=10,
            parser_used="pymupdf",
            parser_time_seconds=1.5,
        )
        assert c.raw_text == "hello"
        assert len(c.sections) == 1
        assert c.sections[0].title == "Intro"
        assert len(c.tables) == 1
        assert len(c.figures) == 1
        assert c.page_count == 10
        assert c.parser_used == "pymupdf"
        assert c.parser_time_seconds == 1.5

    def test_negative_page_count_rejected(self):
        with pytest.raises(ValidationError):
            PDFContent(page_count=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, size_bytes: int | None = None) -> Path:
    """Write a file with valid PDF magic bytes."""
    content = b"%PDF-1.4 minimal"
    if size_bytes is not None:
        content = b"%PDF-1.4" + b"\x00" * max(0, size_bytes - 8)
    path.write_bytes(content)
    return path


def _make_non_pdf(path: Path) -> Path:
    path.write_bytes(b"NOT A PDF FILE")
    return path


def _fake_docling_doc(texts=None, tables=None, pictures=None, raw_text="raw content"):
    """Build a fake Docling document object."""
    doc = MagicMock()
    doc.export_to_text.return_value = raw_text
    doc.texts = texts if texts is not None else []
    doc.tables = tables if tables is not None else []
    doc.pictures = pictures if pictures is not None else []
    return doc


def _text_item(text, label="paragraph"):
    item = MagicMock()
    item.text = text
    item.label = label
    return item


def _heading_item(text, level=1):
    item = MagicMock()
    item.text = text
    item.label = f"section_header_level_{level}" if level > 0 else "heading"
    return item


def _table_item(text):
    item = MagicMock()
    item.export_to_text.return_value = text
    return item


def _picture_item(caption=None):
    item = MagicMock()
    item.caption = caption
    return item


def _make_pymupdf_result() -> PDFContent:
    """Return a typical PyMuPDF parse result."""
    return PDFContent(
        raw_text="Full paper text here",
        sections=[Section(title="Introduction", content="Some text\n", level=1)],
        tables=[],
        figures=[],
        page_count=5,
        parser_used="pymupdf",
        parser_time_seconds=0.1,
    )


def _make_docling_result() -> PDFContent:
    """Return a typical Docling parse result."""
    return PDFContent(
        raw_text="Docling parsed text",
        sections=[Section(title="Introduction", content="Docling text\n", level=1)],
        tables=[],
        figures=[],
        page_count=0,
        parser_used="docling",
        parser_time_seconds=5.0,
    )


def _make_svc(**kwargs):
    """Create a PDFParserService instance."""
    from src.services.pdf_parser.service import PDFParserService

    return PDFParserService(**kwargs)


# ---------------------------------------------------------------------------
# PDFParserService — File Validation (FR-1)
# ---------------------------------------------------------------------------


class TestValidateFile:
    """FR-1: File validation."""

    def test_file_not_found(self, tmp_path):
        svc = _make_svc()
        with pytest.raises(PDFValidationError, match="not found"):
            svc._validate_file(tmp_path / "missing.pdf")

    def test_not_pdf_extension(self, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        svc = _make_svc()
        with pytest.raises(PDFValidationError, match="Not a PDF"):
            svc._validate_file(txt)

    def test_file_too_large(self, tmp_path):
        big = _make_pdf(tmp_path / "big.pdf", size_bytes=51 * 1024 * 1024)
        svc = _make_svc(max_file_size_mb=50)
        with pytest.raises(PDFValidationError, match="too large"):
            svc._validate_file(big)

    def test_invalid_magic_bytes(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        _make_non_pdf(bad)
        svc = _make_svc()
        with pytest.raises(PDFValidationError, match="Invalid PDF"):
            svc._validate_file(bad)

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        svc = _make_svc()
        with pytest.raises(PDFValidationError, match="Invalid PDF"):
            svc._validate_file(empty)

    def test_valid_pdf(self, tmp_path):
        pdf = _make_pdf(tmp_path / "valid.pdf")
        svc = _make_svc()
        svc._validate_file(pdf)  # should not raise


# ---------------------------------------------------------------------------
# PyMuPDF parsing tests (FR-2a)
# ---------------------------------------------------------------------------


class TestParsePyMuPDF:
    """FR-2a: PyMuPDF primary parsing."""

    def test_pymupdf_returns_pdf_content(self, tmp_path):
        svc = _make_svc()
        pymupdf_result = _make_pymupdf_result()

        with patch.object(svc, "_parse_with_pymupdf", return_value=pymupdf_result):
            result = svc._parse_sync(tmp_path / "test.pdf")

        assert isinstance(result, PDFContent)
        assert result.parser_used == "pymupdf"
        assert result.raw_text == "Full paper text here"
        assert len(result.sections) == 1

    def test_pymupdf_is_tried_first(self, tmp_path):
        svc = _make_svc()
        pymupdf_result = _make_pymupdf_result()

        with patch.object(svc, "_parse_with_pymupdf", return_value=pymupdf_result) as mock_pymupdf, \
             patch.object(svc, "_parse_with_docling") as mock_docling:
            svc._parse_sync(tmp_path / "test.pdf")

        mock_pymupdf.assert_called_once()
        mock_docling.assert_not_called()

    def test_pymupdf_page_count(self, tmp_path):
        svc = _make_svc()
        result = _make_pymupdf_result()
        result.page_count = 12

        with patch.object(svc, "_parse_with_pymupdf", return_value=result):
            content = svc._parse_sync(tmp_path / "test.pdf")

        assert content.page_count == 12

    def test_pymupdf_extracts_figures_from_text(self, tmp_path):
        svc = _make_svc()
        result = _make_pymupdf_result()
        result.figures = ["Figure 1: Architecture diagram", "Fig. 2: Results"]

        with patch.object(svc, "_parse_with_pymupdf", return_value=result):
            content = svc._parse_sync(tmp_path / "test.pdf")

        assert len(content.figures) == 2
        assert "Architecture" in content.figures[0]


# ---------------------------------------------------------------------------
# Docling fallback tests (FR-2b)
# ---------------------------------------------------------------------------


class TestDoclingFallback:
    """FR-2b: Docling fallback when PyMuPDF fails."""

    def test_falls_back_to_docling_on_pymupdf_error(self, tmp_path):
        svc = _make_svc()
        docling_result = _make_docling_result()

        with patch.object(svc, "_parse_with_pymupdf", side_effect=RuntimeError("fitz crashed")), \
             patch.object(svc, "_parse_with_docling", return_value=docling_result):
            result = svc._parse_sync(tmp_path / "test.pdf")

        assert result.parser_used == "docling"
        assert result.raw_text == "Docling parsed text"

    def test_falls_back_on_import_error(self, tmp_path):
        svc = _make_svc()
        docling_result = _make_docling_result()

        with patch.object(svc, "_parse_with_pymupdf", side_effect=ImportError("no fitz")), \
             patch.object(svc, "_parse_with_docling", return_value=docling_result):
            result = svc._parse_sync(tmp_path / "test.pdf")

        assert result.parser_used == "docling"

    def test_both_parsers_fail_raises(self, tmp_path):
        svc = _make_svc()

        with patch.object(svc, "_parse_with_pymupdf", side_effect=RuntimeError("pymupdf failed")), \
             patch.object(svc, "_parse_with_docling", side_effect=RuntimeError("docling failed")):
            with pytest.raises(PDFParsingError, match="Both parsers failed"):
                svc._parse_sync(tmp_path / "test.pdf")

    def test_docling_fallback_disabled(self, tmp_path):
        svc = _make_svc(enable_docling_fallback=False)

        with patch.object(svc, "_parse_with_pymupdf", side_effect=RuntimeError("pymupdf failed")):
            with pytest.raises(PDFParsingError, match="fallback is disabled"):
                svc._parse_sync(tmp_path / "test.pdf")

    def test_docling_parsing_error_propagates(self, tmp_path):
        svc = _make_svc()

        with patch.object(svc, "_parse_with_pymupdf", side_effect=RuntimeError("fitz error")), \
             patch.object(svc, "_parse_with_docling", side_effect=PDFParsingError("Docling not installed")):
            with pytest.raises(PDFParsingError, match="Docling not installed"):
                svc._parse_sync(tmp_path / "test.pdf")


# ---------------------------------------------------------------------------
# Docling standalone parsing tests
# ---------------------------------------------------------------------------


class TestDoclingParsing:
    """Direct Docling parsing tests."""

    def test_docling_extracts_sections(self, tmp_path):
        doc = _fake_docling_doc(
            texts=[
                _heading_item("Introduction"),
                _text_item("This is the intro."),
                _heading_item("Methods", level=2),
                _text_item("We did X."),
            ]
        )
        svc = _make_svc()
        converter = MagicMock()
        converter.convert.return_value.document = doc
        svc._docling_converter = converter

        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_with_docling(pdf)

        assert len(result.sections) == 2
        assert result.sections[0].title == "Introduction"
        assert "intro" in result.sections[0].content.lower()
        assert result.sections[1].title == "Methods"

    def test_docling_extracts_tables(self, tmp_path):
        doc = _fake_docling_doc(tables=[_table_item("col1|col2\nA|B")])
        svc = _make_svc()
        converter = MagicMock()
        converter.convert.return_value.document = doc
        svc._docling_converter = converter

        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_with_docling(pdf)

        assert len(result.tables) == 1
        assert "col1" in result.tables[0]

    def test_docling_extracts_figures(self, tmp_path):
        doc = _fake_docling_doc(
            pictures=[
                _picture_item("Figure 1: Architecture"),
                _picture_item(None),  # no caption
            ]
        )
        svc = _make_svc()
        converter = MagicMock()
        converter.convert.return_value.document = doc
        svc._docling_converter = converter

        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_with_docling(pdf)

        assert len(result.figures) == 1
        assert "Architecture" in result.figures[0]

    def test_docling_raw_text(self, tmp_path):
        doc = _fake_docling_doc(raw_text="Full paper text here")
        svc = _make_svc()
        converter = MagicMock()
        converter.convert.return_value.document = doc
        svc._docling_converter = converter

        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_with_docling(pdf)

        assert result.raw_text == "Full paper text here"

    def test_docling_no_headings_creates_default_section(self, tmp_path):
        doc = _fake_docling_doc(
            texts=[
                _text_item("Paragraph 1."),
                _text_item("Paragraph 2."),
            ]
        )
        svc = _make_svc()
        converter = MagicMock()
        converter.convert.return_value.document = doc
        svc._docling_converter = converter

        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_with_docling(pdf)

        assert len(result.sections) == 1
        assert result.sections[0].title == "Introduction"
        assert "Paragraph 1" in result.sections[0].content


class TestDoclingNotInstalled:
    """Docling import failure when used as fallback."""

    def test_raises_on_missing_docling(self, tmp_path):
        svc = _make_svc()
        svc._docling_converter = None

        with patch("builtins.__import__", side_effect=ImportError("no docling")):
            pdf = _make_pdf(tmp_path / "test.pdf")
            with pytest.raises(PDFParsingError, match="Docling not installed"):
                svc._parse_with_docling(pdf)


# ---------------------------------------------------------------------------
# Async parsing tests (FR-3)
# ---------------------------------------------------------------------------


class TestParsePdfAsync:
    """FR-3: Async parsing with timeout."""

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, tmp_path):
        svc = _make_svc(timeout=5)
        pdf = _make_pdf(tmp_path / "test.pdf")
        pymupdf_result = _make_pymupdf_result()

        with patch.object(svc, "_parse_sync", return_value=pymupdf_result):
            result = await svc.parse_pdf(pdf)

        assert isinstance(result, PDFContent)
        assert result.parser_used == "pymupdf"

    @pytest.mark.asyncio
    async def test_parse_pdf_validation_failure(self, tmp_path):
        svc = _make_svc()
        with pytest.raises(PDFValidationError):
            await svc.parse_pdf(tmp_path / "missing.pdf")

    @pytest.mark.asyncio
    async def test_parse_pdf_timeout(self, tmp_path):
        import time as time_mod

        svc = _make_svc(timeout=0.1)

        def slow_parse(path):
            time_mod.sleep(5)
            return _make_pymupdf_result()

        svc._parse_sync = slow_parse
        pdf = _make_pdf(tmp_path / "test.pdf")

        with pytest.raises(PDFParsingError, match="timed out"):
            await svc.parse_pdf(pdf)


# ---------------------------------------------------------------------------
# Batch parsing tests (FR-4)
# ---------------------------------------------------------------------------


class TestParseMultiple:
    """FR-4: Batch parsing (now parallel via asyncio.gather)."""

    @pytest.mark.asyncio
    async def test_all_success(self, tmp_path):
        svc = _make_svc(timeout=5)
        pymupdf_result = _make_pymupdf_result()
        pdfs = [_make_pdf(tmp_path / f"p{i}.pdf") for i in range(3)]

        with patch.object(svc, "_parse_sync", return_value=pymupdf_result):
            results = await svc.parse_multiple(pdfs)

        assert len(results) == 3
        assert all(v is not None for v in results.values())

    @pytest.mark.asyncio
    async def test_partial_failure_continue(self, tmp_path):
        svc = _make_svc(timeout=5)
        pymupdf_result = _make_pymupdf_result()
        good = _make_pdf(tmp_path / "good.pdf")
        bad = tmp_path / "missing.pdf"  # does not exist

        with patch.object(svc, "_parse_sync", return_value=pymupdf_result):
            results = await svc.parse_multiple([bad, good], continue_on_error=True)

        assert results["missing.pdf"] is None
        assert results["good.pdf"] is not None

    @pytest.mark.asyncio
    async def test_stop_on_error(self, tmp_path):
        svc = _make_svc(timeout=5)
        bad = tmp_path / "missing.pdf"

        with pytest.raises(PDFValidationError):
            await svc.parse_multiple([bad], continue_on_error=False)

    @pytest.mark.asyncio
    async def test_empty_list(self):
        svc = _make_svc()
        results = await svc.parse_multiple([])
        assert results == {}


# ---------------------------------------------------------------------------
# Factory tests (FR-5)
# ---------------------------------------------------------------------------


class TestFactory:
    """FR-5: Factory & configuration."""

    def test_factory_creates_service(self):
        from src.services.pdf_parser.factory import make_pdf_parser_service, reset_pdf_parser_cache

        reset_pdf_parser_cache()
        svc = make_pdf_parser_service()
        from src.services.pdf_parser.service import PDFParserService

        assert isinstance(svc, PDFParserService)
        reset_pdf_parser_cache()

    def test_factory_singleton(self):
        from src.services.pdf_parser.factory import make_pdf_parser_service, reset_pdf_parser_cache

        reset_pdf_parser_cache()
        a = make_pdf_parser_service()
        b = make_pdf_parser_service()
        assert a is b
        reset_pdf_parser_cache()

    def test_factory_uses_settings(self):
        from src.services.pdf_parser.factory import make_pdf_parser_service, reset_pdf_parser_cache

        reset_pdf_parser_cache()
        svc = make_pdf_parser_service()
        assert svc.max_pages == 30
        assert svc.max_file_size_mb == 50
        assert svc.timeout == 120
        assert svc.enable_docling_fallback is True
        reset_pdf_parser_cache()


# ---------------------------------------------------------------------------
# Cleanup tests (FR-6)
# ---------------------------------------------------------------------------


class TestCleanup:
    """FR-6: Resource cleanup."""

    def test_close_shuts_executor(self):
        svc = _make_svc()
        svc.close()
        assert svc._docling_converter is None

    def test_close_idempotent(self):
        svc = _make_svc()
        svc.close()
        svc.close()  # should not raise
