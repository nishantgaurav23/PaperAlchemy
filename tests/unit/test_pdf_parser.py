"""Tests for PDF parser service (S3.3)."""

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
            parser_used="docling",
            parser_time_seconds=1.5,
        )
        assert c.raw_text == "hello"
        assert len(c.sections) == 1
        assert c.sections[0].title == "Intro"
        assert len(c.tables) == 1
        assert len(c.figures) == 1
        assert c.page_count == 10
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


def _fake_doc(texts=None, tables=None, pictures=None, raw_text="raw content"):
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


def _make_svc_with_mock_converter(doc, **kwargs):
    """Create a PDFParserService with a pre-set mock converter."""
    from src.services.pdf_parser.service import PDFParserService

    converter = MagicMock()
    converter.convert.return_value.document = doc
    svc = PDFParserService(**kwargs)
    svc._converter = converter
    return svc


# ---------------------------------------------------------------------------
# PDFParserService tests
# ---------------------------------------------------------------------------


class TestValidateFile:
    """FR-1: File validation."""

    def test_file_not_found(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        with pytest.raises(PDFValidationError, match="not found"):
            svc._validate_file(tmp_path / "missing.pdf")

    def test_not_pdf_extension(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        svc = PDFParserService()
        with pytest.raises(PDFValidationError, match="Not a PDF"):
            svc._validate_file(txt)

    def test_file_too_large(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        big = _make_pdf(tmp_path / "big.pdf", size_bytes=51 * 1024 * 1024)
        svc = PDFParserService(max_file_size_mb=50)
        with pytest.raises(PDFValidationError, match="too large"):
            svc._validate_file(big)

    def test_invalid_magic_bytes(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        bad = tmp_path / "bad.pdf"
        _make_non_pdf(bad)
        svc = PDFParserService()
        with pytest.raises(PDFValidationError, match="Invalid PDF"):
            svc._validate_file(bad)

    def test_empty_file(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        svc = PDFParserService()
        with pytest.raises(PDFValidationError, match="Invalid PDF"):
            svc._validate_file(empty)

    def test_valid_pdf(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        pdf = _make_pdf(tmp_path / "valid.pdf")
        svc = PDFParserService()
        svc._validate_file(pdf)  # should not raise


class TestParseSyncSectionExtraction:
    """FR-2: Section-aware parsing via Docling (mocked)."""

    def test_extracts_sections(self, tmp_path):
        doc = _fake_doc(
            texts=[
                _heading_item("Introduction"),
                _text_item("This is the intro."),
                _heading_item("Methods", level=2),
                _text_item("We did X."),
            ]
        )
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert isinstance(result, PDFContent)
        assert len(result.sections) == 2
        assert result.sections[0].title == "Introduction"
        assert "intro" in result.sections[0].content.lower()
        assert result.sections[1].title == "Methods"

    def test_extracts_tables(self, tmp_path):
        doc = _fake_doc(tables=[_table_item("col1|col2\nA|B")])
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert len(result.tables) == 1
        assert "col1" in result.tables[0]

    def test_extracts_figures(self, tmp_path):
        doc = _fake_doc(
            pictures=[
                _picture_item("Figure 1: Architecture"),
                _picture_item(None),  # no caption
            ]
        )
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert len(result.figures) == 1
        assert "Architecture" in result.figures[0]

    def test_raw_text(self, tmp_path):
        doc = _fake_doc(raw_text="Full paper text here")
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert result.raw_text == "Full paper text here"

    def test_no_headings_creates_default_section(self, tmp_path):
        doc = _fake_doc(
            texts=[
                _text_item("Paragraph 1."),
                _text_item("Paragraph 2."),
            ]
        )
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert len(result.sections) == 1
        assert result.sections[0].title == "Introduction"
        assert "Paragraph 1" in result.sections[0].content

    def test_parser_time_recorded(self, tmp_path):
        doc = _fake_doc()
        svc = _make_svc_with_mock_converter(doc)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = svc._parse_sync(pdf)

        assert result.parser_time_seconds >= 0
        assert result.parser_used == "docling"


class TestDoclingNotInstalled:
    """FR-2 edge case: Docling import fails."""

    def test_raises_on_missing_docling(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        svc._converter = None

        with patch("builtins.__import__", side_effect=ImportError("no docling")):
            pdf = _make_pdf(tmp_path / "test.pdf")
            with pytest.raises(PDFParsingError, match="Docling not installed"):
                svc._parse_sync(pdf)


# ---------------------------------------------------------------------------
# Async parsing tests
# ---------------------------------------------------------------------------


class TestParsePdfAsync:
    """FR-3: Async parsing with timeout."""

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, tmp_path):
        doc = _fake_doc(raw_text="text")
        svc = _make_svc_with_mock_converter(doc, timeout=5)
        pdf = _make_pdf(tmp_path / "test.pdf")
        result = await svc.parse_pdf(pdf)

        assert isinstance(result, PDFContent)
        assert result.raw_text == "text"

    @pytest.mark.asyncio
    async def test_parse_pdf_validation_failure(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        with pytest.raises(PDFValidationError):
            await svc.parse_pdf(tmp_path / "missing.pdf")

    @pytest.mark.asyncio
    async def test_parse_pdf_timeout(self, tmp_path):
        import time as time_mod

        doc = _fake_doc()
        svc = _make_svc_with_mock_converter(doc, timeout=0.1)

        # Override _parse_sync to be slow
        original_parse = svc._parse_sync

        def slow_parse(path):
            time_mod.sleep(5)
            return original_parse(path)

        svc._parse_sync = slow_parse
        pdf = _make_pdf(tmp_path / "test.pdf")

        with pytest.raises(PDFParsingError, match="timed out"):
            await svc.parse_pdf(pdf)


# ---------------------------------------------------------------------------
# Batch parsing tests
# ---------------------------------------------------------------------------


class TestParseMultiple:
    """FR-4: Batch parsing."""

    @pytest.mark.asyncio
    async def test_all_success(self, tmp_path):
        doc = _fake_doc(raw_text="text")
        svc = _make_svc_with_mock_converter(doc, timeout=5)
        pdfs = [_make_pdf(tmp_path / f"p{i}.pdf") for i in range(3)]
        results = await svc.parse_multiple(pdfs)

        assert len(results) == 3
        assert all(v is not None for v in results.values())

    @pytest.mark.asyncio
    async def test_partial_failure_continue(self, tmp_path):
        doc = _fake_doc(raw_text="text")
        svc = _make_svc_with_mock_converter(doc, timeout=5)
        good = _make_pdf(tmp_path / "good.pdf")
        bad = tmp_path / "missing.pdf"  # does not exist

        results = await svc.parse_multiple([bad, good], continue_on_error=True)
        assert results["missing.pdf"] is None
        assert results["good.pdf"] is not None

    @pytest.mark.asyncio
    async def test_stop_on_error(self, tmp_path):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService(timeout=5)
        bad = tmp_path / "missing.pdf"

        with pytest.raises(PDFValidationError):
            await svc.parse_multiple([bad], continue_on_error=False)

    @pytest.mark.asyncio
    async def test_empty_list(self):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        results = await svc.parse_multiple([])
        assert results == {}


# ---------------------------------------------------------------------------
# Factory tests
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
        reset_pdf_parser_cache()


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestCleanup:
    """FR-6: Resource cleanup."""

    def test_close_shuts_executor(self):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        svc.close()
        assert svc._converter is None

    def test_close_idempotent(self):
        from src.services.pdf_parser.service import PDFParserService

        svc = PDFParserService()
        svc.close()
        svc.close()  # should not raise
