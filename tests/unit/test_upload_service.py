"""Tests for the PDF upload service (UploadService).

Tests validation, metadata extraction, and the full upload pipeline.
All external services are mocked.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile
from src.schemas.pdf import PDFContent, Section
from src.services.upload.service import UploadService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PDF_BYTES = b"%PDF-1.4 fake content for testing" + b"\x00" * 1000


def _make_pdf_content(
    raw_text: str = "This is the full text of the paper about transformers.",
    sections: list[Section] | None = None,
) -> PDFContent:
    if sections is None:
        sections = [
            Section(title="Abstract", content="We present a novel approach to transformers.", level=1),
            Section(title="Introduction", content="Transformers have revolutionized NLP.", level=1),
            Section(title="Methods", content="We use self-attention mechanisms.", level=1),
        ]
    return PDFContent(
        raw_text=raw_text,
        sections=sections,
        tables=[],
        figures=[],
        page_count=10,
        parser_used="docling",
        parser_time_seconds=2.0,
    )


def _make_upload_file(
    content: bytes = VALID_PDF_BYTES,
    filename: str = "test_paper.pdf",
    content_type: str = "application/pdf",
) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename, headers={"content-type": content_type})


def _make_paper_orm():
    paper = MagicMock()
    paper.id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    paper.arxiv_id = "upload_test123"
    paper.title = "Test Paper"
    paper.authors = []
    paper.abstract = "Test abstract"
    paper.parsing_status = "success"
    return paper


def _make_chunks():
    from src.schemas.indexing import ChunkMetadata, TextChunk

    return [
        TextChunk(
            text="Chunk 1 text about transformers",
            metadata=ChunkMetadata(
                chunk_index=0, start_char=0, end_char=100, word_count=50, overlap_with_previous=0, overlap_with_next=10
            ),
            arxiv_id="upload_test123",
            paper_id="12345678-1234-5678-1234-567812345678",
        ),
        TextChunk(
            text="Chunk 2 text about attention",
            metadata=ChunkMetadata(
                chunk_index=1, start_char=90, end_char=200, word_count=50, overlap_with_previous=10, overlap_with_next=0
            ),
            arxiv_id="upload_test123",
            paper_id="12345678-1234-5678-1234-567812345678",
        ),
    ]


@pytest.fixture
def mock_pdf_parser():
    parser = AsyncMock()
    parser.parse_pdf = AsyncMock(return_value=_make_pdf_content())
    return parser


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=_make_paper_orm())
    return repo


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_text_chunker():
    chunker = MagicMock()
    chunker.chunk_paper = MagicMock(return_value=_make_chunks())
    return chunker


@pytest.fixture
def mock_embeddings_client():
    client = AsyncMock()
    client.embed_passages = AsyncMock(return_value=[[0.1] * 1024, [0.2] * 1024])
    return client


@pytest.fixture
def mock_opensearch_client():
    client = MagicMock()
    client.bulk_index_chunks = MagicMock(return_value={"success": 2, "failed": 0})
    return client


@pytest.fixture
def upload_service():
    return UploadService(max_file_size_mb=50)


# ---------------------------------------------------------------------------
# FR-1: PDF File Validation
# ---------------------------------------------------------------------------


class TestValidatePDFFile:
    """Tests for UploadService.validate_pdf()."""

    @pytest.mark.asyncio
    async def test_validate_pdf_file_valid(self, upload_service):
        """Valid PDF passes all validation checks."""
        file = _make_upload_file()
        content = await upload_service.validate_pdf(file)
        assert content.startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_validate_pdf_file_too_large(self, upload_service):
        """Rejects files exceeding the size limit."""
        from src.exceptions import PDFValidationError

        big_content = b"%PDF-1.4 " + b"\x00" * (51 * 1024 * 1024)  # 51MB
        file = _make_upload_file(content=big_content)
        with pytest.raises(PDFValidationError, match="exceeds.*50"):
            await upload_service.validate_pdf(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_wrong_extension(self, upload_service):
        """Rejects files without .pdf extension."""
        from src.exceptions import PDFValidationError

        file = _make_upload_file(filename="paper.txt")
        with pytest.raises(PDFValidationError, match="PDF"):
            await upload_service.validate_pdf(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_wrong_magic_bytes(self, upload_service):
        """Rejects files with .pdf extension but non-PDF content."""
        from src.exceptions import PDFValidationError

        file = _make_upload_file(content=b"not a pdf file at all")
        with pytest.raises(PDFValidationError, match="Invalid PDF"):
            await upload_service.validate_pdf(file)

    @pytest.mark.asyncio
    async def test_validate_pdf_file_empty(self, upload_service):
        """Rejects empty files."""
        from src.exceptions import PDFValidationError

        file = _make_upload_file(content=b"")
        with pytest.raises(PDFValidationError, match="[Ee]mpty"):
            await upload_service.validate_pdf(file)


# ---------------------------------------------------------------------------
# FR-2/FR-3: Metadata Extraction
# ---------------------------------------------------------------------------


class TestMetadataExtraction:
    """Tests for UploadService._extract_metadata()."""

    def test_extracts_title_from_sections(self, upload_service):
        """Extracts title from first non-Abstract section heading."""
        content = _make_pdf_content()
        meta = upload_service._extract_metadata(content, "test_paper.pdf")
        # Title should be derived since sections don't have a clear "title" section
        assert meta["title"]  # non-empty

    def test_extracts_abstract_from_section(self, upload_service):
        """Extracts abstract from 'Abstract' section content."""
        content = _make_pdf_content()
        meta = upload_service._extract_metadata(content, "test_paper.pdf")
        assert "novel approach" in meta["abstract"]

    def test_no_abstract_fallback(self, upload_service):
        """Falls back to first N chars of raw_text when no Abstract section."""
        content = _make_pdf_content(
            raw_text="This is a long paper about neural networks and their applications in science.",
            sections=[Section(title="Introduction", content="Some intro content.", level=1)],
        )
        meta = upload_service._extract_metadata(content, "test_paper.pdf")
        assert "neural networks" in meta["abstract"]

    def test_title_from_filename_fallback(self, upload_service):
        """Uses filename (sans extension) when no title can be extracted from content."""
        content = _make_pdf_content(raw_text="", sections=[])
        meta = upload_service._extract_metadata(content, "my_research_paper.pdf")
        assert meta["title"] == "my_research_paper"


# ---------------------------------------------------------------------------
# FR-4/FR-5: Full Upload Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessUpload:
    """Tests for UploadService.process_upload()."""

    async def test_process_upload_full_pipeline(
        self,
        upload_service,
        mock_pdf_parser,
        mock_paper_repo,
        mock_session,
        mock_text_chunker,
        mock_embeddings_client,
        mock_opensearch_client,
    ):
        """Full pipeline: parse → save → chunk → embed → index."""
        file = _make_upload_file()
        result = await upload_service.process_upload(
            file=file,
            pdf_parser=mock_pdf_parser,
            paper_repo=mock_paper_repo,
            session=mock_session,
            text_chunker=mock_text_chunker,
            embeddings_client=mock_embeddings_client,
            opensearch_client=mock_opensearch_client,
        )

        assert result.paper_id is not None
        assert result.chunks_indexed == 2
        assert result.parsing_status == "success"
        assert result.indexing_status == "success"

    async def test_process_upload_creates_paper_record(
        self,
        upload_service,
        mock_pdf_parser,
        mock_paper_repo,
        mock_session,
        mock_text_chunker,
        mock_embeddings_client,
        mock_opensearch_client,
    ):
        """Verify Paper record created with correct fields."""
        file = _make_upload_file()
        await upload_service.process_upload(
            file=file,
            pdf_parser=mock_pdf_parser,
            paper_repo=mock_paper_repo,
            session=mock_session,
            text_chunker=mock_text_chunker,
            embeddings_client=mock_embeddings_client,
            opensearch_client=mock_opensearch_client,
        )

        mock_paper_repo.create.assert_called_once()
        paper_create = mock_paper_repo.create.call_args[0][0]
        assert paper_create.arxiv_id.startswith("upload_")
        assert paper_create.parsing_status == "success"
        assert paper_create.pdf_content is not None

    async def test_process_upload_chunks_and_indexes(
        self,
        upload_service,
        mock_pdf_parser,
        mock_paper_repo,
        mock_session,
        mock_text_chunker,
        mock_embeddings_client,
        mock_opensearch_client,
    ):
        """Verify chunking + embedding + indexing flow is called."""
        file = _make_upload_file()
        await upload_service.process_upload(
            file=file,
            pdf_parser=mock_pdf_parser,
            paper_repo=mock_paper_repo,
            session=mock_session,
            text_chunker=mock_text_chunker,
            embeddings_client=mock_embeddings_client,
            opensearch_client=mock_opensearch_client,
        )

        mock_text_chunker.chunk_paper.assert_called_once()
        mock_embeddings_client.embed_passages.assert_called_once()
        mock_opensearch_client.bulk_index_chunks.assert_called_once()

    async def test_process_upload_indexing_fails_gracefully(
        self,
        upload_service,
        mock_pdf_parser,
        mock_paper_repo,
        mock_session,
        mock_text_chunker,
        mock_embeddings_client,
        mock_opensearch_client,
    ):
        """Paper saved even when indexing fails — returns warning."""
        mock_embeddings_client.embed_passages = AsyncMock(side_effect=Exception("Jina API down"))

        file = _make_upload_file()
        result = await upload_service.process_upload(
            file=file,
            pdf_parser=mock_pdf_parser,
            paper_repo=mock_paper_repo,
            session=mock_session,
            text_chunker=mock_text_chunker,
            embeddings_client=mock_embeddings_client,
            opensearch_client=mock_opensearch_client,
        )

        # Paper should still be saved
        mock_paper_repo.create.assert_called_once()
        mock_session.commit.assert_called()
        # But indexing failed
        assert result.indexing_status == "failed"
        assert result.chunks_indexed == 0
        assert len(result.warnings) > 0
        assert result.paper_id is not None
