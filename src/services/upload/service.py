"""PDF upload service: validate → parse → save → chunk → embed → index → analyze."""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from src.exceptions import PDFParsingError, PDFValidationError
from src.repositories.paper import PaperRepository
from src.schemas.api.upload import UploadResponse
from src.schemas.paper import PaperCreate
from src.schemas.pdf import PDFContent
from src.services.analysis import run_ai_analysis

logger = logging.getLogger(__name__)


class UploadService:
    """Orchestrates PDF upload: validate → parse → save → chunk → embed → index."""

    def __init__(self, max_file_size_mb: int = 50) -> None:
        self.max_file_size_mb = max_file_size_mb

    # ------------------------------------------------------------------
    # FR-1: PDF File Validation
    # ------------------------------------------------------------------

    async def validate_pdf(self, file: UploadFile) -> bytes:
        """Validate uploaded file is a valid PDF within size limits.

        Returns the file bytes on success. Raises PDFValidationError on failure.
        """
        filename = file.filename or "unknown"

        # Check extension
        if not filename.lower().endswith(".pdf"):
            raise PDFValidationError(f"Only PDF files are accepted. Got: {filename}")

        # Read content
        content = await file.read()
        await file.seek(0)

        # Check empty
        if not content:
            raise PDFValidationError("Empty file uploaded")

        # Check size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise PDFValidationError(
                f"File size {size_mb:.1f}MB exceeds limit of {self.max_file_size_mb}MB",
                status_code=413,
            )

        # Check magic bytes
        if not content.startswith(b"%PDF"):
            raise PDFValidationError(f"Invalid PDF file: {filename} does not have PDF magic bytes")

        return content

    # ------------------------------------------------------------------
    # FR-2: Metadata Extraction
    # ------------------------------------------------------------------

    # Section titles to skip when looking for the paper title
    _SKIP_SECTION_TITLES = {
        "abstract", "copyright", "license", "acknowledgements",
        "acknowledgments", "references", "bibliography",
    }

    @staticmethod
    def _looks_like_boilerplate(text: str) -> bool:
        """Detect copyright/license boilerplate or venue headers that shouldn't be used as a title."""
        lower = text.lower().strip()
        boilerplate_phrases = [
            "permission", "granted", "copyright", "license", "redistribution",
            "provided", "attribution", "hereby", "rights reserved",
            "creative commons", "open access",
        ]
        if sum(1 for p in boilerplate_phrases if p in lower) >= 2:
            return True

        # Venue/conference headers common in academic PDFs (appear before the title)
        venue_patterns = [
            "published as", "accepted at", "accepted to", "to appear in",
            "presented at", "proceedings of", "workshop on",
            "conference on", "conference paper", "journal of",
            "transactions on", "preprint", "under review",
            "arxiv:", "arxiv preprint",
        ]
        return any(lower.startswith(p) or p in lower for p in venue_patterns)

    def _extract_metadata(self, content: PDFContent, filename: str) -> dict[str, Any]:
        """Extract title, abstract, and authors from parsed PDF content."""
        title = ""
        abstract = ""
        authors: list[str] = []

        # Try to extract abstract from "Abstract" section
        for section in content.sections:
            if section.title.lower().strip() == "abstract":
                abstract = section.content.strip()
                break

        # Fallback abstract: first 500 chars of raw_text
        if not abstract and content.raw_text:
            abstract = content.raw_text[:500].strip()

        # Try to extract title from the first line(s) of raw text — most
        # academic PDFs start with the paper title before anything else.
        if content.raw_text:
            first_lines = content.raw_text.strip().split("\n")
            for line in first_lines[:5]:
                candidate = line.strip()
                # Skip empty lines and very long lines (likely body text)
                if not candidate or len(candidate) > 200:
                    continue
                # Skip lines that look like boilerplate
                if self._looks_like_boilerplate(candidate):
                    continue
                # A good title candidate: non-empty, reasonable length, not boilerplate
                if len(candidate) >= 5:
                    title = candidate
                    break

        # Fallback: try section titles (skip boilerplate sections)
        if not title:
            for section in content.sections:
                section_title = section.title.strip()
                if not section_title:
                    continue
                if section_title.lower() in self._SKIP_SECTION_TITLES:
                    continue
                if len(section_title) > 200:
                    continue
                if self._looks_like_boilerplate(section_title):
                    continue
                title = section_title
                break

        # Fallback title: filename without extension
        if not title:
            title = Path(filename).stem

        # Try to extract authors: look for lines between title and abstract
        # in the raw text (common academic paper layout)
        if content.raw_text and title:
            raw_lines = content.raw_text.strip().split("\n")
            title_found = False
            for line in raw_lines[:20]:
                stripped = line.strip()
                if not stripped:
                    continue
                if not title_found:
                    if title in stripped:
                        title_found = True
                    continue
                # Stop at abstract or section headings
                lower = stripped.lower()
                if lower.startswith("abstract") or lower.startswith("1 ") or lower.startswith("1."):
                    break
                # Skip very long lines (likely body text)
                if len(stripped) > 300:
                    break
                # Lines between title and abstract are likely authors
                # Split on common author separators
                if "," in stripped or " and " in stripped.lower():
                    # Clean up: remove affiliations (lines with numbers, @, university, etc.)
                    if any(kw in stripped.lower() for kw in ["@", "university", "institute", "department", "lab"]):
                        continue
                    # Split authors by comma, "and", or similar
                    raw_authors = stripped.replace(" and ", ", ").split(",")
                    for a in raw_authors:
                        name = a.strip().rstrip("*†‡§∥").strip()
                        if name and len(name) > 2 and not name[0].isdigit():
                            authors.append(name)
                    if authors:
                        break

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
        }

    # ------------------------------------------------------------------
    # FR-3/4/5: Full Upload Pipeline
    # ------------------------------------------------------------------

    async def process_upload(
        self,
        file: UploadFile,
        pdf_parser: Any,
        paper_repo: PaperRepository,
        session: AsyncSession,
        text_chunker: Any,
        embeddings_client: Any,
        opensearch_client: Any,
        llm_provider: Any = None,
    ) -> UploadResponse:
        """Full upload pipeline: validate → parse → save → chunk → embed → index → analyze."""
        warnings: list[str] = []
        filename = file.filename or "unknown.pdf"

        # Step 1: Validate
        pdf_bytes = await self.validate_pdf(file)

        # Step 2: Parse PDF via temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                pdf_content = await pdf_parser.parse_pdf(tmp_path)
            except PDFParsingError:
                raise
            except Exception as e:
                raise PDFParsingError(f"Failed to parse PDF: {e}") from e

        # Step 3: Extract metadata
        metadata = self._extract_metadata(pdf_content, filename)

        # Step 4: Create paper record
        upload_id = f"upload_{uuid.uuid4().hex[:16]}"
        sections_dicts = [s.model_dump() for s in pdf_content.sections]

        paper_create = PaperCreate(
            arxiv_id=upload_id,
            title=metadata["title"],
            authors=metadata["authors"],
            abstract=metadata["abstract"],
            categories=["uploaded"],
            published_date=datetime.now(UTC),
            pdf_url=f"upload://{filename}",
            pdf_content=pdf_content.raw_text,
            sections=sections_dicts,
            parsing_status="success",
        )

        paper = await paper_repo.create(paper_create)
        await session.commit()

        # Step 5: Chunk + Embed + Index (graceful degradation)
        chunks_indexed = 0
        indexing_status = "success"

        try:
            chunks = text_chunker.chunk_paper(
                title=metadata["title"],
                abstract=metadata["abstract"],
                full_text=pdf_content.raw_text,
                arxiv_id=upload_id,
                paper_id=str(paper.id),
                sections=pdf_content.sections,
            )

            if not chunks:
                warnings.append("No chunks generated from PDF content")
                indexing_status = "skipped"
            else:
                # Embed chunks
                chunk_texts = [c.text for c in chunks]
                embeddings = await embeddings_client.embed_passages(chunk_texts)

                # Build index documents
                index_docs = []
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    chunk_data = {
                        "chunk_id": f"{upload_id}_chunk_{chunk.metadata.chunk_index}",
                        "arxiv_id": upload_id,
                        "paper_id": str(paper.id),
                        "chunk_index": chunk.metadata.chunk_index,
                        "chunk_text": chunk.text,
                        "chunk_word_count": chunk.metadata.word_count,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "title": metadata["title"],
                        "abstract": metadata["abstract"],
                        "authors": metadata["authors"],
                    }
                    index_docs.append({"chunk_data": chunk_data, "embedding": embedding})

                result = opensearch_client.bulk_index_chunks(index_docs)
                chunks_indexed = result.get("success", 0)

        except Exception as e:
            logger.warning("Indexing failed for %s: %s", upload_id, e)
            indexing_status = "failed"
            warnings.append(f"Indexing failed: {e}")
            # Ensure paper is still saved
            await session.commit()

        if not metadata["abstract"]:
            warnings.append("No abstract found in PDF")

        # Step 6: AI Analysis (summary, highlights, methodology) — graceful degradation
        if llm_provider is not None:
            try:
                analysis_results = await run_ai_analysis(
                    paper_id=paper.id,
                    paper_repo=paper_repo,
                    session=session,
                    llm_provider=llm_provider,
                )
                warnings.extend(analysis_results.get("warnings", []))
                await session.commit()
            except Exception as e:
                logger.warning("AI analysis failed for %s: %s", upload_id, e)
                warnings.append(f"AI analysis failed: {e}")
        else:
            warnings.append("LLM provider not available — AI analysis skipped")

        message = f"Paper uploaded successfully: {metadata['title']}"
        if warnings:
            message += f" ({len(warnings)} warning(s))"

        return UploadResponse(
            paper_id=paper.id,
            arxiv_id=upload_id,
            title=metadata["title"],
            authors=metadata["authors"],
            abstract=metadata["abstract"],
            page_count=pdf_content.page_count,
            chunks_indexed=chunks_indexed,
            parsing_status="success",
            indexing_status=indexing_status,
            warnings=warnings,
            message=message,
        )
