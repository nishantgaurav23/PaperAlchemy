"""Section-aware text chunker for academic papers.

Hybrid strategy: section-based chunking when sections are available
(from PDF parser), falling back to word-based sliding window.

Pipeline: Paper → TextChunker → List[TextChunk] → Embeddings → OpenSearch
"""

from __future__ import annotations

import json
import logging
import re

from src.schemas.indexing import ChunkMetadata, TextChunk
from src.schemas.pdf import Section

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking academic papers into overlapping segments.

    Uses a hybrid strategy: section-based when sections are available,
    word-based with overlap as fallback.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
    ):
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    # =================================================================
    # FR-6: Main Entry Point
    # =================================================================

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: dict[str, str] | str | list | None = None,
    ) -> list[TextChunk]:
        """Chunk a paper using hybrid section-based approach.

        Tries section-based chunking first, falls back to word-based.
        """
        if sections:
            try:
                section_chunks = self._chunk_by_sections(title, abstract, arxiv_id, paper_id, sections)
                if section_chunks:
                    logger.info(f"Created {len(section_chunks)} section-based chunks for {arxiv_id}")
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section-based chunking failed for {arxiv_id}: {e}")

        logger.info(f"Using word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    # =================================================================
    # FR-2: Word-Based Chunking
    # =================================================================

    def chunk_text(self, text: str, arxiv_id: str, paper_id: str) -> list[TextChunk]:
        """Chunk text into overlapping segments using sliding window."""
        if not text or not text.strip():
            logger.warning(f"Empty text provided for paper {arxiv_id}")
            return []

        words = self._split_into_words(text)

        if len(words) < self.min_chunk_size:
            logger.warning(f"Text for paper {arxiv_id} has only {len(words)} words, less than minimum {self.min_chunk_size}")
            if words:
                return [
                    TextChunk(
                        text=self._reconstruct_text(words),
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text),
                            word_count=len(words),
                            overlap_with_previous=0,
                            overlap_with_next=0,
                        ),
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                    )
                ]
            return []

        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(words):
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            start_char = len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            end_char = len(" ".join(words[:chunk_end]))

            overlap_with_previous = min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            overlap_with_next = self.overlap_size if chunk_end < len(words) else 0

            chunk = TextChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_with_previous,
                    overlap_with_next=overlap_with_next,
                    section_title=None,
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            chunks.append(chunk)

            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            if chunk_end >= len(words):
                break

        logger.info(f"Chunked paper {arxiv_id}: {len(words)} words -> {len(chunks)} chunks")
        return chunks

    # =================================================================
    # FR-3: Section Parsing
    # =================================================================

    def _parse_sections(self, sections: dict[str, str] | str | list | None) -> dict[str, str]:
        """Parse heterogeneous section input formats into uniform dict."""
        if sections is None:
            return {}

        if isinstance(sections, dict):
            return sections

        if isinstance(sections, list):
            return self._parse_section_list(sections)

        if isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return self._parse_section_list(parsed)
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections JSON")

        return {}

    def _parse_section_list(self, sections: list) -> dict[str, str]:
        """Parse a list of section objects/dicts into a dict."""
        result = {}
        for i, section in enumerate(sections):
            if isinstance(section, Section):
                result[section.title] = section.content
            elif isinstance(section, dict):
                title = section.get("title", section.get("heading", f"Section {i + 1}"))
                content = section.get("content", section.get("text", ""))
                result[title] = content
            else:
                result[f"Section {i + 1}"] = str(section)
        return result

    # =================================================================
    # FR-4: Section Filtering
    # =================================================================

    def _filter_sections(self, sections_dict: dict[str, str], abstract: str) -> dict[str, str]:
        """Filter out metadata sections and abstract duplicates."""
        filtered = {}
        abstract_words = set(abstract.lower().split())

        for section_title, section_content in sections_dict.items():
            content_str = str(section_content).strip()

            if not content_str:
                continue

            if self._is_metadata_section(section_title):
                continue

            if self._is_duplicate_abstract(content_str, abstract, abstract_words):
                logger.debug(f"Skipping duplicate abstract section: {section_title}")
                continue

            if len(content_str.split()) < 20 and self._is_metadata_content(content_str):
                logger.debug(f"Skipping metadata section: {section_title}")
                continue

            filtered[section_title] = content_str

        return filtered

    def _is_metadata_section(self, section_title: str) -> bool:
        """Check if a section title indicates metadata content."""
        title_lower = section_title.lower().strip()

        metadata_indicators = [
            "content",
            "header",
            "authors",
            "author",
            "affiliation",
            "email",
            "arxiv",
            "preprint",
            "submitted",
            "received",
            "accepted",
        ]

        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        return any(indicator in title_lower and len(title_lower) < 20 for indicator in metadata_indicators)

    def _is_duplicate_abstract(self, content: str, abstract: str, abstract_words: set) -> bool:
        """Check if section content is a duplicate of the abstract."""
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        content_words = set(content_lower.split())
        if len(abstract_words) > 10:
            overlap = len(abstract_words.intersection(content_words))
            overlap_ratio = overlap / len(abstract_words)
            if overlap_ratio > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Check if short content contains only metadata patterns."""
        content_lower = content.lower()

        metadata_patterns = [
            "@",
            "arxiv:",
            "university",
            "institute",
            "department",
            "college",
            "gmail.com",
            "edu",
            "ac.uk",
            "preprint",
        ]

        word_count = len(content.split())
        if word_count < 30:
            metadata_word_count = sum(1 for pattern in metadata_patterns if pattern in content_lower)
            if metadata_word_count >= 2:
                return True

        return False

    # =================================================================
    # FR-5: Section-Based Chunking
    # =================================================================

    def _chunk_by_sections(
        self,
        title: str,
        abstract: str,
        arxiv_id: str,
        paper_id: str,
        sections: dict[str, str] | str | list,
    ) -> list[TextChunk]:
        """Implement hybrid section-based chunking strategy."""
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(f"No meaningful sections found after filtering for {arxiv_id}")
            return []

        header = f"{title}\n\nAbstract: {abstract}\n\n"

        chunks: list[TextChunk] = []
        small_sections: list[tuple] = []
        section_items = list(sections_dict.items())

        for i, (section_title, section_content) in enumerate(section_items):
            content_str = str(section_content) if section_content else ""
            section_words = len(content_str.split())

            if section_words < 100:
                small_sections.append((section_title, content_str, section_words))

                is_last = i == len(section_items) - 1
                next_is_large = not is_last and len(str(section_items[i + 1][1]).split()) >= 100

                if is_last or next_is_large:
                    chunks.extend(self._create_combined_chunk(header, small_sections, chunks, arxiv_id, paper_id))
                    small_sections = []

            elif 100 <= section_words <= 800:
                chunk_text = f"{header}Section: {section_title}\n\n{content_str}"
                chunk = self._create_section_chunk(chunk_text, section_title, len(chunks), arxiv_id, paper_id)
                chunks.append(chunk)

            else:
                section_text = f"Section: {section_title}\n\n{content_str}"
                full_section_text = f"{header}{section_text}"
                section_chunks = self._split_large_section(
                    full_section_text, header, section_title, len(chunks), arxiv_id, paper_id
                )
                chunks.extend(section_chunks)

        return chunks

    def _create_combined_chunk(
        self,
        header: str,
        small_sections: list[tuple],
        existing_chunks: list[TextChunk],
        arxiv_id: str,
        paper_id: str,
    ) -> list[TextChunk]:
        """Combine multiple small sections into a single chunk."""
        if not small_sections:
            return []

        combined_content = []
        total_words = 0

        for section_title, content, word_count in small_sections:
            combined_content.append(f"Section: {section_title}\n\n{content}")
            total_words += word_count

        combined_text = f"{header}{chr(10).join(chr(10).join(['']) + part for part in combined_content).strip()}"

        if total_words + len(header.split()) < 200 and existing_chunks:
            prev_chunk = existing_chunks[-1]
            merged_text = f"{prev_chunk.text}\n\n{chr(10).join(chr(10).join(['']) + part for part in combined_content).strip()}"

            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev_chunk.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev_chunk.metadata.section_title} + Combined",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []

        sections_titles = [title for title, _, _ in small_sections]
        combined_title = " + ".join(sections_titles[:3])
        if len(sections_titles) > 3:
            combined_title += f" + {len(sections_titles) - 3} more"

        chunk = self._create_section_chunk(combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id)
        return [chunk]

    def _create_section_chunk(
        self,
        chunk_text: str,
        section_title: str,
        chunk_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> TextChunk:
        """Create a single TextChunk for a section-based chunk."""
        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=section_title,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _split_large_section(
        self,
        full_section_text: str,
        header: str,
        section_title: str,
        base_chunk_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> list[TextChunk]:
        """Split large sections (>800 words) using word-based chunking."""
        section_only = full_section_text[len(header) :]
        traditional_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        enhanced_chunks = []
        for i, chunk in enumerate(traditional_chunks):
            enhanced_text = f"{header}{chunk.text}"
            enhanced_chunk = TextChunk(
                text=enhanced_text,
                metadata=ChunkMetadata(
                    chunk_index=base_chunk_index + i,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char + len(header),
                    word_count=len(enhanced_text.split()),
                    overlap_with_previous=chunk.metadata.overlap_with_previous,
                    overlap_with_next=chunk.metadata.overlap_with_next,
                    section_title=f"{section_title} (Part {i + 1})",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    # =================================================================
    # Utilities
    # =================================================================

    def _split_into_words(self, text: str) -> list[str]:
        """Split text into words using whitespace boundaries."""
        return re.findall(r"\S+", text)

    def _reconstruct_text(self, words: list[str]) -> str:
        """Reconstruct text from a list of words."""
        return " ".join(words)
