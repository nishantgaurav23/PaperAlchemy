"""
Section-Aware Text Chunker for academic papers.

Why it's needed:
    Academic papers have logical structure (Introduction, Methods, Results).
    Naive fixed-size chunking splits mid-sentence and loses section context.
    Section-aware chunking preserves the author's document structure, producing
    chunks that are more coherent and yield better search results.

What it does:
    - chunk_paper(): The main entry point. Tries section-based chunking first
      (using PDF parser output), falls back to word-based if sections aren't
      available.
    - chunk_text(): Traditional word-based chunking with overlap. Used as
      fallback and for splitting large sections (>800 words).
    - _chunk_by_sections(): Hybrid strategy that handles sections of different
      sizes: small (<100 words) get combined, medium (100-800) become single
      chunks, large (>800) get split with word-based chunking.

How it helps:
    - Better search relevance: chunks align with paper structure
    - No lost context: overlapping words at chunk boundaries
    - Header prepending: every chunk includes paper title + abstract
    - Metadata filtering: skip metadata sections (authors, affiliations)
    - Duplicate detection: skip sections that duplicate the abstract

Architecture:
    TextChunker is configured via ChunkingSettings (default: 600 words/chunk,
    100 word overlap). It produces List[TextChunk] which the HybridIndexer
    consumes for embedding and indexing.

    Pipeline flow:
        Paper (PostgreSQL) → TextChunker → List[TextChunk]
        → JinaClient.embed_passages() → List[embedding]
        → OpenSearch bulk index (chunk + embedding)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.schemas.indexing.models import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Service for chunking academic papers into overlapping segments.

    Uses a hybrid strategy: section-based when sections are available,
    word-based with overlap as fallback.

    Configuration (from ChunkingSettings):
        chunk_size: Target words per chunk (default 600)
        overlap_size: Words shared between adjacent chunks (default 100)
        min_chunk_size: Minimum words for a valid chunk (default 100)
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 100,
        min_chunk_size: int = 100
    ):
        """Initialize text chunker with size parameters.

        Args:
            chunk_size: Target number of words per chunk. 600 works well
                        for academic text — long enough for context, short
                        enough for precise retrieval.
            overlap_size: Number of overlapping words between adjacent chunks.
                          100 words ensures no information is lost at chunk
                          boundaries (a sentence split across chunks appears
                          in both).
            min_chunk_size: Minimum words for a chunk to be valid. Chunks
                            smaller than this are either merged with neighbors
                            or returned as a single small chunk.

        Raises:
            ValueError: If overlap_size >= chunk_size (would cause infinite loop).
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        # Overlap must be strictly less than chunk size, otherwise
        # the sliding window never advances and loops forever
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, "
            f"overlap_size={overlap_size}, min_chunk_size={min_chunk_size}"
        )

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words using whitespace boundaries.

        Uses regex to find all non-whitespace sequences, which handles
        multiple spaces, tabs, and newlines correctly.

        Args:
            text: Input text to split.

        Returns:
            List of words (non-whitespace tokens).
        """
        # \S+ matches one or more non-whitespace characters
        # This handles: multiple spaces, tabs, newlines, mixed whitespace
        words = re.findall(r"\S+", text)
        return words

    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct text from a list of words.

        Joins words with single spaces. Original whitespace formatting
        (indentation, double spaces) is normalized to single spaces.

        Args:
            words: List of words to join.

        Returns:
            Reconstructed text string.
        """
        return " ".join(words)

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict[str, str], str, list]] = None,
    ) -> List[TextChunk]:
        """Chunk a paper using hybrid section-based approach.

        Strategy (in priority order):
        1. If sections are available (from PDF parsing), use section-based
           chunking which preserves document structure.
        2. If section-based chunking fails or sections aren't available,
           fall back to traditional word-based chunking.

        Section-based strategy details:
        - Sections 100-800 words: Use as single chunk with title+abstract header
        - Sections <100 words: Combine with adjacent sections
        - Sections >800 words: Split using traditional word-based chunking

        Args:
            title: Paper title (prepended to every chunk as header).
            abstract: Paper abstract (prepended to every chunk as header).
            full_text: Full text content from PDF parsing.
            arxiv_id: ArXiv ID (e.g., "2401.12345") for chunk identification.
            paper_id: PostgreSQL primary key for database joins.
            sections: Parsed sections from PDF parser. Can be:
                      - Dict[str, str]: {"Introduction": "text...", ...}
                      - List[dict]: [{"title": "...", "content": "..."}, ...]
                      - str: JSON string of either format above
                      - None: sections not available

        Returns:
            List of TextChunk objects with metadata. Empty list if no text.
        """
        # Try section-based chunking first (better quality)
        if sections:
            try:
                section_chunks = self._chunk_by_sections(
                    title, abstract, arxiv_id, paper_id, sections
                )
                if section_chunks:
                    logger.info(
                        f"Created {len(section_chunks)} section-based "
                        f"chunks for {arxiv_id}"
                    )
                    return section_chunks
            except Exception as e:
                logger.warning(
                    f"Section-based chunking failed for {arxiv_id}: {e}"
                )

        # Fallback to traditional word-based chunking
        logger.info(f"Using traditional word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(
        self,
        text: str,
        arxiv_id: str,
        paper_id: str
    ) -> List[TextChunk]:
        """Chunk text into overlapping segments using sliding window.

        Uses a sliding window of chunk_size words, advancing by
        (chunk_size - overlap_size) words each step. This ensures
        every part of the text is covered, with overlap_size words
        shared between consecutive chunks.

        Args:
            text: Full text to chunk.
            arxiv_id: ArXiv ID for chunk identification.
            paper_id: Database ID for joins.

        Returns:
            List of TextChunk objects. Empty list if text is empty.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for paper {arxiv_id}")
            return []

        # Split into words for word-level chunking
        words = self._split_into_words(text)

        # Handle text shorter than minimum chunk size
        if len(words) < self.min_chunk_size:
            logger.warning(
                f"Text for paper {arxiv_id} has only {len(words)} words, "
                f"less than minimum {self.min_chunk_size}"
            )
            # Return single chunk if there's any text at all
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
        chunk_index = 0  # Sequential chunk counter
        current_position = 0  # Current word offset in the document

        # Sliding window loop
        while current_position < len(words):
            # Define chunk boundaries in word indices
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))

            # Extract words for this chunk
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            # Calculate approximate character offsets for source highlighting
            start_char = (
                len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            )
            end_char = len(" ".join(words[:chunk_end]))

            # Calculate overlap amounts for metadata
            overlap_with_previous = (
                min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            )
            overlap_with_next = (
                self.overlap_size if chunk_end < len(words) else 0
            )

            # Create TextChunk with full metadata
            chunk = TextChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_with_previous,
                    overlap_with_next=overlap_with_next,
                    section_title=None,  # No section info in fallback mode
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            chunks.append(chunk)

            # Advance window by (chunk_size - overlap_size) words
            # This creates the overlap between consecutive chunks
            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            # Stop if we've processed all words
            if chunk_end >= len(words):
                break

        logger.info(
            f"Chunked paper {arxiv_id}: {len(words)} words -> {len(chunks)} chunks"
        )
        return chunks

    # =================================================================
    # Section-Based Chunking (private methods)
    # =================================================================

    def _chunk_by_sections(
        self,
        title: str,
        abstract: str,
        arxiv_id: str,
        paper_id: str,
        sections: Union[Dict[str, str], str, list]
    ) -> List[TextChunk]:
        """Implement hybrid section-based chunking strategy.

        Each chunk gets a header (title + abstract) prepended so it can
        be understood in isolation during search. This is critical because
        search returns individual chunks, not full papers.

        Args:
            title: Paper title for header.
            abstract: Paper abstract for header.
            arxiv_id: ArXiv ID.
            paper_id: Database ID.
            sections: Raw sections data (dict, list, or JSON string).

        Returns:
            List of TextChunk objects, or empty list if sections can't be parsed.
        """
        # Parse sections into a uniform dict format
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        # Filter out metadata/duplicate sections
        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(
                f"No meaningful sections found after filtering for {arxiv_id}"
            )
            return []

        # Build header that gets prepended to every chunk
        # This ensures each chunk has enough context to be understood alone
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        # Process sections using size-based strategy
        chunks = []
        small_sections = []  # Buffer for combining small sections

        section_items = list(sections_dict.items())

        for i, (section_title, section_content) in enumerate(section_items):
            content_str = str(section_content) if section_content else ""
            section_words = len(content_str.split())

            if section_words < 100:
                # Small section: accumulate for combining
                small_sections.append(
                    (section_title, content_str, section_words)
                )

                # Flush accumulated small sections when:
                # 1. This is the last section, OR
                # 2. The next section is large enough to stand alone
                is_last = i == len(section_items) - 1
                next_is_large = (
                    not is_last
                    and len(str(section_items[i + 1][1]).split()) >= 100
                )

                if is_last or next_is_large:
                    chunks.extend(
                        self._create_combined_chunk(
                            header, small_sections, chunks,
                            arxiv_id, paper_id
                        )
                    )
                    small_sections = []

            elif 100 <= section_words <= 800:
                # Medium section: perfect size for a single chunk
                chunk_text = (
                    f"{header}Section: {section_title}\n\n{content_str}"
                )
                chunk = self._create_section_chunk(
                    chunk_text, section_title, len(chunks),
                    arxiv_id, paper_id
                )
                chunks.append(chunk)

            else:
                # Large section (>800 words): split with word-based chunking
                section_text = (
                    f"Section: {section_title}\n\n{content_str}"
                )
                full_section_text = f"{header}{section_text}"

                section_chunks = self._split_large_section(
                    full_section_text, header, section_title,
                    len(chunks), arxiv_id, paper_id
                )
                chunks.extend(section_chunks)

        return chunks

    def _parse_sections(
        self,
        sections: Union[Dict[str, str], str, list]
    ) -> Dict[str, str]:
        """Parse sections data into a uniform dictionary format.

        Handles all formats that PDF parsers might produce:
        - Dict[str, str]: already in the right format
        - List[dict]: list of {"title": "...", "content": "..."} objects
        - str: JSON-encoded version of either format

        Args:
            sections: Raw sections data from PDF parser.

        Returns:
            Dict mapping section titles to content strings.
            Empty dict if parsing fails.
        """
        if isinstance(sections, dict):
            return sections
        elif isinstance(sections, list):
            # Handle list of section objects
            result = {}
            for i, section in enumerate(sections):
                if isinstance(section, dict):
                    # Try common key names: title/heading, content/text
                    title = section.get(
                        "title",
                        section.get("heading", f"Section {i + 1}")
                    )
                    content = section.get(
                        "content",
                        section.get("text", "")
                    )
                    result[title] = content
                else:
                    result[f"Section {i + 1}"] = str(section)
            return result
        elif isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list):
                    result = {}
                    for i, section in enumerate(parsed):
                        if isinstance(section, dict):
                            title = section.get(
                                "title",
                                section.get("heading", f"Section {i + 1}")
                            )
                            content = section.get(
                                "content",
                                section.get("text", "")
                            )
                            result[title] = content
                        else:
                            result[f"Section {i + 1}"] = str(section)
                    return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections JSON")
        return {}

    def _filter_sections(
        self,
        sections_dict: Dict[str, str],
        abstract: str
    ) -> Dict[str, str]:
        """Filter out metadata sections and abstract duplicates.

        Removes sections that don't contain useful content:
        - Empty sections
        - Metadata sections (author lists, affiliations, emails)
        - Sections that are duplicates of the abstract

        Args:
            sections_dict: Dictionary of section title → content.
            abstract: Paper abstract for duplicate detection.

        Returns:
            Filtered dictionary with only meaningful sections.
        """
        filtered = {}
        # Pre-compute abstract word set for overlap checking
        abstract_words = set(abstract.lower().split())

        for section_title, section_content in sections_dict.items():
            content_str = str(section_content).strip()

            # Skip empty sections
            if not content_str:
                continue

            # Skip metadata/header sections (author names, affiliations)
            if self._is_metadata_section(section_title):
                continue

            # Skip sections that duplicate the abstract
            if self._is_duplicate_abstract(
                content_str, abstract, abstract_words
            ):
                logger.debug(
                    f"Skipping duplicate abstract section: {section_title}"
                )
                continue

            # Skip very short sections that are just metadata
            if (
                len(content_str.split()) < 20
                and self._is_metadata_content(content_str)
            ):
                logger.debug(
                    f"Skipping metadata section: {section_title}"
                )
                continue

            filtered[section_title] = content_str

        return filtered

    def _is_metadata_section(self, section_title: str) -> bool:
        """Check if a section title indicates metadata content.

        Academic papers often have sections for author info, affiliations,
        and submission details that shouldn't be chunked for search.

        Args:
            section_title: The section heading to check.

        Returns:
            True if the section likely contains metadata, not paper content.
        """
        title_lower = section_title.lower().strip()

        # Common metadata section titles to skip
        metadata_indicators = [
            "content", "header", "authors", "author",
            "affiliation", "email", "arxiv", "preprint",
            "submitted", "received", "accepted",
        ]

        # Skip exact matches or very short titles
        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        # Check for partial matches in short titles
        for indicator in metadata_indicators:
            if indicator in title_lower and len(title_lower) < 20:
                return True

        return False

    def _is_duplicate_abstract(
        self,
        content: str,
        abstract: str,
        abstract_words: set
    ) -> bool:
        """Check if section content is a duplicate of the abstract.

        Many PDFs include the abstract as a separate section. Since we
        already prepend the abstract as a header, indexing it again
        would waste space and bias search results.

        Uses two detection methods:
        1. Substring match (catches exact duplicates)
        2. Word overlap >80% (catches reformatted duplicates)

        Args:
            content: Section content to check.
            abstract: Paper abstract.
            abstract_words: Pre-computed set of abstract words (lowercase).

        Returns:
            True if the content is likely a duplicate of the abstract.
        """
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Method 1: Direct substring match
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Method 2: Word overlap ratio
        content_words = set(content_lower.split())
        if len(abstract_words) > 10:  # Only check substantial abstracts
            overlap = len(abstract_words.intersection(content_words))
            overlap_ratio = overlap / len(abstract_words)
            if overlap_ratio > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Check if short content contains only metadata patterns.

        Catches sections that have titles like "1." but contain only
        author emails, institutional affiliations, or arXiv identifiers.

        Args:
            content: Section content to check.

        Returns:
            True if content is mostly metadata.
        """
        content_lower = content.lower()

        # Patterns that indicate metadata, not paper content
        metadata_patterns = [
            "@", "arxiv:", "university", "institute",
            "department", "college", "gmail.com", "edu",
            "ac.uk", "preprint",
        ]

        word_count = len(content.split())
        if word_count < 30:  # Only check short content
            # Count how many metadata patterns appear
            metadata_word_count = sum(
                1 for pattern in metadata_patterns
                if pattern in content_lower
            )
            if metadata_word_count >= 2:
                return True

        return False

    def _create_combined_chunk(
        self,
        header: str,
        small_sections: List,
        existing_chunks: List,
        arxiv_id: str,
        paper_id: str
    ) -> List[TextChunk]:
        """Combine multiple small sections into a single chunk.

        Small sections (<100 words) are too short for meaningful search.
        Combining them creates properly-sized chunks while preserving
        all section content.

        If the combined content is still too small (<200 words) and there's
        a previous chunk, merges into the previous chunk instead.

        Args:
            header: Title + abstract header to prepend.
            small_sections: List of (title, content, word_count) tuples.
            existing_chunks: Already-created chunks (for merging).
            arxiv_id: ArXiv ID.
            paper_id: Database ID.

        Returns:
            List containing 0 or 1 TextChunk objects.
        """
        if not small_sections:
            return []

        # Combine all small section texts
        combined_content = []
        total_words = 0

        for section_title, content, word_count in small_sections:
            combined_content.append(
                f"Section: {section_title}\n\n{content}"
            )
            total_words += word_count

        combined_text = f"{header}{'\\n\\n'.join(combined_content)}"

        # If still too small, merge with previous chunk
        if total_words + len(header.split()) < 200 and existing_chunks:
            prev_chunk = existing_chunks[-1]
            merged_text = (
                f"{prev_chunk.text}\\n\\n{'\\n\\n'.join(combined_content)}"
            )

            # Replace the previous chunk with the merged version
            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev_chunk.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=(
                        f"{prev_chunk.metadata.section_title} + Combined"
                    ),
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []  # Merged into existing, no new chunk

        # Create new chunk with combined content
        sections_titles = [title for title, _, _ in small_sections]
        combined_title = " + ".join(sections_titles[:3])
        if len(sections_titles) > 3:
            combined_title += f" + {len(sections_titles) - 3} more"

        chunk = self._create_section_chunk(
            combined_text, combined_title, len(existing_chunks),
            arxiv_id, paper_id
        )
        return [chunk]

    def _create_section_chunk(
        self,
        chunk_text: str,
        section_title: str,
        chunk_index: int,
        arxiv_id: str,
        paper_id: str
    ) -> TextChunk:
        """Create a single TextChunk for a section-based chunk.

        Args:
            chunk_text: Full text including header.
            section_title: Name of the section.
            chunk_index: Position in the chunk sequence.
            arxiv_id: ArXiv ID.
            paper_id: Database ID.

        Returns:
            TextChunk with section metadata.
        """
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
        paper_id: str
    ) -> List[TextChunk]:
        """Split large sections (>800 words) using word-based chunking.

        The section content (without header) is chunked using the traditional
        sliding window approach. Then the header is prepended to each
        resulting chunk so every chunk has full paper context.

        Args:
            full_section_text: Complete text including header.
            header: Title + abstract header (to prepend to each sub-chunk).
            section_title: Section name (appended with part number).
            base_chunk_index: Starting chunk index for this section.
            arxiv_id: ArXiv ID.
            paper_id: Database ID.

        Returns:
            List of TextChunk objects, one per sub-chunk.
        """
        # Remove header — we'll add it back to each sub-chunk
        section_only = full_section_text[len(header):]

        # Use traditional word-based chunking on section content
        traditional_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        # Add header to each sub-chunk and update metadata
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
                    # Mark as part N of the section
                    section_title=f"{section_title} (Part {i + 1})",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks
