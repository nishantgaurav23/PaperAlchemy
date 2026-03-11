"""Tests for the section-aware text chunker (S4.2).

TDD: These tests are written FIRST, before implementation.
"""

from __future__ import annotations

import pytest
from src.schemas.indexing import ChunkMetadata, TextChunk
from src.schemas.pdf import Section
from src.services.indexing.text_chunker import TextChunker

# ---------------------------------------------------------------------------
# FR-1: Data Models
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_chunk_metadata_model(self):
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=100,
            word_count=50,
            overlap_with_previous=0,
            overlap_with_next=10,
        )
        assert meta.chunk_index == 0
        assert meta.section_title is None  # default

    def test_chunk_metadata_with_section(self):
        meta = ChunkMetadata(
            chunk_index=1,
            start_char=100,
            end_char=500,
            word_count=200,
            overlap_with_previous=10,
            overlap_with_next=10,
            section_title="Introduction",
        )
        assert meta.section_title == "Introduction"


class TestTextChunk:
    def test_text_chunk_model(self):
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=100,
            word_count=50,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        chunk = TextChunk(
            text="Hello world",
            metadata=meta,
            arxiv_id="2401.12345",
            paper_id="uuid-123",
        )
        assert chunk.text == "Hello world"
        assert chunk.arxiv_id == "2401.12345"
        assert chunk.metadata.chunk_index == 0


# ---------------------------------------------------------------------------
# FR-7: Configuration Validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_valid_config(self):
        chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)
        assert chunker.chunk_size == 600
        assert chunker.overlap_size == 100
        assert chunker.min_chunk_size == 100

    def test_init_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="Overlap size must be less than chunk size"):
            TextChunker(chunk_size=100, overlap_size=100)

    def test_init_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="Overlap size must be less than chunk size"):
            TextChunker(chunk_size=100, overlap_size=200)


# ---------------------------------------------------------------------------
# FR-2: Word-Based Chunking
# ---------------------------------------------------------------------------


def _make_text(word_count: int) -> str:
    """Generate text with exactly `word_count` words."""
    return " ".join(f"word{i}" for i in range(word_count))


class TestChunkText:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)

    def test_chunk_text_basic(self):
        """600-word text → 1 chunk."""
        text = _make_text(600)
        chunks = self.chunker.chunk_text(text, "2401.00001", "pid-1")
        assert len(chunks) == 1
        assert chunks[0].metadata.chunk_index == 0
        assert chunks[0].metadata.word_count == 600
        assert chunks[0].arxiv_id == "2401.00001"
        assert chunks[0].paper_id == "pid-1"

    def test_chunk_text_with_overlap(self):
        """1100-word text → 2 overlapping chunks (600 + advance 500 → second starts at 500)."""
        text = _make_text(1100)
        chunks = self.chunker.chunk_text(text, "2401.00002", "pid-2")
        assert len(chunks) == 2
        # First chunk: words 0-599
        assert chunks[0].metadata.word_count == 600
        assert chunks[0].metadata.overlap_with_next == 100
        assert chunks[0].metadata.overlap_with_previous == 0
        # Second chunk: words 500-1099
        assert chunks[1].metadata.chunk_index == 1
        assert chunks[1].metadata.overlap_with_previous == 100

    def test_chunk_text_three_chunks(self):
        """1600-word text → 3 chunks with overlap."""
        text = _make_text(1600)
        chunks = self.chunker.chunk_text(text, "2401.00003", "pid-3")
        assert len(chunks) == 3

    def test_chunk_text_empty(self):
        chunks = self.chunker.chunk_text("", "2401.00004", "pid-4")
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        chunks = self.chunker.chunk_text("   \n\t  ", "2401.00005", "pid-5")
        assert chunks == []

    def test_chunk_text_below_minimum(self):
        """Short text below min_chunk_size → single chunk."""
        text = _make_text(50)
        chunks = self.chunker.chunk_text(text, "2401.00006", "pid-6")
        assert len(chunks) == 1
        assert chunks[0].metadata.word_count == 50

    def test_chunk_text_char_offsets(self):
        """Verify start_char and end_char are reasonable."""
        text = _make_text(600)
        chunks = self.chunker.chunk_text(text, "2401.00007", "pid-7")
        assert chunks[0].metadata.start_char == 0
        assert chunks[0].metadata.end_char > 0

    def test_chunk_text_multi_chunk_offsets(self):
        """With multiple chunks, second chunk start_char > 0."""
        text = _make_text(1100)
        chunks = self.chunker.chunk_text(text, "2401.00008", "pid-8")
        assert chunks[1].metadata.start_char > 0
        assert chunks[1].metadata.start_char < chunks[1].metadata.end_char


# ---------------------------------------------------------------------------
# FR-3: Section Parsing
# ---------------------------------------------------------------------------


class TestParseSections:
    def setup_method(self):
        self.chunker = TextChunker()

    def test_parse_sections_dict(self):
        sections = {"Introduction": "Some text", "Methods": "More text"}
        result = self.chunker._parse_sections(sections)
        assert result == {"Introduction": "Some text", "Methods": "More text"}

    def test_parse_sections_list_of_dicts(self):
        sections = [
            {"title": "Introduction", "content": "Some text"},
            {"title": "Methods", "content": "More text"},
        ]
        result = self.chunker._parse_sections(sections)
        assert result == {"Introduction": "Some text", "Methods": "More text"}

    def test_parse_sections_section_objects(self):
        """Handle Section objects from PDFContent."""
        sections = [
            Section(title="Introduction", content="Intro text", level=1),
            Section(title="Methods", content="Method text", level=2),
        ]
        result = self.chunker._parse_sections(sections)
        assert result == {"Introduction": "Intro text", "Methods": "Method text"}

    def test_parse_sections_json_string(self):
        import json

        sections = json.dumps({"Introduction": "Text here"})
        result = self.chunker._parse_sections(sections)
        assert result == {"Introduction": "Text here"}

    def test_parse_sections_json_list_string(self):
        import json

        sections = json.dumps(
            [
                {"title": "Intro", "content": "Text"},
            ]
        )
        result = self.chunker._parse_sections(sections)
        assert result == {"Intro": "Text"}

    def test_parse_sections_invalid_json(self):
        result = self.chunker._parse_sections("not json {{{")
        assert result == {}

    def test_parse_sections_none(self):
        result = self.chunker._parse_sections(None)
        assert result == {}

    def test_parse_sections_list_heading_key(self):
        """Handle 'heading' key as alias for 'title'."""
        sections = [{"heading": "Background", "text": "Background text"}]
        result = self.chunker._parse_sections(sections)
        assert result == {"Background": "Background text"}


# ---------------------------------------------------------------------------
# FR-4: Section Filtering
# ---------------------------------------------------------------------------


class TestFilterSections:
    def setup_method(self):
        self.chunker = TextChunker()

    def test_filter_removes_empty(self):
        sections = {"Introduction": "Good text", "Empty": ""}
        result = self.chunker._filter_sections(sections, "abstract text")
        assert "Empty" not in result
        assert "Introduction" in result

    def test_filter_removes_metadata_sections(self):
        sections = {
            "Authors": "John Doe, Jane Doe",
            "Introduction": "This paper presents...",
        }
        result = self.chunker._filter_sections(sections, "abstract")
        assert "Authors" not in result
        assert "Introduction" in result

    def test_filter_removes_abstract_duplicate_substring(self):
        abstract = "This paper presents a novel approach to text chunking."
        sections = {
            "Abstract": "This paper presents a novel approach to text chunking.",
            "Introduction": "We introduce our method.",
        }
        result = self.chunker._filter_sections(sections, abstract)
        assert "Abstract" not in result
        assert "Introduction" in result

    def test_filter_removes_abstract_duplicate_overlap(self):
        abstract = " ".join(f"word{i}" for i in range(20))
        # >80% word overlap
        section_text = " ".join(f"word{i}" for i in range(20)) + " extra"
        sections = {"Abstract Section": section_text, "Methods": "Different content here."}
        result = self.chunker._filter_sections(sections, abstract)
        assert "Abstract Section" not in result

    def test_filter_removes_metadata_content(self):
        """Short sections with metadata patterns (@ emails, university) are filtered."""
        sections = {
            "1.": "john@university.edu Department of CS",
            "Introduction": "This is the real content of the paper and it is long enough.",
        }
        result = self.chunker._filter_sections(sections, "abstract")
        assert "1." not in result

    def test_filter_all_removed_returns_empty(self):
        sections = {"Authors": "Author names", "": ""}
        result = self.chunker._filter_sections(sections, "abstract")
        assert result == {}


# ---------------------------------------------------------------------------
# FR-5: Section-Based Chunking
# ---------------------------------------------------------------------------


class TestChunkBySections:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)

    def test_medium_section_single_chunk(self):
        """Section with 200 words → single chunk with header."""
        sections = {"Introduction": _make_text(200)}
        chunks = self.chunker._chunk_by_sections("Paper Title", "Paper abstract text", "2401.10001", "pid-10", sections)
        assert len(chunks) == 1
        assert "Paper Title" in chunks[0].text
        assert "Abstract:" in chunks[0].text
        assert chunks[0].metadata.section_title == "Introduction"

    def test_small_sections_combined(self):
        """Multiple small sections (<100 words each) combined into one chunk."""
        sections = {
            "Acknowledgments": _make_text(30),
            "Funding": _make_text(40),
            "Data Availability": _make_text(20),
        }
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.10002", "pid-11", sections)
        # All small sections combined into ≤ 1-2 chunks
        assert len(chunks) >= 1

    def test_large_section_split(self):
        """Section with 1000 words → split into multiple sub-chunks."""
        sections = {"Methods": _make_text(1000)}
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.10003", "pid-12", sections)
        assert len(chunks) >= 2
        # Each sub-chunk should have part numbering
        assert "Part 1" in chunks[0].metadata.section_title
        assert "Part 2" in chunks[1].metadata.section_title

    def test_mixed_section_sizes(self):
        """Mix of small, medium, and large sections."""
        sections = {
            "Abstract Note": _make_text(20),
            "Introduction": _make_text(300),
            "Methods": _make_text(900),
            "Conclusion": _make_text(50),
        }
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.10004", "pid-13", sections)
        assert len(chunks) >= 3  # At least intro + methods (split) + conclusion group


# ---------------------------------------------------------------------------
# FR-6: Main Entry Point
# ---------------------------------------------------------------------------


class TestChunkPaper:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)

    def test_chunk_paper_with_sections(self):
        """Section-based chunking used when sections provided."""
        sections = [
            Section(title="Introduction", content=_make_text(300), level=1),
            Section(title="Methods", content=_make_text(200), level=1),
        ]
        chunks = self.chunker.chunk_paper(
            title="Test Paper",
            abstract="Test abstract",
            full_text=_make_text(500),
            arxiv_id="2401.20001",
            paper_id="pid-20",
            sections=sections,
        )
        assert len(chunks) >= 2
        # Should use section-based (check for header)
        assert "Test Paper" in chunks[0].text

    def test_chunk_paper_without_sections(self):
        """Word-based fallback when no sections."""
        chunks = self.chunker.chunk_paper(
            title="Test Paper",
            abstract="Test abstract",
            full_text=_make_text(600),
            arxiv_id="2401.20002",
            paper_id="pid-21",
            sections=None,
        )
        assert len(chunks) == 1
        # Word-based doesn't prepend header
        assert chunks[0].metadata.section_title is None

    def test_chunk_paper_empty_sections_fallback(self):
        """Empty sections list → fallback to word-based."""
        chunks = self.chunker.chunk_paper(
            title="Title",
            abstract="Abstract",
            full_text=_make_text(600),
            arxiv_id="2401.20003",
            paper_id="pid-22",
            sections=[],
        )
        assert len(chunks) >= 1

    def test_chunk_paper_section_failure_fallback(self):
        """Graceful fallback when section chunking fails."""
        # Pass sections that will cause _parse_sections to return empty
        # (all metadata sections that get filtered out)
        sections = {"Authors": "Author list"}
        chunks = self.chunker.chunk_paper(
            title="Title",
            abstract="Abstract",
            full_text=_make_text(600),
            arxiv_id="2401.20004",
            paper_id="pid-23",
            sections=sections,
        )
        # Should fallback to word-based
        assert len(chunks) >= 1

    def test_chunk_paper_empty_text_no_sections(self):
        """No text, no sections → empty list."""
        chunks = self.chunker.chunk_paper(
            title="Title",
            abstract="Abstract",
            full_text="",
            arxiv_id="2401.20005",
            paper_id="pid-24",
            sections=None,
        )
        assert chunks == []


# ---------------------------------------------------------------------------
# Additional: Header & Metadata
# ---------------------------------------------------------------------------


class TestHeaderAndMetadata:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)

    def test_header_prepended(self):
        """Every section chunk starts with title + abstract."""
        sections = {"Introduction": _make_text(200)}
        chunks = self.chunker._chunk_by_sections("My Paper Title", "My abstract here", "2401.30001", "pid-30", sections)
        assert chunks[0].text.startswith("My Paper Title")
        assert "Abstract: My abstract here" in chunks[0].text

    def test_section_title_in_metadata(self):
        """Section title preserved in chunk metadata."""
        sections = {"Results": _make_text(200)}
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.30002", "pid-31", sections)
        assert chunks[0].metadata.section_title == "Results"

    def test_large_section_part_numbering(self):
        """Split sections have 'Part N' in section_title."""
        sections = {"Discussion": _make_text(1500)}
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.30003", "pid-32", sections)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert f"Part {i + 1}" in chunk.metadata.section_title

    def test_chunk_indices_sequential(self):
        """Chunk indices are sequential starting from 0."""
        sections = {
            "Introduction": _make_text(200),
            "Methods": _make_text(300),
            "Results": _make_text(200),
        }
        chunks = self.chunker._chunk_by_sections("Title", "Abstract", "2401.30004", "pid-33", sections)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
