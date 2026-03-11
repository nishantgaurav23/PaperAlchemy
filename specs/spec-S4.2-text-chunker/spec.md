# Spec S4.2 -- Text Chunker (Section-Aware)

## Overview
Section-aware text chunking service for academic papers. Splits parsed PDF content into overlapping chunks suitable for embedding and indexing in OpenSearch. Uses a hybrid strategy: section-based chunking when sections are available (from Docling PDF parser), falling back to word-based sliding window. Each chunk preserves metadata (position, section title, word count, overlap) for downstream indexing and retrieval.

Key design:
- 600-word target chunks, 100-word overlap, 100-word minimum
- Section-based: small sections (<100 words) are combined, medium (100-800) become single chunks, large (>800) are split with word-based chunking
- Header prepending: every chunk includes paper title + abstract for standalone context
- Metadata filtering: skip author/affiliation sections, abstract duplicates
- Produces `List[TextChunk]` consumed by the embedding + indexing pipeline

## Dependencies
- **S3.3** (PDF Parser) — provides `PDFContent` with `sections: list[Section]` that drives section-aware chunking

## Target Location
- `src/services/indexing/text_chunker.py` — main `TextChunker` class
- `src/services/indexing/__init__.py` — public API exports
- `src/schemas/indexing.py` — `TextChunk` and `ChunkMetadata` Pydantic models

## Functional Requirements

### FR-1: Chunk Metadata & Data Models
- **What**: Pydantic models for chunk output
- **Inputs**: Chunk text, positional metadata, paper identifiers
- **Outputs**: `TextChunk` and `ChunkMetadata` models
- **Details**:
  - `ChunkMetadata`: chunk_index, start_char, end_char, word_count, overlap_with_previous, overlap_with_next, section_title (optional)
  - `TextChunk`: text, metadata (ChunkMetadata), arxiv_id, paper_id
- **Edge cases**: Missing section_title (None for word-based chunks)

### FR-2: Word-Based Chunking (Fallback)
- **What**: Traditional sliding-window chunking with overlap
- **Inputs**: `text: str`, `arxiv_id: str`, `paper_id: str`
- **Outputs**: `list[TextChunk]` with sequential chunk_index, character offsets, overlap metadata
- **Details**:
  - Split text into words via regex (`\S+`)
  - Sliding window: chunk_size words, advance by (chunk_size - overlap_size)
  - Track start_char/end_char for source highlighting
  - Texts shorter than min_chunk_size return single chunk
- **Edge cases**: Empty text (return []), text shorter than min_chunk_size, very long text

### FR-3: Section Parsing
- **What**: Parse heterogeneous section input formats into uniform dict
- **Inputs**: Sections as `dict[str, str]`, `list[dict]`, `list[Section]`, `str` (JSON), or `None`
- **Outputs**: `dict[str, str]` mapping section title → content
- **Details**:
  - Handle `Section` objects from PDFContent (title/content attributes)
  - Handle list of dicts with title/heading and content/text keys
  - Handle JSON-encoded strings
- **Edge cases**: Invalid JSON, empty sections, mixed formats

### FR-4: Section Filtering
- **What**: Remove metadata sections and abstract duplicates
- **Inputs**: `dict[str, str]` sections, abstract text
- **Outputs**: Filtered dict with only meaningful content sections
- **Details**:
  - Skip empty sections
  - Skip metadata sections: authors, affiliations, emails, arxiv headers
  - Skip abstract duplicates (substring match or >80% word overlap)
  - Skip short sections with metadata patterns (@, university, .edu, etc.)
- **Edge cases**: All sections filtered out (return [])

### FR-5: Section-Based Chunking (Primary)
- **What**: Hybrid section-aware chunking strategy
- **Inputs**: title, abstract, arxiv_id, paper_id, sections
- **Outputs**: `list[TextChunk]` with section metadata
- **Details**:
  - Small sections (<100 words): accumulate and combine into single chunk
  - Medium sections (100-800 words): single chunk with header
  - Large sections (>800 words): split with word-based chunking, header prepended to each sub-chunk
  - Header format: `{title}\n\nAbstract: {abstract}\n\n`
  - Combined small sections: flush when next section is large or at end
  - Very small combined content (<200 words): merge into previous chunk
- **Edge cases**: No sections after filtering, all sections small, single massive section

### FR-6: Main Entry Point (chunk_paper)
- **What**: Orchestrates section-based vs word-based chunking
- **Inputs**: title, abstract, full_text, arxiv_id, paper_id, sections (optional)
- **Outputs**: `list[TextChunk]`
- **Details**:
  - If sections provided → try section-based chunking first
  - If section-based fails or no sections → fall back to word-based on full_text
  - Log chunking strategy and results
- **Edge cases**: sections=None, section chunking raises exception (catch and fallback)

### FR-7: Configuration
- **What**: TextChunker configured via ChunkingSettings from config.py
- **Inputs**: chunk_size (default 600), overlap_size (default 100), min_chunk_size (default 100)
- **Outputs**: Configured TextChunker instance
- **Details**:
  - Validate overlap_size < chunk_size (raise ValueError otherwise)
  - ChunkingSettings already exists in config.py
- **Edge cases**: Invalid config (overlap >= chunk_size)

## Tangible Outcomes
- [ ] `src/schemas/indexing.py` exists with `TextChunk` and `ChunkMetadata` models
- [ ] `src/services/indexing/text_chunker.py` exists with `TextChunker` class
- [ ] `src/services/indexing/__init__.py` exports `TextChunker`
- [ ] Word-based chunking: 600-word chunks, 100-word overlap, correct char offsets
- [ ] Section-based chunking: handles small/medium/large sections correctly
- [ ] Section parsing: handles dict, list, Section objects, JSON string
- [ ] Section filtering: removes metadata sections, abstract duplicates
- [ ] Header prepending: every section chunk includes title + abstract
- [ ] Fallback: gracefully falls back to word-based when sections unavailable/fail
- [ ] Validation: overlap >= chunk_size raises ValueError
- [ ] All tests pass: `pytest tests/unit/test_text_chunker.py -v`
- [ ] Lint passes: `ruff check src/services/indexing/ src/schemas/indexing.py`
- [ ] Notebook created: `notebooks/specs/S4.2_chunker.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_chunk_metadata_model`: Validates ChunkMetadata fields and defaults
2. `test_text_chunk_model`: Validates TextChunk with metadata
3. `test_init_valid_config`: TextChunker initializes with valid params
4. `test_init_overlap_exceeds_chunk_size`: Raises ValueError
5. `test_chunk_text_basic`: 600-word text → 1 chunk with correct metadata
6. `test_chunk_text_with_overlap`: 1100-word text → 2 overlapping chunks
7. `test_chunk_text_empty`: Empty text → empty list
8. `test_chunk_text_below_minimum`: Short text → single chunk
9. `test_chunk_text_char_offsets`: Correct start_char/end_char values
10. `test_parse_sections_dict`: Dict input parsed correctly
11. `test_parse_sections_list_of_dicts`: List[dict] input parsed correctly
12. `test_parse_sections_section_objects`: List[Section] input parsed correctly
13. `test_parse_sections_json_string`: JSON string parsed correctly
14. `test_parse_sections_invalid_json`: Invalid JSON returns empty dict
15. `test_filter_sections_removes_metadata`: Author/affiliation sections removed
16. `test_filter_sections_removes_abstract_duplicate`: Duplicate abstract filtered
17. `test_filter_sections_removes_empty`: Empty sections filtered
18. `test_chunk_by_sections_medium`: Medium section (100-800 words) → single chunk with header
19. `test_chunk_by_sections_small_combined`: Small sections (<100 words) combined
20. `test_chunk_by_sections_large_split`: Large section (>800 words) split into sub-chunks
21. `test_chunk_paper_with_sections`: Section-based chunking used when sections provided
22. `test_chunk_paper_without_sections`: Word-based fallback when no sections
23. `test_chunk_paper_section_failure_fallback`: Graceful fallback on section error
24. `test_header_prepended`: Every section chunk starts with title + abstract
25. `test_section_title_in_metadata`: Section title preserved in chunk metadata
26. `test_large_section_part_numbering`: Split sections have "Part 1", "Part 2" in section_title

### Mocking Strategy
- No external services to mock — TextChunker is pure computation
- Use `Section` objects from `src/schemas/pdf.py` for section input tests
- Generate test texts with known word counts for predictable chunk boundaries

### Coverage
- All public methods tested
- All section input formats tested
- Edge cases: empty text, short text, no sections, all filtered, huge sections
- Metadata accuracy: chunk_index, char offsets, word counts, overlap amounts
