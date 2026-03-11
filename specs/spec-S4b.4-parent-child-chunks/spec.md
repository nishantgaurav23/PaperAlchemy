# Spec S4b.4 -- Parent-Child Chunk Retrieval

## Overview
Implement a parent-child chunking strategy where papers are indexed as small child chunks (200 words) for precise retrieval, but at query time the parent section/chunk (600 words) is returned to provide richer context to the LLM. This two-tier approach improves retrieval precision (small chunks match better) while maintaining generation quality (large chunks give more context).

The existing `TextChunker` (S4.2) creates 600-word chunks. This spec adds a `ParentChildChunker` that:
1. Creates parent chunks (600 words) — same as current chunking
2. Splits each parent into smaller child chunks (200 words) with overlap
3. Links children to parents via `parent_chunk_id`
4. At retrieval time, given child chunk IDs, expands to parent chunks (deduplicating)

## Dependencies
- **S4.2** (Text chunker) — `done` — provides `TextChunker`, `TextChunk`, `ChunkMetadata`

## Target Location
- `src/services/indexing/parent_child.py`

## Functional Requirements

### FR-1: ParentChildChunker Initialization
- **What**: Initialize chunker with configurable parent/child sizes
- **Inputs**: `parent_chunk_size` (default 600), `child_chunk_size` (default 200), `child_overlap` (default 50), `min_chunk_size` (default 50)
- **Outputs**: Configured `ParentChildChunker` instance
- **Edge cases**: child_chunk_size >= parent_chunk_size raises ValueError, child_overlap >= child_chunk_size raises ValueError

### FR-2: Create Parent-Child Chunks
- **What**: Given a paper's text (or sections), produce parent chunks and their child sub-chunks. Each child stores a reference to its parent's index.
- **Inputs**: `title`, `abstract`, `full_text`, `arxiv_id`, `paper_id`, `sections` (optional)
- **Outputs**: `ParentChildResult` containing `parents: list[TextChunk]` and `children: list[ChildChunk]`
- **Edge cases**: Empty text returns empty result. Text shorter than child_chunk_size produces one parent with one child.

### FR-3: ChildChunk Schema
- **What**: A `ChildChunk` extends the concept of `TextChunk` with a `parent_chunk_index` field linking to the parent.
- **Inputs**: text, metadata, arxiv_id, paper_id, parent_chunk_index
- **Outputs**: Pydantic model suitable for indexing

### FR-4: Split Parent into Children
- **What**: Given a parent chunk's text, split it into overlapping child chunks (200 words, 50 overlap).
- **Inputs**: Parent `TextChunk`
- **Outputs**: List of `ChildChunk` objects, each referencing the parent
- **Edge cases**: Parent with fewer words than child_chunk_size → single child. Empty parent text → empty list.

### FR-5: Parent Expansion (Retrieval-Time)
- **What**: Given a list of child chunk search results (each with `parent_chunk_id` or parent reference), retrieve and return the unique parent chunks. This is the key retrieval-time operation.
- **Inputs**: List of child search result dicts (from OpenSearch), `OpenSearchClient` instance
- **Outputs**: List of unique parent chunk dicts, deduplicated, preserving best score per parent
- **Edge cases**: Children from same parent → single parent returned. Missing parent_chunk_id → return child as-is. Empty input → empty output.

### FR-6: Prepare Children for Indexing
- **What**: Convert `ChildChunk` objects into OpenSearch-ready dicts with proper field mapping (chunk_text, parent_chunk_id, chunk_index, etc.)
- **Inputs**: List of `ChildChunk` objects, parent chunks for reference
- **Outputs**: List of dicts ready for `OpenSearchClient.bulk_index_chunks()`
- **Edge cases**: Ensure parent_chunk_id is a stable, deterministic identifier (e.g., `{arxiv_id}_parent_{chunk_index}`)

## Tangible Outcomes
- [ ] `ParentChildChunker` class in `src/services/indexing/parent_child.py`
- [ ] `ChildChunk` and `ParentChildResult` schemas in `src/schemas/indexing.py`
- [ ] Parent chunks (600w) split into child chunks (200w, 50 overlap)
- [ ] Each child references its parent via `parent_chunk_index`
- [ ] `expand_to_parents()` deduplicates and returns unique parent chunks from child results
- [ ] `prepare_for_indexing()` produces OpenSearch-ready dicts with `parent_chunk_id` field
- [ ] All public methods tested with pytest
- [ ] Notebook: `notebooks/specs/S4b.4_parent_child.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_init_valid_config`: Valid init with default and custom sizes
2. `test_init_invalid_child_size`: child_chunk_size >= parent_chunk_size raises ValueError
3. `test_init_invalid_overlap`: child_overlap >= child_chunk_size raises ValueError
4. `test_create_parent_child_chunks`: Full paper → parents + children with correct links
5. `test_create_parent_child_empty_text`: Empty text → empty result
6. `test_create_parent_child_short_text`: Short text → single parent, single child
7. `test_split_parent_into_children`: Single parent → correct number of children with overlap
8. `test_split_parent_short`: Parent shorter than child_chunk_size → one child
9. `test_child_references_parent`: Each child's parent_chunk_index matches parent
10. `test_expand_to_parents_basic`: Child results → unique parent chunks
11. `test_expand_to_parents_dedup`: Multiple children from same parent → one parent
12. `test_expand_to_parents_missing_id`: Children without parent_chunk_id returned as-is
13. `test_expand_to_parents_empty`: Empty input → empty output
14. `test_prepare_for_indexing`: Children → OpenSearch-ready dicts with parent_chunk_id
15. `test_section_based_parent_child`: Section-aware chunking produces parent-child hierarchy

### Mocking Strategy
- No external services needed for chunking logic (pure computation)
- Mock `OpenSearchClient` for `expand_to_parents()` tests (mock `get_chunks_by_paper`)
- Use fixtures with realistic paper text samples

### Coverage
- All public functions tested
- Edge cases: empty text, short text, single section, many sections
- Error paths: invalid config, missing fields
