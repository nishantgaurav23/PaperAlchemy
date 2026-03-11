# Spec S5.5 -- Citation Enforcement

## Overview

Citation enforcement is the core differentiator of PaperAlchemy as a research assistant. While S5.2 (RAG chain) already provides citation-enforcing **prompts** and numbered source references, S5.5 adds a **post-processing layer** that parses, validates, and formats LLM output to guarantee every response includes proper inline citations `[1], [2]` mapped to real papers with title, authors, and arXiv links.

This module sits between RAG chain output and the final response delivered to the user, ensuring citation quality regardless of LLM compliance.

## Dependencies

- **S5.2** (RAG chain) — Provides `RAGResponse` with `answer` text and `sources` list

## Target Location

- `src/services/rag/citation.py` — Citation parser, validator, formatter

## Functional Requirements

### FR-1: Citation Parser
- **What**: Extract all inline citation references `[N]` from LLM-generated text
- **Inputs**: `answer: str` — raw LLM output text
- **Outputs**: `list[int]` — sorted, deduplicated list of citation indices found in text
- **Edge cases**:
  - No citations found → returns empty list
  - Duplicate citations `[1] ... [1]` → deduplicated
  - Invalid references like `[0]`, `[-1]`, `[abc]` → ignored
  - Nested brackets `[[1]]` → extracts `[1]`
  - Citations in markdown links `[[1]](url)` → still extracted
  - Range-style `[1-3]` → NOT expanded (treated as literal, ignored)

### FR-2: Citation Validator
- **What**: Validate that inline citation indices map to actual sources in the source list
- **Inputs**: `cited_indices: list[int]`, `sources: list[SourceReference]`
- **Outputs**: `CitationValidation` with:
  - `valid_citations: list[int]` — indices that map to real sources
  - `invalid_citations: list[int]` — indices that don't map to any source (hallucinated refs)
  - `uncited_sources: list[int]` — source indices not referenced in text
  - `citation_coverage: float` — ratio of cited sources to total sources (0.0-1.0)
  - `is_valid: bool` — True if at least one valid citation AND no invalid citations
- **Edge cases**:
  - Empty sources list → all citations invalid
  - Empty cited indices → coverage = 0.0, is_valid = False (unless sources also empty, then True)
  - All sources cited → coverage = 1.0

### FR-3: Source List Formatter
- **What**: Generate a standardized markdown source list from SourceReference objects
- **Inputs**: `sources: list[SourceReference]`, `cited_only: bool = False`
- **Outputs**: Formatted markdown string:
  ```
  **Sources:**
  1. [Paper Title](https://arxiv.org/abs/XXXX.XXXXX) — Author1, Author2, Year
  2. [Paper Title](https://arxiv.org/abs/XXXX.XXXXX) — Author1 et al., Year
  ```
- **Formatting rules**:
  - If ≤3 authors: list all names
  - If >3 authors: "FirstAuthor et al."
  - Year extracted from arxiv_id (first 2 digits of ID → 20XX) or omitted if unavailable
  - If `cited_only=True`: only include sources that were actually cited in the text
  - If no sources: return empty string
- **Edge cases**:
  - Missing title → use arxiv_id as fallback
  - Missing authors → omit author section
  - Missing arxiv_url → use plain text (no link)

### FR-4: Citation Enforcer (Main Entry Point)
- **What**: Post-process a RAGResponse to ensure citation quality and append formatted source list
- **Inputs**: `response: RAGResponse`
- **Outputs**: `CitationResult` with:
  - `formatted_answer: str` — answer text with appended source list
  - `validation: CitationValidation` — citation quality report
  - `sources_markdown: str` — the formatted source list (also appended to answer)
- **Behavior**:
  1. Parse citations from `response.answer`
  2. Validate against `response.sources`
  3. Strip any existing "Sources:" section from LLM output (LLM sometimes generates its own)
  4. Format source list from `response.sources`
  5. Append formatted source list to cleaned answer
  6. Return result with validation metadata
- **Edge cases**:
  - LLM already included a "Sources:" section → strip it, replace with standardized format
  - LLM output has no citations at all → still append source list (sources were retrieved)
  - Empty answer → return empty answer with empty source list

### FR-5: Streaming Citation Support
- **What**: Process streaming text to extract and validate citations in real-time
- **Inputs**: `tokens: AsyncIterator[str]`, `sources: list[SourceReference]`
- **Outputs**: `AsyncIterator[str]` that yields tokens, then yields the formatted source list at the end
- **Behavior**:
  1. Pass through all tokens as-is (don't buffer the entire response)
  2. Accumulate text internally for citation tracking
  3. Detect and strip any LLM-generated "Sources:" section from the stream
  4. After stream completes, yield `\n\n` + formatted source list
  5. Make validation result available via a property after stream completes
- **Edge cases**:
  - Stream interrupted → return whatever was accumulated
  - Empty stream → yield nothing

## Tangible Outcomes

- [ ] `src/services/rag/citation.py` exists with all classes and functions
- [ ] `parse_citations(text)` extracts inline `[N]` references correctly
- [ ] `validate_citations(cited, sources)` produces accurate CitationValidation
- [ ] `format_source_list(sources)` generates proper markdown with arXiv links
- [ ] `enforce_citations(response)` returns CitationResult with cleaned + formatted answer
- [ ] `stream_with_citations(tokens, sources)` yields tokens then source list
- [ ] Existing LLM-generated "Sources:" sections are stripped and replaced
- [ ] Author formatting: ≤3 listed, >3 uses "et al."
- [ ] 100% of RAG responses go through citation enforcement
- [ ] All tests pass with mocked data (no external services)

## Test-Driven Requirements

### Tests to Write First

1. `test_parse_citations_basic`: Extract `[1]`, `[2]`, `[3]` from text
2. `test_parse_citations_empty`: No citations in text → empty list
3. `test_parse_citations_duplicates`: Duplicate `[1]` → deduplicated
4. `test_parse_citations_invalid_indices`: `[0]`, `[-1]`, `[abc]` ignored
5. `test_parse_citations_nested_brackets`: `[[1]]` → extracts 1
6. `test_validate_citations_all_valid`: All cited indices map to sources
7. `test_validate_citations_hallucinated`: Citation `[5]` with only 3 sources → invalid
8. `test_validate_citations_uncited_sources`: Source [3] not cited → in uncited list
9. `test_validate_citations_coverage`: 2 of 3 cited → coverage = 0.667
10. `test_validate_citations_empty_sources`: No sources → all invalid
11. `test_validate_citations_no_citations_no_sources`: Both empty → is_valid = True
12. `test_format_source_list_basic`: Proper markdown with title, URL, authors
13. `test_format_source_list_many_authors`: >3 authors → "et al."
14. `test_format_source_list_few_authors`: ≤3 authors → all listed
15. `test_format_source_list_cited_only`: Filter to only cited sources
16. `test_format_source_list_missing_fields`: Missing title/authors → graceful fallback
17. `test_format_source_list_empty`: No sources → empty string
18. `test_enforce_citations_full`: Complete flow: parse → validate → format → append
19. `test_enforce_citations_strips_existing_sources`: LLM-generated "Sources:" replaced
20. `test_enforce_citations_no_citations_in_text`: Still appends source list
21. `test_enforce_citations_empty_answer`: Returns empty result
22. `test_stream_with_citations`: Yields tokens then source list
23. `test_stream_with_citations_strips_sources`: Detects and removes LLM source section from stream

### Mocking Strategy

- No external services needed — all inputs are in-memory data structures
- Use `SourceReference` fixtures with known arxiv_ids, titles, authors
- Use `RAGResponse` fixtures with known answer text containing `[1]`, `[2]` citations
- For streaming tests: use `async def` generators yielding token strings

### Coverage

- All public functions tested
- Edge cases: empty inputs, missing fields, hallucinated citations
- Integration with RAGResponse data model verified
- Author formatting rules verified
- Streaming path verified
