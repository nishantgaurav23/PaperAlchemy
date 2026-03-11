"""Tests for citation enforcement (S5.5).

TDD: Tests written FIRST, implementation follows.
"""

from __future__ import annotations

import pytest
from src.services.rag.citation import (
    CitationResult,
    enforce_citations,
    format_source_list,
    parse_citations,
    stream_with_citations,
    validate_citations,
)
from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(index: int, arxiv_id: str = "", title: str = "", authors: list[str] | None = None) -> SourceReference:
    """Helper to create a SourceReference with defaults."""
    return SourceReference(
        index=index,
        arxiv_id=arxiv_id or f"230{index}.0000{index}",
        title=title or f"Paper {index}",
        authors=authors if authors is not None else [f"Author{index}A", f"Author{index}B"],
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id or f'230{index}.0000{index}'}",
        chunk_text=f"Chunk text for paper {index}",
        score=0.9 - index * 0.1,
    )


def _make_sources(n: int) -> list[SourceReference]:
    return [_make_source(i) for i in range(1, n + 1)]


def _make_rag_response(answer: str, n_sources: int = 3) -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=_make_sources(n_sources),
        query="test query",
        retrieval_metadata=RetrievalMetadata(),
        llm_metadata=LLMMetadata(),
    )


# ===========================================================================
# FR-1: Citation Parser
# ===========================================================================


class TestParseCitations:
    def test_basic(self):
        text = "Transformers [1] are great. BERT [2] is also good. GPT [3] too."
        assert parse_citations(text) == [1, 2, 3]

    def test_empty(self):
        assert parse_citations("No citations here.") == []

    def test_duplicates(self):
        text = "Transformers [1] are used in [1] many applications [2]."
        assert parse_citations(text) == [1, 2]

    def test_invalid_indices(self):
        text = "Invalid refs [0] and [-1] and [abc] should be ignored. Valid [1]."
        assert parse_citations(text) == [1]

    def test_nested_brackets(self):
        text = "Sometimes LLMs produce [[1]] nested brackets [[2]]."
        assert parse_citations(text) == [1, 2]

    def test_markdown_links(self):
        text = "See [[1]](http://example.com) and [[2]](http://example.com)."
        assert parse_citations(text) == [1, 2]

    def test_range_style_ignored(self):
        text = "See [1-3] for details. Also [2]."
        # [1-3] is NOT a valid citation; only [2] is
        assert parse_citations(text) == [2]

    def test_large_indices(self):
        text = "Source [10] and [15] are relevant."
        assert parse_citations(text) == [10, 15]

    def test_empty_string(self):
        assert parse_citations("") == []


# ===========================================================================
# FR-2: Citation Validator
# ===========================================================================


class TestValidateCitations:
    def test_all_valid(self):
        sources = _make_sources(3)
        result = validate_citations([1, 2, 3], sources)
        assert result.valid_citations == [1, 2, 3]
        assert result.invalid_citations == []
        assert result.uncited_sources == []
        assert result.citation_coverage == 1.0
        assert result.is_valid is True

    def test_hallucinated(self):
        sources = _make_sources(3)
        result = validate_citations([1, 2, 5], sources)
        assert result.valid_citations == [1, 2]
        assert result.invalid_citations == [5]
        assert result.uncited_sources == [3]
        assert result.is_valid is False  # has invalid citations

    def test_uncited_sources(self):
        sources = _make_sources(3)
        result = validate_citations([1], sources)
        assert result.uncited_sources == [2, 3]
        assert result.citation_coverage == pytest.approx(1 / 3, abs=0.01)

    def test_coverage(self):
        sources = _make_sources(3)
        result = validate_citations([1, 2], sources)
        assert result.citation_coverage == pytest.approx(2 / 3, abs=0.01)

    def test_empty_sources(self):
        result = validate_citations([1, 2], [])
        assert result.valid_citations == []
        assert result.invalid_citations == [1, 2]
        assert result.is_valid is False

    def test_no_citations_no_sources(self):
        result = validate_citations([], [])
        assert result.is_valid is True
        assert result.citation_coverage == 0.0

    def test_no_citations_with_sources(self):
        sources = _make_sources(2)
        result = validate_citations([], sources)
        assert result.is_valid is False
        assert result.uncited_sources == [1, 2]
        assert result.citation_coverage == 0.0


# ===========================================================================
# FR-3: Source List Formatter
# ===========================================================================


class TestFormatSourceList:
    def test_basic(self):
        sources = [_make_source(1, arxiv_id="2301.00001", title="Attention Is All You Need", authors=["Vaswani", "Shazeer"])]
        result = format_source_list(sources)
        assert "**Sources:**" in result
        assert "[Attention Is All You Need]" in result
        assert "https://arxiv.org/abs/2301.00001" in result
        assert "Vaswani, Shazeer" in result

    def test_many_authors(self):
        sources = [_make_source(1, authors=["Author1", "Author2", "Author3", "Author4"])]
        result = format_source_list(sources)
        assert "Author1 et al." in result
        assert "Author2" not in result

    def test_few_authors(self):
        sources = [_make_source(1, authors=["Author1", "Author2", "Author3"])]
        result = format_source_list(sources)
        assert "Author1, Author2, Author3" in result

    def test_one_author(self):
        sources = [_make_source(1, authors=["SingleAuthor"])]
        result = format_source_list(sources)
        assert "SingleAuthor" in result

    def test_cited_only(self):
        sources = _make_sources(3)
        result = format_source_list(sources, cited_indices=[1, 3])
        # Should only include sources 1 and 3, not 2
        assert "Paper 1" in result
        assert "Paper 3" in result
        assert "Paper 2" not in result

    def test_missing_title(self):
        sources = [_make_source(1, title="", arxiv_id="2301.00001")]
        result = format_source_list(sources)
        # Falls back to arxiv_id
        assert "2301.00001" in result

    def test_missing_authors(self):
        sources = [_make_source(1, authors=[])]
        result = format_source_list(sources)
        # Should still produce a valid line without crashing
        assert "Paper 1" in result

    def test_empty(self):
        assert format_source_list([]) == ""

    def test_year_extraction(self):
        sources = [_make_source(1, arxiv_id="2301.00001", title="Test Paper")]
        result = format_source_list(sources)
        assert "2023" in result

    def test_multiple_sources_numbered(self):
        sources = _make_sources(3)
        result = format_source_list(sources)
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ===========================================================================
# FR-4: Citation Enforcer
# ===========================================================================


class TestEnforceCitations:
    def test_full_flow(self):
        answer = "Transformers [1] use attention [2]. BERT [3] builds on this."
        response = _make_rag_response(answer, n_sources=3)
        result = enforce_citations(response)

        assert isinstance(result, CitationResult)
        assert "**Sources:**" in result.formatted_answer
        assert result.validation.is_valid is True
        assert result.validation.citation_coverage == 1.0
        assert result.sources_markdown != ""

    def test_strips_existing_sources(self):
        answer = "Transformers [1] are great.\n\n**Sources:**\n1. Some LLM-generated source\n2. Another one\n"
        response = _make_rag_response(answer, n_sources=2)
        result = enforce_citations(response)

        # The LLM-generated sources section should be replaced
        # There should be exactly one "**Sources:**" section
        assert result.formatted_answer.count("**Sources:**") == 1
        # The standardized source list should use our format
        assert "Paper 1" in result.formatted_answer

    def test_no_citations_in_text(self):
        answer = "This answer has no inline citations at all."
        response = _make_rag_response(answer, n_sources=2)
        result = enforce_citations(response)

        # Should still append source list
        assert "**Sources:**" in result.formatted_answer
        assert result.validation.is_valid is False

    def test_empty_answer(self):
        response = _make_rag_response("", n_sources=0)
        result = enforce_citations(response)
        assert result.formatted_answer == ""
        assert result.sources_markdown == ""

    def test_answer_text_preserved(self):
        answer = "Transformers [1] use attention mechanisms."
        response = _make_rag_response(answer, n_sources=2)
        result = enforce_citations(response)

        # Original answer text should be in the formatted answer
        assert "Transformers [1] use attention mechanisms." in result.formatted_answer

    def test_strips_sources_variations(self):
        """Test stripping various forms of LLM-generated source sections."""
        answer = "Some answer [1].\n\nSources:\n1. Paper A\n"
        response = _make_rag_response(answer, n_sources=2)
        result = enforce_citations(response)
        assert result.formatted_answer.count("**Sources:**") == 1


# ===========================================================================
# FR-5: Streaming Citation Support
# ===========================================================================


class TestStreamWithCitations:
    @pytest.mark.asyncio
    async def test_yields_tokens_then_sources(self):
        sources = _make_sources(2)

        async def mock_stream():
            yield "Hello "
            yield "[1] "
            yield "world [2]."

        tokens = []
        async for token in stream_with_citations(mock_stream(), sources):
            tokens.append(token)

        full_text = "".join(tokens)
        assert "Hello" in full_text
        assert "**Sources:**" in full_text

    @pytest.mark.asyncio
    async def test_strips_llm_sources_from_stream(self):
        sources = _make_sources(2)

        async def mock_stream():
            yield "Answer [1].\n\n"
            yield "**Sources:**\n"
            yield "1. LLM source\n"

        tokens = []
        async for token in stream_with_citations(mock_stream(), sources):
            tokens.append(token)

        full_text = "".join(tokens)
        # LLM-generated sources should be stripped and replaced
        assert full_text.count("**Sources:**") == 1
        assert "LLM source" not in full_text
        assert "Paper 1" in full_text

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        sources = _make_sources(2)

        async def mock_stream():
            return
            yield  # noqa: RET504 - make it an async generator

        tokens = []
        async for token in stream_with_citations(mock_stream(), sources):
            tokens.append(token)

        # Empty stream yields nothing
        assert tokens == []

    @pytest.mark.asyncio
    async def test_validation_available(self):
        sources = _make_sources(2)

        async def mock_stream():
            yield "Result [1]."

        streamer = stream_with_citations(mock_stream(), sources)
        async for _ in streamer:
            pass

        # After consuming the stream, validation should be available
        assert hasattr(streamer, "validation")
