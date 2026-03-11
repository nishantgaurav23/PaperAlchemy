"""Tests for the RAG chain service (S5.2).

TDD: These tests are written BEFORE implementation.
All external services (LLM, retrieval pipeline) are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.exceptions import LLMServiceError
from src.schemas.api.search import SearchHit
from src.services.llm.provider import LLMResponse, UsageMetadata
from src.services.retrieval.pipeline import RetrievalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_search_hits(count: int = 3) -> list[SearchHit]:
    """Create test SearchHit objects with realistic metadata."""
    hits = []
    for i in range(1, count + 1):
        hits.append(
            SearchHit(
                arxiv_id=f"2301.0000{i}",
                title=f"Test Paper {i}: A Study on Topic {i}",
                authors=[f"Author {i}A", f"Author {i}B"],
                abstract=f"Abstract for paper {i}.",
                pdf_url=f"https://arxiv.org/pdf/2301.0000{i}",
                score=1.0 - (i * 0.1),
                chunk_text=f"This is the relevant text from paper {i} about the research topic.",
                chunk_id=f"chunk-{i}",
                section_title=f"Section {i}",
            )
        )
    return hits


def _make_retrieval_result(hits: list[SearchHit] | None = None) -> RetrievalResult:
    """Create a test RetrievalResult."""
    if hits is None:
        hits = _make_search_hits()
    return RetrievalResult(
        results=hits,
        query="What are transformers?",
        expanded_queries=["transformer architecture", "self-attention mechanisms"],
        hypothetical_document="Transformers are a neural network architecture...",
        stages_executed=["multi_query", "hyde", "hybrid_search", "rerank"],
        total_candidates=20,
        timings={"multi_query": 0.5, "hyde": 0.3, "hybrid_search": 0.2, "rerank": 0.1},
    )


def _make_llm_response(text: str | None = None) -> LLMResponse:
    """Create a test LLMResponse with citation-formatted text."""
    if text is None:
        text = (
            "Transformers are a neural network architecture based on self-attention [1]. "
            "They have been widely adopted in NLP tasks [2] and extended to vision [3].\n\n"
            "**Sources:**\n"
            "1. [Test Paper 1: A Study on Topic 1](https://arxiv.org/abs/2301.00001) — Author 1A, Author 1B, 2023\n"
            "2. [Test Paper 2: A Study on Topic 2](https://arxiv.org/abs/2301.00002) — Author 2A, Author 2B, 2023\n"
            "3. [Test Paper 3: A Study on Topic 3](https://arxiv.org/abs/2301.00003) — Author 3A, Author 3B, 2023"
        )
    return LLMResponse(
        text=text,
        model="gemini-2.0-flash",
        provider="gemini",
        usage=UsageMetadata(prompt_tokens=500, completion_tokens=200, total_tokens=700, latency_ms=1200.0),
    )


@pytest.fixture()
def mock_llm_provider():
    """Mock LLM provider that returns citation-formatted responses."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=_make_llm_response())

    async def _stream(*args, **kwargs):
        for token in ["Transformers ", "are ", "based on ", "self-attention [1]."]:
            yield token

    provider.generate_stream = MagicMock(return_value=_stream())
    return provider


@pytest.fixture()
def mock_retrieval_pipeline():
    """Mock retrieval pipeline that returns pre-built results."""
    pipeline = AsyncMock()
    pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result())
    return pipeline


@pytest.fixture()
def rag_chain(mock_llm_provider, mock_retrieval_pipeline):
    """Create a RAGChain with mocked dependencies."""
    from src.services.rag.chain import RAGChain

    return RAGChain(
        llm_provider=mock_llm_provider,
        retrieval_pipeline=mock_retrieval_pipeline,
    )


# ---------------------------------------------------------------------------
# FR-1: RAG Chain Service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_rag_chain_returns_answer_with_sources(rag_chain):
    """RAGChain.aquery returns a RAGResponse with non-empty answer and sources."""
    from src.services.rag.models import RAGResponse

    result = await rag_chain.aquery("What are transformers?")

    assert isinstance(result, RAGResponse)
    assert len(result.answer) > 0
    assert len(result.sources) > 0
    assert result.query == "What are transformers?"


@pytest.mark.asyncio()
async def test_rag_chain_answer_contains_citations(rag_chain):
    """Answer text includes inline citation references like [1], [2]."""
    result = await rag_chain.aquery("What are transformers?")

    assert "[1]" in result.answer
    assert "[2]" in result.answer


@pytest.mark.asyncio()
async def test_rag_chain_sources_have_metadata(rag_chain):
    """Each source has arxiv_id, title, authors, and arxiv_url."""
    result = await rag_chain.aquery("What are transformers?")

    for source in result.sources:
        assert source.arxiv_id != ""
        assert source.title != ""
        assert len(source.authors) > 0
        assert source.arxiv_url.startswith("https://arxiv.org/abs/")


@pytest.mark.asyncio()
async def test_rag_chain_empty_query_raises_error(rag_chain):
    """Empty string query raises ValueError."""
    with pytest.raises(ValueError, match="[Qq]uery"):
        await rag_chain.aquery("")

    with pytest.raises(ValueError, match="[Qq]uery"):
        await rag_chain.aquery("   ")


@pytest.mark.asyncio()
async def test_rag_chain_no_documents_found(mock_llm_provider, mock_retrieval_pipeline):
    """When retrieval returns empty results, return graceful 'no papers found' message."""
    from src.services.rag.chain import RAGChain

    mock_retrieval_pipeline.retrieve = AsyncMock(return_value=RetrievalResult(results=[], query="quantum teleportation in cats"))

    chain = RAGChain(llm_provider=mock_llm_provider, retrieval_pipeline=mock_retrieval_pipeline)
    result = await chain.aquery("quantum teleportation in cats")

    assert "relevant papers" in result.answer.lower() or "no papers found" in result.answer.lower()
    assert len(result.sources) == 0
    # Should NOT call LLM when no documents are found
    mock_llm_provider.generate.assert_not_called()


@pytest.mark.asyncio()
async def test_rag_chain_llm_error_propagates(mock_retrieval_pipeline):
    """LLM failure propagates the appropriate exception."""
    from src.services.rag.chain import RAGChain

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=LLMServiceError("LLM is down"))

    chain = RAGChain(llm_provider=provider, retrieval_pipeline=mock_retrieval_pipeline)

    with pytest.raises(LLMServiceError, match="LLM is down"):
        await chain.aquery("What are transformers?")


# ---------------------------------------------------------------------------
# FR-2: Citation-Enforcing Prompt Templates
# ---------------------------------------------------------------------------


def test_rag_chain_prompt_includes_context():
    """Built prompt contains chunk text and paper metadata."""
    from src.services.rag.prompts import build_user_prompt

    hits = _make_search_hits(2)
    prompt = build_user_prompt(query="What are transformers?", search_hits=hits)

    # Should contain chunk text from the hits
    assert "relevant text from paper 1" in prompt
    assert "relevant text from paper 2" in prompt
    # Should contain paper metadata
    assert "2301.00001" in prompt
    assert "Test Paper 1" in prompt
    assert "Author 1A" in prompt


def test_rag_chain_prompt_enforces_citations():
    """System prompt includes citation instructions."""
    from src.services.rag.prompts import SYSTEM_PROMPT

    assert "[1]" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT.lower()
    assert "source" in SYSTEM_PROMPT.lower() or "reference" in SYSTEM_PROMPT.lower()
    assert "arxiv" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# FR-4: Streaming RAG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_rag_chain_streaming_yields_tokens(rag_chain):
    """Stream method yields text chunks."""
    tokens = []
    async for chunk in rag_chain.aquery_stream("What are transformers?"):
        tokens.append(chunk)

    # Should have yielded some text tokens
    text_tokens = [t for t in tokens if not t.startswith("\n\n[SOURCES]")]
    assert len(text_tokens) > 0


@pytest.mark.asyncio()
async def test_rag_chain_streaming_ends_with_sources(rag_chain):
    """Stream ends with a [SOURCES] JSON block containing source metadata."""
    tokens = []
    async for chunk in rag_chain.aquery_stream("What are transformers?"):
        tokens.append(chunk)

    # Last token should be the sources JSON
    last_token = tokens[-1]
    assert last_token.startswith("\n\n[SOURCES]")

    # Extract and parse the JSON
    json_str = last_token.replace("\n\n[SOURCES]", "")
    sources = json.loads(json_str)
    assert isinstance(sources, list)
    assert len(sources) > 0
    assert "arxiv_id" in sources[0]
    assert "title" in sources[0]


# ---------------------------------------------------------------------------
# FR-1 (continued): Parameter forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_rag_chain_categories_passed_to_retrieval(rag_chain, mock_retrieval_pipeline):
    """Category filter is forwarded to the retrieval pipeline."""
    await rag_chain.aquery("What are transformers?", categories=["cs.AI", "cs.CL"])

    call_kwargs = mock_retrieval_pipeline.retrieve.call_args
    assert call_kwargs.kwargs.get("categories") == ["cs.AI", "cs.CL"]


@pytest.mark.asyncio()
async def test_rag_chain_temperature_passed_to_llm(rag_chain, mock_llm_provider):
    """Temperature override is forwarded to the LLM provider."""
    await rag_chain.aquery("What are transformers?", temperature=0.3)

    call_kwargs = mock_llm_provider.generate.call_args
    assert call_kwargs.kwargs.get("temperature") == 0.3


# ---------------------------------------------------------------------------
# FR-5: Factory Function
# ---------------------------------------------------------------------------


def test_factory_creates_rag_chain():
    """Factory returns a configured RAGChain instance."""
    from src.services.rag.factory import create_rag_chain

    mock_llm = AsyncMock()
    mock_pipeline = AsyncMock()

    chain = create_rag_chain(llm_provider=mock_llm, retrieval_pipeline=mock_pipeline)

    from src.services.rag.chain import RAGChain

    assert isinstance(chain, RAGChain)


# ---------------------------------------------------------------------------
# FR-3: RAG Response Model
# ---------------------------------------------------------------------------


def test_rag_response_model_fields():
    """RAGResponse has all required fields."""
    from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference

    source = SourceReference(
        index=1,
        arxiv_id="2301.00001",
        title="Test Paper",
        authors=["Author A"],
        arxiv_url="https://arxiv.org/abs/2301.00001",
        chunk_text="Some text",
        score=0.95,
    )

    response = RAGResponse(
        answer="Test answer [1].",
        sources=[source],
        query="test query",
        retrieval_metadata=RetrievalMetadata(
            stages_executed=["hybrid_search"],
            total_candidates=10,
            timings={"hybrid_search": 0.2},
            expanded_queries=[],
        ),
        llm_metadata=LLMMetadata(
            provider="gemini",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
        ),
    )

    assert response.answer == "Test answer [1]."
    assert len(response.sources) == 1
    assert response.sources[0].arxiv_url == "https://arxiv.org/abs/2301.00001"
    assert response.retrieval_metadata.total_candidates == 10
    assert response.llm_metadata.provider == "gemini"
