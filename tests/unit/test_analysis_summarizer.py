"""Tests for the paper summarizer service and endpoint (S8.2)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.schemas.api.analysis import PaperSummary, SummaryResponse
from src.services.analysis.summarizer import SummarizerService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAPER_ID = uuid.uuid4()


def _make_paper(
    *,
    paper_id: uuid.UUID = PAPER_ID,
    title: str = "Attention Is All You Need",
    abstract: str = "We propose a new simple network architecture, the Transformer.",
    sections: list[dict] | None = None,
    pdf_content: str | None = None,
) -> MagicMock:
    """Create a mock Paper ORM object."""
    paper = MagicMock()
    paper.id = paper_id
    paper.title = title
    paper.abstract = abstract
    paper.sections = sections
    paper.pdf_content = pdf_content
    paper.authors = ["Vaswani, A.", "Shazeer, N."]
    paper.arxiv_id = "1706.03762"
    return paper


FULL_SECTIONS = [
    {
        "title": "Introduction",
        "content": "Sequence transduction models are based on complex recurrent architectures.",
        "level": 1,
    },
    {
        "title": "Methodology",
        "content": "We use multi-head attention with scaled dot-product.",
        "level": 1,
    },
    {
        "title": "Results",
        "content": "Our model achieves 28.4 BLEU on WMT 2014 English-to-German.",
        "level": 1,
    },
    {
        "title": "Conclusion",
        "content": "The Transformer is the first sequence transduction model based entirely on attention.",
        "level": 1,
    },
]

STRUCTURED_LLM_OUTPUT = """{
    "objective": "Propose a new architecture based solely on attention mechanisms.",
    "method": "Multi-head self-attention with positional encoding.",
    "key_findings": [
        "Achieves 28.4 BLEU on WMT 2014 English-to-German translation.",
        "Training time is significantly reduced compared to recurrent models.",
        "Generalizes well to other tasks like English constituency parsing."
    ],
    "contribution": "Introduces the Transformer, based entirely on attention.",
    "limitations": "May struggle with very long sequences due to quadratic complexity."
}"""


def _make_llm_response(text: str = STRUCTURED_LLM_OUTPUT) -> MagicMock:
    """Create a mock LLMResponse."""
    resp = MagicMock()
    resp.text = text
    resp.model = "gemini-3-flash"
    resp.provider = "gemini"
    resp.usage = MagicMock()
    resp.usage.latency_ms = 1200.0
    resp.usage.total_tokens = 500
    return resp


# ---------------------------------------------------------------------------
# Service Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_full_paper():
    """FR-1: Full paper with sections -> structured summary with all 5 fields."""
    paper = _make_paper(sections=FULL_SECTIONS, pdf_content="Full text content here.")
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)
    mock_session = AsyncMock()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=_make_llm_response())

    service = SummarizerService()
    result = await service.summarize(
        paper_id=PAPER_ID,
        paper_repo=mock_repo,
        session=mock_session,
        llm_provider=mock_llm,
    )

    assert isinstance(result, SummaryResponse)
    assert result.summary.objective
    assert result.summary.method
    assert len(result.summary.key_findings) >= 1
    assert result.summary.contribution
    assert result.summary.limitations
    assert result.paper_id == PAPER_ID
    assert result.title == "Attention Is All You Need"
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_abstract_only():
    """FR-1 edge case: Paper with abstract but no sections -> summary with warning."""
    paper = _make_paper(sections=None, pdf_content=None)
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)
    mock_session = AsyncMock()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=_make_llm_response())

    service = SummarizerService()
    result = await service.summarize(
        paper_id=PAPER_ID,
        paper_repo=mock_repo,
        session=mock_session,
        llm_provider=mock_llm,
    )

    assert isinstance(result, SummaryResponse)
    assert "abstract only" in " ".join(result.warnings).lower() or len(result.warnings) > 0


@pytest.mark.asyncio
async def test_summarize_paper_not_found():
    """FR-1 edge case: Non-existent paper -> PaperNotFoundError."""
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=None)
    mock_session = AsyncMock()
    mock_llm = AsyncMock()

    service = SummarizerService()
    with pytest.raises(PaperNotFoundError):
        await service.summarize(
            paper_id=uuid.uuid4(),
            paper_repo=mock_repo,
            session=mock_session,
            llm_provider=mock_llm,
        )


@pytest.mark.asyncio
async def test_summarize_insufficient_content():
    """FR-1 edge case: Paper with empty abstract and no sections -> InsufficientContentError."""
    paper = _make_paper(abstract="", sections=None, pdf_content=None)
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)
    mock_session = AsyncMock()
    mock_llm = AsyncMock()

    service = SummarizerService()
    with pytest.raises(InsufficientContentError):
        await service.summarize(
            paper_id=PAPER_ID,
            paper_repo=mock_repo,
            session=mock_session,
            llm_provider=mock_llm,
        )


@pytest.mark.asyncio
async def test_summarize_llm_failure():
    """FR-1 edge case: LLM provider raises exception -> LLMServiceError propagated."""
    paper = _make_paper(sections=FULL_SECTIONS)
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)
    mock_session = AsyncMock()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=LLMServiceError("LLM unavailable"))

    service = SummarizerService()
    with pytest.raises(LLMServiceError):
        await service.summarize(
            paper_id=PAPER_ID,
            paper_repo=mock_repo,
            session=mock_session,
            llm_provider=mock_llm,
        )


# ---------------------------------------------------------------------------
# Content Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_content_full_paper():
    """FR-2: Extract and format abstract + key sections."""
    paper = _make_paper(sections=FULL_SECTIONS)

    service = SummarizerService()
    content = service.extract_content(paper)

    assert "Attention Is All You Need" in content
    assert "Transformer" in content  # from abstract
    assert "multi-head attention" in content  # from methodology section
    assert "28.4 BLEU" in content  # from results section


def test_extract_content_truncation():
    """FR-2 edge case: Very long paper content truncated to ~4000 words."""
    # Create a paper with very long sections
    long_section = {"title": "Long Section", "content": "word " * 5000, "level": 1}
    paper = _make_paper(sections=[long_section])

    service = SummarizerService()
    content = service.extract_content(paper)

    word_count = len(content.split())
    assert word_count <= 4500  # allow some header overhead


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


def test_summary_response_schema():
    """FR-3: Verify SummaryResponse schema validates correctly."""
    summary = PaperSummary(
        objective="Test objective",
        method="Test method",
        key_findings=["Finding 1", "Finding 2"],
        contribution="Test contribution",
        limitations="Test limitations",
    )
    response = SummaryResponse(
        paper_id=PAPER_ID,
        title="Test Paper",
        summary=summary,
        model="gemini-3-flash",
        provider="gemini",
        latency_ms=1200.0,
        warnings=[],
    )

    assert response.paper_id == PAPER_ID
    assert response.summary.objective == "Test objective"
    assert len(response.summary.key_findings) == 2
    assert response.model == "gemini-3-flash"


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


def _make_test_app(mock_repo, mock_llm):
    """Create a minimal FastAPI app with the analysis router and mocked deps."""
    from fastapi import FastAPI
    from src.dependency import get_db_session, get_llm_provider, get_paper_repository
    from src.routers.analysis import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    mock_session = AsyncMock()
    app.dependency_overrides[get_db_session] = lambda: mock_session
    app.dependency_overrides[get_paper_repository] = lambda: mock_repo
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    return app


@pytest.mark.asyncio
async def test_summary_endpoint_success():
    """FR-3: POST /api/v1/papers/{paper_id}/summary returns 200."""
    paper = _make_paper()
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=_make_llm_response())

    app = _make_test_app(mock_repo, mock_llm)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/api/v1/papers/{PAPER_ID}/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["objective"]
    assert data["paper_id"] == str(PAPER_ID)


@pytest.mark.asyncio
async def test_summary_endpoint_not_found():
    """FR-3 edge case: POST for non-existent paper returns 404."""
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=None)
    mock_llm = AsyncMock()

    app = _make_test_app(mock_repo, mock_llm)

    fake_id = uuid.uuid4()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/api/v1/papers/{fake_id}/summary")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_force_regeneration():
    """FR-4: ?force=true always calls LLM (no caching behavior difference in this implementation)."""
    paper = _make_paper(sections=FULL_SECTIONS)
    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=paper)
    mock_session = AsyncMock()

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=_make_llm_response())

    service = SummarizerService()
    result = await service.summarize(
        paper_id=PAPER_ID,
        paper_repo=mock_repo,
        session=mock_session,
        llm_provider=mock_llm,
        force=True,
    )

    assert isinstance(result, SummaryResponse)
    mock_llm.generate.assert_called_once()
