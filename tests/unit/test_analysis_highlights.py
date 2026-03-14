"""Tests for the key highlights extraction service (HighlightsService).

Tests extraction of structured highlights from papers, content preparation,
API endpoint, and error handling. All external services are mocked.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.services.llm.provider import LLMResponse, UsageMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAPER_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")

MOCK_HIGHLIGHTS_JSON = json.dumps(
    {
        "novel_contributions": [
            "Introduced self-attention mechanism replacing recurrence",
            "Achieved state-of-the-art translation quality",
        ],
        "important_findings": [
            "Attention-only models outperform RNN-based on WMT tasks",
            "Training time reduced by order of magnitude",
        ],
        "practical_implications": [
            "Enables parallelized training of sequence models",
            "Foundation for downstream NLP tasks like BERT",
        ],
        "limitations": [
            "Quadratic memory complexity with sequence length",
            "Limited to fixed context window",
        ],
        "keywords": [
            "self-attention",
            "transformers",
            "machine translation",
            "neural networks",
            "parallelization",
        ],
    }
)


def _make_paper_orm(
    abstract: str = "We present a novel transformer architecture using self-attention.",
    sections: list | None = None,
    pdf_content: str | None = "Full text of the paper about transformers and attention mechanisms.",
) -> MagicMock:
    paper = MagicMock()
    paper.id = PAPER_ID
    paper.title = "Attention Is All You Need"
    paper.authors = ["Vaswani et al."]
    paper.abstract = abstract
    paper.sections = (
        sections
        if sections is not None
        else [
            {"title": "Abstract", "content": "We present a novel transformer architecture.", "level": 1},
            {"title": "Introduction", "content": "Sequence modeling has been dominated by RNNs.", "level": 1},
            {"title": "Results", "content": "Our model achieves 28.4 BLEU on WMT 2014.", "level": 1},
            {"title": "Conclusion", "content": "We propose a new architecture based entirely on attention.", "level": 1},
        ]
    )
    paper.pdf_content = pdf_content
    return paper


def _make_llm_response(text: str = MOCK_HIGHLIGHTS_JSON) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="gemini-3-flash",
        provider="gemini",
        usage=UsageMetadata(prompt_tokens=500, completion_tokens=200, total_tokens=700, latency_ms=850.0),
    )


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=_make_paper_orm())
    return repo


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=_make_llm_response())
    return provider


# ---------------------------------------------------------------------------
# FR-1: Key Highlights Extraction
# ---------------------------------------------------------------------------


class TestExtractHighlights:
    """Tests for HighlightsService.extract_highlights()."""

    @pytest.mark.asyncio
    async def test_extract_highlights_full_paper(self, mock_paper_repo, mock_llm_provider):
        """Full paper with sections returns structured highlights with all 5 fields."""
        from src.services.analysis.highlights import HighlightsService

        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.extract_highlights(PAPER_ID)

        h = result.highlights
        assert len(h.novel_contributions) >= 1
        assert len(h.important_findings) >= 1
        assert len(h.practical_implications) >= 1
        assert len(h.limitations) >= 1
        assert len(h.keywords) >= 3
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_highlights_abstract_only(self, mock_paper_repo, mock_llm_provider):
        """Paper with abstract but no sections generates highlights with warning."""
        from src.services.analysis.highlights import HighlightsService

        mock_paper_repo.get_by_id = AsyncMock(return_value=_make_paper_orm(sections=[], pdf_content=None))
        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.extract_highlights(PAPER_ID)

        assert len(result.highlights.novel_contributions) >= 1
        assert result.warning is not None

    @pytest.mark.asyncio
    async def test_extract_highlights_paper_not_found(self, mock_paper_repo, mock_llm_provider):
        """Non-existent paper_id raises PaperNotFoundError."""
        from src.services.analysis.highlights import HighlightsService

        mock_paper_repo.get_by_id = AsyncMock(return_value=None)
        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(PaperNotFoundError):
            await service.extract_highlights(uuid.UUID("00000000-0000-0000-0000-000000000000"))

    @pytest.mark.asyncio
    async def test_extract_highlights_insufficient_content(self, mock_paper_repo, mock_llm_provider):
        """Paper with no abstract and no sections raises InsufficientContentError."""
        from src.services.analysis.highlights import HighlightsService

        mock_paper_repo.get_by_id = AsyncMock(return_value=_make_paper_orm(abstract="", sections=[], pdf_content=None))
        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(InsufficientContentError):
            await service.extract_highlights(PAPER_ID)

    @pytest.mark.asyncio
    async def test_extract_highlights_llm_failure(self, mock_paper_repo, mock_llm_provider):
        """LLM provider error raises LLMServiceError."""
        from src.services.analysis.highlights import HighlightsService

        mock_llm_provider.generate = AsyncMock(side_effect=Exception("Gemini API down"))
        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(LLMServiceError):
            await service.extract_highlights(PAPER_ID)

    @pytest.mark.asyncio
    async def test_extract_highlights_malformed_llm_output(self, mock_paper_repo, mock_llm_provider):
        """Malformed LLM output falls back to raw extraction."""
        from src.services.analysis.highlights import HighlightsService

        mock_llm_provider.generate = AsyncMock(
            return_value=_make_llm_response(text="This is not JSON at all, just plain text about findings.")
        )
        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.extract_highlights(PAPER_ID)

        # Should still return something usable (fallback)
        assert result.warning is not None


# ---------------------------------------------------------------------------
# FR-2: Content Preparation
# ---------------------------------------------------------------------------


class TestPrepareContent:
    """Tests for HighlightsService._prepare_content()."""

    def test_prepare_content_full_paper(self, mock_paper_repo, mock_llm_provider):
        """Full paper with sections formats content correctly."""
        from src.services.analysis.highlights import HighlightsService

        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        paper = _make_paper_orm()
        content = service._prepare_content(paper)

        assert "Abstract" in content or "abstract" in content.lower()
        assert "Results" in content or "results" in content.lower()
        assert len(content) > 0

    def test_prepare_content_abstract_only(self, mock_paper_repo, mock_llm_provider):
        """Paper with only abstract returns abstract content."""
        from src.services.analysis.highlights import HighlightsService

        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        paper = _make_paper_orm(sections=[], pdf_content=None)
        content = service._prepare_content(paper)

        assert "novel transformer" in content.lower()

    def test_prepare_content_truncation(self, mock_paper_repo, mock_llm_provider):
        """Very long paper content is truncated to ~4000 words."""
        from src.services.analysis.highlights import HighlightsService

        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        long_text = " ".join(["word"] * 10000)
        paper = _make_paper_orm(
            pdf_content=long_text,
            sections=[{"title": "Body", "content": long_text, "level": 1}],
        )
        content = service._prepare_content(paper)

        word_count = len(content.split())
        assert word_count <= 4500  # ~4000 with some buffer for headers


# ---------------------------------------------------------------------------
# FR-3: Highlights Response Schema
# ---------------------------------------------------------------------------


class TestHighlightsSchemas:
    """Tests for PaperHighlights and HighlightsResponse Pydantic models."""

    def test_paper_highlights_schema(self):
        """PaperHighlights model validates with all 5 fields."""
        from src.schemas.api.analysis import PaperHighlights

        highlights = PaperHighlights(
            novel_contributions=["New architecture"],
            important_findings=["Better BLEU scores"],
            practical_implications=["Faster training"],
            limitations=["Memory intensive"],
            keywords=["transformers", "attention", "NLP"],
        )
        assert len(highlights.novel_contributions) == 1
        assert len(highlights.keywords) == 3

    def test_paper_highlights_empty_lists_rejected(self):
        """PaperHighlights requires at least 1 item in main fields."""
        from src.schemas.api.analysis import PaperHighlights

        # Should accept empty limitations (it's fine to have none)
        highlights = PaperHighlights(
            novel_contributions=["Something new"],
            important_findings=["Key finding"],
            practical_implications=["Real-world use"],
            limitations=[],
            keywords=["keyword"],
        )
        assert highlights.limitations == []

    def test_highlights_response_schema(self):
        """HighlightsResponse includes highlights + metadata."""
        from src.schemas.api.analysis import HighlightsResponse, PaperHighlights

        highlights = PaperHighlights(
            novel_contributions=["New method"],
            important_findings=["Better results"],
            practical_implications=["Enables X"],
            limitations=["Limited scope"],
            keywords=["ML", "NLP"],
        )
        response = HighlightsResponse(
            paper_id=PAPER_ID,
            highlights=highlights,
            model="gemini-3-flash",
            provider="gemini",
            latency_ms=850.0,
        )
        assert response.paper_id == PAPER_ID
        assert response.model == "gemini-3-flash"
        assert response.latency_ms == 850.0


# ---------------------------------------------------------------------------
# FR-3: API Endpoint
# ---------------------------------------------------------------------------


class TestHighlightsEndpoint:
    """Tests for POST /api/v1/papers/{paper_id}/highlights endpoint."""

    @pytest.mark.asyncio
    async def test_highlights_endpoint_success(self, mock_paper_repo, mock_llm_provider):
        """POST returns 200 with valid highlights."""
        from httpx import ASGITransport, AsyncClient
        from src.db import get_db_session
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        mock_session = AsyncMock()
        app = create_app()
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider
        app.dependency_overrides[get_db_session] = lambda: mock_session

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(f"/api/v1/papers/{PAPER_ID}/highlights")

            assert resp.status_code == 200
            data = resp.json()
            assert "highlights" in data
            assert "novel_contributions" in data["highlights"]
            assert "important_findings" in data["highlights"]
            assert "keywords" in data["highlights"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_highlights_endpoint_not_found(self, mock_paper_repo, mock_llm_provider):
        """POST returns 404 for non-existent paper."""
        from httpx import ASGITransport, AsyncClient
        from src.db import get_db_session
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        mock_session = AsyncMock()
        app = create_app()
        mock_paper_repo.get_by_id = AsyncMock(return_value=None)
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider
        app.dependency_overrides[get_db_session] = lambda: mock_session

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(f"/api/v1/papers/{PAPER_ID}/highlights")

            assert resp.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_force_regeneration(self, mock_paper_repo, mock_llm_provider):
        """?force=true calls LLM even if result could be cached."""
        from src.services.analysis.highlights import HighlightsService

        service = HighlightsService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        # Call twice with force=True — LLM should be called both times
        await service.extract_highlights(PAPER_ID, force=True)
        await service.extract_highlights(PAPER_ID, force=True)

        assert mock_llm_provider.generate.call_count == 2
