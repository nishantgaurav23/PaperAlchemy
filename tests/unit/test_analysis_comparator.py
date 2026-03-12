"""Tests for the paper comparison service (ComparatorService).

Tests side-by-side comparison of 2+ papers, multi-paper content extraction,
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

PAPER_ID_1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
PAPER_ID_2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
PAPER_ID_3 = uuid.UUID("33333333-3333-3333-3333-333333333333")

MOCK_COMPARISON_JSON = json.dumps(
    {
        "methods_comparison": "Paper 1 uses self-attention while Paper 2 uses bidirectional pre-training.",
        "results_comparison": "Paper 1 achieves 28.4 BLEU on WMT while Paper 2 achieves SoTA on 11 NLP tasks.",
        "contributions_comparison": "Paper 1 introduces the Transformer; Paper 2 introduces masked language modeling.",
        "limitations_comparison": "Both suffer from quadratic attention complexity. Paper 2 requires large pre-training compute.",
        "common_themes": [
            "Self-attention mechanisms",
            "Large-scale language modeling",
            "Transfer learning for NLP",
        ],
        "key_differences": [
            "Paper 1 focuses on translation, Paper 2 on general NLP",
            "Paper 1 is encoder-decoder, Paper 2 is encoder-only",
            "Paper 2 uses masked pre-training, Paper 1 does not",
        ],
        "verdict": (
            "Paper 1 laid the architectural foundation; Paper 2 demonstrated its power "
            "for transfer learning. They are complementary contributions."
        ),
    }
)


def _make_paper_orm(
    paper_id: uuid.UUID = PAPER_ID_1,
    title: str = "Attention Is All You Need",
    authors: list[str] | None = None,
    abstract: str = "We present a novel transformer architecture using self-attention.",
    sections: list | None = None,
    pdf_content: str | None = "Full text of the paper about transformers and attention mechanisms.",
) -> MagicMock:
    paper = MagicMock()
    paper.id = paper_id
    paper.title = title
    paper.authors = authors or ["Vaswani et al."]
    paper.abstract = abstract
    paper.sections = (
        sections
        if sections is not None
        else [
            {"title": "Introduction", "content": "Sequence modeling has been dominated by RNNs.", "level": 1},
            {"title": "Results", "content": "Our model achieves 28.4 BLEU on WMT 2014.", "level": 1},
            {"title": "Conclusion", "content": "We propose a new architecture based entirely on attention.", "level": 1},
        ]
    )
    paper.pdf_content = pdf_content
    return paper


PAPER_1 = _make_paper_orm(
    paper_id=PAPER_ID_1,
    title="Attention Is All You Need",
    authors=["Vaswani et al."],
)
PAPER_2 = _make_paper_orm(
    paper_id=PAPER_ID_2,
    title="BERT: Pre-training of Deep Bidirectional Transformers",
    authors=["Devlin et al."],
    abstract="We introduce BERT, a bidirectional pre-training approach for language representations.",
    sections=[
        {"title": "Introduction", "content": "Language model pre-training has shown effective.", "level": 1},
        {"title": "Results", "content": "BERT achieves state-of-the-art on 11 NLP tasks.", "level": 1},
    ],
)
PAPER_3 = _make_paper_orm(
    paper_id=PAPER_ID_3,
    title="GPT-3: Language Models are Few-Shot Learners",
    authors=["Brown et al."],
    abstract="We demonstrate that scaling up language models improves few-shot performance.",
    sections=[
        {"title": "Introduction", "content": "Recent work has shown that large LMs can learn tasks.", "level": 1},
        {"title": "Results", "content": "GPT-3 achieves strong few-shot performance.", "level": 1},
    ],
)


def _make_llm_response(text: str = MOCK_COMPARISON_JSON) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="gemini-3-flash",
        provider="gemini",
        usage=UsageMetadata(prompt_tokens=1200, completion_tokens=400, total_tokens=1600, latency_ms=1500.0),
    )


def _paper_lookup(paper_map: dict[uuid.UUID, MagicMock | None]):
    """Create an async side_effect function for get_by_id that returns papers from a map."""

    async def _get_by_id(paper_id):
        return paper_map.get(paper_id)

    return _get_by_id


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(side_effect=_paper_lookup({PAPER_ID_1: PAPER_1, PAPER_ID_2: PAPER_2, PAPER_ID_3: PAPER_3}))
    return repo


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=_make_llm_response())
    return provider


# ---------------------------------------------------------------------------
# FR-1: Multi-Paper Comparison Generation
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for ComparatorService.compare()."""

    @pytest.mark.asyncio
    async def test_compare_two_papers(self, mock_paper_repo, mock_llm_provider):
        """Two papers produce a structured comparison with all fields."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.compare([PAPER_ID_1, PAPER_ID_2])

        c = result.comparison
        assert len(result.comparison.papers) == 2
        assert c.methods_comparison
        assert c.results_comparison
        assert c.contributions_comparison
        assert c.limitations_comparison
        assert len(c.common_themes) >= 1
        assert len(c.key_differences) >= 1
        assert c.verdict
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_three_papers(self, mock_paper_repo, mock_llm_provider):
        """Three papers: all included in output."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.compare([PAPER_ID_1, PAPER_ID_2, PAPER_ID_3])

        assert len(result.comparison.papers) == 3

    @pytest.mark.asyncio
    async def test_compare_paper_not_found(self, mock_paper_repo, mock_llm_provider):
        """One paper_id doesn't exist -> PaperNotFoundError."""
        from src.services.analysis.comparator import ComparatorService

        missing_id = uuid.UUID("99999999-9999-9999-9999-999999999999")
        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(PaperNotFoundError):
            await service.compare([PAPER_ID_1, missing_id])

    @pytest.mark.asyncio
    async def test_compare_insufficient_content(self, mock_paper_repo, mock_llm_provider):
        """One paper has no abstract/sections -> InsufficientContentError."""
        from src.services.analysis.comparator import ComparatorService

        empty_paper = _make_paper_orm(paper_id=PAPER_ID_2, abstract="", sections=[], pdf_content=None)
        mock_paper_repo.get_by_id = AsyncMock(side_effect=_paper_lookup({PAPER_ID_1: PAPER_1, PAPER_ID_2: empty_paper}))
        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(InsufficientContentError):
            await service.compare([PAPER_ID_1, PAPER_ID_2])

    @pytest.mark.asyncio
    async def test_compare_fewer_than_two(self, mock_paper_repo, mock_llm_provider):
        """Single paper_id -> ValueError."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(ValueError, match="at least 2"):
            await service.compare([PAPER_ID_1])

    @pytest.mark.asyncio
    async def test_compare_more_than_five(self, mock_paper_repo, mock_llm_provider):
        """Six paper_ids -> ValueError."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        ids = [uuid.uuid4() for _ in range(6)]

        with pytest.raises(ValueError, match="at most 5"):
            await service.compare(ids)

    @pytest.mark.asyncio
    async def test_compare_duplicate_ids(self, mock_paper_repo, mock_llm_provider):
        """Duplicate IDs are deduplicated."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.compare([PAPER_ID_1, PAPER_ID_2, PAPER_ID_1])

        assert len(result.comparison.papers) == 2

    @pytest.mark.asyncio
    async def test_compare_llm_failure(self, mock_paper_repo, mock_llm_provider):
        """LLM provider error raises LLMServiceError."""
        from src.services.analysis.comparator import ComparatorService

        mock_llm_provider.generate = AsyncMock(side_effect=Exception("Gemini API down"))
        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(LLMServiceError):
            await service.compare([PAPER_ID_1, PAPER_ID_2])

    @pytest.mark.asyncio
    async def test_compare_malformed_llm_output(self, mock_paper_repo, mock_llm_provider):
        """Malformed LLM output falls back with warning."""
        from src.services.analysis.comparator import ComparatorService

        mock_llm_provider.generate = AsyncMock(return_value=_make_llm_response(text="This is not JSON at all, just plain text."))
        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.compare([PAPER_ID_1, PAPER_ID_2])

        assert result.warning is not None


# ---------------------------------------------------------------------------
# FR-2: Multi-Paper Content Extraction
# ---------------------------------------------------------------------------


class TestExtractMultiContent:
    """Tests for ComparatorService._extract_multi_content()."""

    def test_extract_multi_content(self, mock_paper_repo, mock_llm_provider):
        """Multi-paper content formatted with per-paper labels."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        content = service._extract_multi_content([PAPER_1, PAPER_2])

        assert "Paper 1" in content
        assert "Paper 2" in content
        assert "Attention Is All You Need" in content
        assert "BERT" in content

    def test_extract_multi_content_truncation(self, mock_paper_repo, mock_llm_provider):
        """Combined content is truncated to stay within limits."""
        from src.services.analysis.comparator import ComparatorService

        service = ComparatorService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        long_text = " ".join(["word"] * 10000)
        p1 = _make_paper_orm(paper_id=PAPER_ID_1, pdf_content=long_text, sections=[{"title": "Body", "content": long_text}])
        p2 = _make_paper_orm(paper_id=PAPER_ID_2, pdf_content=long_text, sections=[{"title": "Body", "content": long_text}])
        content = service._extract_multi_content([p1, p2])

        word_count = len(content.split())
        assert word_count <= 7000  # ~6000 with some buffer for labels


# ---------------------------------------------------------------------------
# FR-3: Schemas
# ---------------------------------------------------------------------------


class TestComparisonSchemas:
    """Tests for PaperComparison, ComparisonRequest, ComparisonResponse Pydantic models."""

    def test_comparison_request_valid(self):
        """ComparisonRequest accepts 2-5 paper IDs."""
        from src.schemas.api.analysis import ComparisonRequest

        req = ComparisonRequest(paper_ids=[PAPER_ID_1, PAPER_ID_2])
        assert len(req.paper_ids) == 2

    def test_comparison_request_too_few(self):
        """ComparisonRequest rejects fewer than 2 IDs."""
        from pydantic import ValidationError
        from src.schemas.api.analysis import ComparisonRequest

        with pytest.raises(ValidationError):
            ComparisonRequest(paper_ids=[PAPER_ID_1])

    def test_comparison_request_too_many(self):
        """ComparisonRequest rejects more than 5 IDs."""
        from pydantic import ValidationError
        from src.schemas.api.analysis import ComparisonRequest

        with pytest.raises(ValidationError):
            ComparisonRequest(paper_ids=[uuid.uuid4() for _ in range(6)])

    def test_comparison_response_schema(self):
        """ComparisonResponse includes comparison + metadata."""
        from src.schemas.api.analysis import ComparedPaper, ComparisonResponse, PaperComparison

        comparison = PaperComparison(
            papers=[
                ComparedPaper(id=PAPER_ID_1, title="Paper A", authors=["Author A"]),
                ComparedPaper(id=PAPER_ID_2, title="Paper B", authors=["Author B"]),
            ],
            methods_comparison="Methods differ in X.",
            results_comparison="Results differ in Y.",
            contributions_comparison="Contributions differ in Z.",
            limitations_comparison="Both are limited by W.",
            common_themes=["Theme 1"],
            key_differences=["Diff 1"],
            verdict="Paper A focuses on X, Paper B on Y.",
        )
        response = ComparisonResponse(
            paper_ids=[PAPER_ID_1, PAPER_ID_2],
            comparison=comparison,
            model="gemini-3-flash",
            provider="gemini",
            latency_ms=1500.0,
        )
        assert len(response.paper_ids) == 2
        assert response.model == "gemini-3-flash"


# ---------------------------------------------------------------------------
# FR-3: API Endpoint
# ---------------------------------------------------------------------------


class TestComparisonEndpoint:
    """Tests for POST /api/v1/papers/compare endpoint."""

    @pytest.mark.asyncio
    async def test_comparison_endpoint_success(self, mock_paper_repo, mock_llm_provider):
        """POST returns 200 with valid comparison."""
        from httpx import ASGITransport, AsyncClient
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        app = create_app()
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/papers/compare",
                    json={"paper_ids": [str(PAPER_ID_1), str(PAPER_ID_2)]},
                )

            assert resp.status_code == 200
            data = resp.json()
            assert "comparison" in data
            assert "methods_comparison" in data["comparison"]
            assert "key_differences" in data["comparison"]
            assert "papers" in data["comparison"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_comparison_endpoint_not_found(self, mock_paper_repo, mock_llm_provider):
        """POST returns 404 for non-existent paper."""
        from httpx import ASGITransport, AsyncClient
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        app = create_app()
        missing_id = uuid.UUID("99999999-9999-9999-9999-999999999999")
        mock_paper_repo.get_by_id = AsyncMock(side_effect=_paper_lookup({PAPER_ID_1: PAPER_1}))
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/papers/compare",
                    json={"paper_ids": [str(PAPER_ID_1), str(missing_id)]},
                )

            assert resp.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_comparison_endpoint_validation(self, mock_paper_repo, mock_llm_provider):
        """POST with <2 papers returns 422."""
        from httpx import ASGITransport, AsyncClient
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        app = create_app()
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/papers/compare",
                    json={"paper_ids": [str(PAPER_ID_1)]},
                )

            assert resp.status_code == 422
        finally:
            app.dependency_overrides.clear()
