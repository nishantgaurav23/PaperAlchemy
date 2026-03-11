"""Tests for parent-child chunk retrieval (S4b.4).

TDD: These tests are written FIRST, before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.schemas.indexing import ChildChunk, ChunkMetadata, ParentChildResult, TextChunk
from src.services.indexing.parent_child import ParentChildChunker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_text(word_count: int) -> str:
    """Generate text with the given word count."""
    words = [f"word{i}" for i in range(word_count)]
    return " ".join(words)


SAMPLE_PAPER_TEXT = _make_text(1500)  # ~2-3 parent chunks
SHORT_PAPER_TEXT = _make_text(100)  # single parent, single child
TINY_PAPER_TEXT = _make_text(30)  # below min_chunk_size


# ---------------------------------------------------------------------------
# FR-1: Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_valid_config(self):
        chunker = ParentChildChunker()
        assert chunker.parent_chunk_size == 600
        assert chunker.child_chunk_size == 200
        assert chunker.child_overlap == 50
        assert chunker.min_chunk_size == 50

    def test_init_custom_config(self):
        chunker = ParentChildChunker(
            parent_chunk_size=800,
            child_chunk_size=250,
            child_overlap=60,
            min_chunk_size=30,
        )
        assert chunker.parent_chunk_size == 800
        assert chunker.child_chunk_size == 250
        assert chunker.child_overlap == 60
        assert chunker.min_chunk_size == 30

    def test_init_invalid_child_size(self):
        with pytest.raises(ValueError, match="child_chunk_size must be less than parent_chunk_size"):
            ParentChildChunker(parent_chunk_size=200, child_chunk_size=200)

    def test_init_invalid_child_size_greater(self):
        with pytest.raises(ValueError, match="child_chunk_size must be less than parent_chunk_size"):
            ParentChildChunker(parent_chunk_size=200, child_chunk_size=300)

    def test_init_invalid_overlap(self):
        with pytest.raises(ValueError, match="child_overlap must be less than child_chunk_size"):
            ParentChildChunker(child_chunk_size=200, child_overlap=200)


# ---------------------------------------------------------------------------
# FR-3: ChildChunk Schema
# ---------------------------------------------------------------------------


class TestChildChunkSchema:
    def test_child_chunk_model(self):
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=100,
            word_count=50,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        child = ChildChunk(
            text="some text",
            metadata=meta,
            arxiv_id="2401.12345",
            paper_id="uuid-123",
            parent_chunk_index=0,
        )
        assert child.parent_chunk_index == 0
        assert child.arxiv_id == "2401.12345"

    def test_parent_child_result_model(self):
        result = ParentChildResult()
        assert result.parents == []
        assert result.children == []

    def test_parent_child_result_with_data(self):
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=100,
            word_count=50,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text="parent", metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")
        child = ChildChunk(text="child", metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123", parent_chunk_index=0)
        result = ParentChildResult(parents=[parent], children=[child])
        assert len(result.parents) == 1
        assert len(result.children) == 1


# ---------------------------------------------------------------------------
# FR-4: Split Parent into Children
# ---------------------------------------------------------------------------


class TestSplitParentIntoChildren:
    def test_split_parent_into_children(self):
        """A 600-word parent with child_size=200, overlap=50 should produce multiple children."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50)
        parent_text = _make_text(600)
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=len(parent_text),
            word_count=600,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text=parent_text, metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")

        children = chunker.split_parent_into_children(parent)

        assert len(children) > 1
        # Each child should reference parent index 0
        for child in children:
            assert child.parent_chunk_index == 0
            assert child.arxiv_id == "2401.12345"
            assert child.metadata.word_count <= 200

    def test_split_parent_short(self):
        """Parent shorter than child_chunk_size → single child."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50, min_chunk_size=10)
        parent_text = _make_text(100)
        meta = ChunkMetadata(
            chunk_index=2,
            start_char=0,
            end_char=len(parent_text),
            word_count=100,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text=parent_text, metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")

        children = chunker.split_parent_into_children(parent)

        assert len(children) == 1
        assert children[0].parent_chunk_index == 2
        assert children[0].text == parent_text

    def test_split_parent_empty_text(self):
        """Parent with empty text → empty list."""
        chunker = ParentChildChunker()
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=0,
            word_count=0,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text="", metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")

        children = chunker.split_parent_into_children(parent)
        assert children == []

    def test_child_references_parent(self):
        """All children must reference the correct parent chunk_index."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50)
        parent_text = _make_text(500)
        meta = ChunkMetadata(
            chunk_index=5,
            start_char=0,
            end_char=len(parent_text),
            word_count=500,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text=parent_text, metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")

        children = chunker.split_parent_into_children(parent)
        for child in children:
            assert child.parent_chunk_index == 5

    def test_children_have_sequential_indices(self):
        """Child chunk indices should be sequential starting from 0."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50)
        parent_text = _make_text(600)
        meta = ChunkMetadata(
            chunk_index=0,
            start_char=0,
            end_char=len(parent_text),
            word_count=600,
            overlap_with_previous=0,
            overlap_with_next=0,
        )
        parent = TextChunk(text=parent_text, metadata=meta, arxiv_id="2401.12345", paper_id="uuid-123")

        children = chunker.split_parent_into_children(parent)
        indices = [c.metadata.chunk_index for c in children]
        assert indices == list(range(len(children)))


# ---------------------------------------------------------------------------
# FR-2: Create Parent-Child Chunks
# ---------------------------------------------------------------------------


class TestCreateParentChildChunks:
    def test_create_parent_child_chunks(self):
        """Full paper should produce parents and children with correct links."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50)
        result = chunker.create_parent_child_chunks(
            title="Test Paper",
            abstract="An abstract.",
            full_text=SAMPLE_PAPER_TEXT,
            arxiv_id="2401.12345",
            paper_id="uuid-123",
        )

        assert isinstance(result, ParentChildResult)
        assert len(result.parents) > 0
        assert len(result.children) > 0
        # More children than parents
        assert len(result.children) > len(result.parents)
        # All children reference a valid parent index
        parent_indices = {p.metadata.chunk_index for p in result.parents}
        for child in result.children:
            assert child.parent_chunk_index in parent_indices

    def test_create_parent_child_empty_text(self):
        """Empty text → empty result."""
        chunker = ParentChildChunker()
        result = chunker.create_parent_child_chunks(
            title="Test",
            abstract="Abstract",
            full_text="",
            arxiv_id="2401.12345",
            paper_id="uuid-123",
        )
        assert result.parents == []
        assert result.children == []

    def test_create_parent_child_short_text(self):
        """Short text → one parent, at least one child."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50, min_chunk_size=10)
        result = chunker.create_parent_child_chunks(
            title="Test",
            abstract="Abstract",
            full_text=SHORT_PAPER_TEXT,
            arxiv_id="2401.12345",
            paper_id="uuid-123",
        )
        assert len(result.parents) == 1
        assert len(result.children) >= 1
        assert result.children[0].parent_chunk_index == result.parents[0].metadata.chunk_index

    def test_section_based_parent_child(self):
        """Section-aware chunking produces parent-child hierarchy."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50)
        sections = {
            "Introduction": _make_text(400),
            "Methods": _make_text(500),
            "Results": _make_text(300),
        }
        result = chunker.create_parent_child_chunks(
            title="Test Paper",
            abstract="An abstract of the paper.",
            full_text=_make_text(1200),
            arxiv_id="2401.12345",
            paper_id="uuid-123",
            sections=sections,
        )
        assert len(result.parents) > 0
        assert len(result.children) > 0
        # All children link to existing parents
        parent_indices = {p.metadata.chunk_index for p in result.parents}
        for child in result.children:
            assert child.parent_chunk_index in parent_indices


# ---------------------------------------------------------------------------
# FR-5: Parent Expansion
# ---------------------------------------------------------------------------


class TestExpandToParents:
    def test_expand_to_parents_basic(self):
        """Child results with different parent IDs → both parents returned."""
        chunker = ParentChildChunker()
        child_results = [
            {
                "chunk_text": "child 1 text",
                "parent_chunk_id": "2401.12345_parent_0",
                "arxiv_id": "2401.12345",
                "score": 0.9,
            },
            {
                "chunk_text": "child 2 text",
                "parent_chunk_id": "2401.12345_parent_1",
                "arxiv_id": "2401.12345",
                "score": 0.8,
            },
        ]

        os_client = MagicMock()
        os_client.client.get.side_effect = [
            {"_source": {"chunk_text": "parent 0 text", "arxiv_id": "2401.12345", "chunk_index": 0}},
            {"_source": {"chunk_text": "parent 1 text", "arxiv_id": "2401.12345", "chunk_index": 1}},
        ]

        parents = chunker.expand_to_parents(child_results, os_client)
        assert len(parents) == 2

    def test_expand_to_parents_dedup(self):
        """Multiple children from same parent → one parent returned with best score."""
        chunker = ParentChildChunker()
        child_results = [
            {
                "chunk_text": "child 1",
                "parent_chunk_id": "2401.12345_parent_0",
                "arxiv_id": "2401.12345",
                "score": 0.9,
            },
            {
                "chunk_text": "child 2",
                "parent_chunk_id": "2401.12345_parent_0",
                "arxiv_id": "2401.12345",
                "score": 0.7,
            },
        ]

        os_client = MagicMock()
        os_client.client.get.return_value = {
            "_source": {"chunk_text": "parent 0 text", "arxiv_id": "2401.12345", "chunk_index": 0}
        }

        parents = chunker.expand_to_parents(child_results, os_client)
        assert len(parents) == 1
        assert parents[0]["score"] == 0.9  # best score preserved

    def test_expand_to_parents_missing_id(self):
        """Children without parent_chunk_id → returned as-is."""
        chunker = ParentChildChunker()
        child_results = [
            {
                "chunk_text": "orphan child",
                "arxiv_id": "2401.12345",
                "score": 0.8,
            },
        ]

        os_client = MagicMock()
        parents = chunker.expand_to_parents(child_results, os_client)
        assert len(parents) == 1
        assert parents[0]["chunk_text"] == "orphan child"

    def test_expand_to_parents_empty(self):
        """Empty input → empty output."""
        chunker = ParentChildChunker()
        os_client = MagicMock()
        parents = chunker.expand_to_parents([], os_client)
        assert parents == []

    def test_expand_to_parents_fetch_failure_fallback(self):
        """If parent fetch fails, return child as-is."""
        chunker = ParentChildChunker()
        child_results = [
            {
                "chunk_text": "child text",
                "parent_chunk_id": "2401.12345_parent_0",
                "arxiv_id": "2401.12345",
                "score": 0.8,
            },
        ]

        os_client = MagicMock()
        os_client.client.get.side_effect = Exception("not found")

        parents = chunker.expand_to_parents(child_results, os_client)
        assert len(parents) == 1
        assert parents[0]["chunk_text"] == "child text"


# ---------------------------------------------------------------------------
# FR-6: Prepare for Indexing
# ---------------------------------------------------------------------------


class TestPrepareForIndexing:
    def test_prepare_for_indexing(self):
        """Children → OpenSearch-ready dicts with parent_chunk_id."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50, min_chunk_size=10)
        result = chunker.create_parent_child_chunks(
            title="Test Paper",
            abstract="Abstract.",
            full_text=_make_text(600),
            arxiv_id="2401.12345",
            paper_id="uuid-123",
        )

        docs = chunker.prepare_for_indexing(result)

        assert len(docs) > 0
        for doc in docs:
            assert "chunk_data" in doc
            data = doc["chunk_data"]
            assert "chunk_text" in data
            assert "parent_chunk_id" in data
            assert "arxiv_id" in data
            assert "paper_id" in data
            assert "chunk_index" in data
            assert data["parent_chunk_id"].startswith("2401.12345_parent_")

    def test_prepare_for_indexing_empty(self):
        """Empty result → empty list."""
        chunker = ParentChildChunker()
        result = ParentChildResult()
        docs = chunker.prepare_for_indexing(result)
        assert docs == []

    def test_prepare_for_indexing_deterministic_ids(self):
        """Parent chunk IDs should be deterministic: {arxiv_id}_parent_{index}."""
        chunker = ParentChildChunker(parent_chunk_size=600, child_chunk_size=200, child_overlap=50, min_chunk_size=10)
        result = chunker.create_parent_child_chunks(
            title="Test",
            abstract="Abstract.",
            full_text=_make_text(600),
            arxiv_id="2401.99999",
            paper_id="uuid-456",
        )
        docs = chunker.prepare_for_indexing(result)
        parent_ids = {doc["chunk_data"]["parent_chunk_id"] for doc in docs}
        for pid in parent_ids:
            assert pid.startswith("2401.99999_parent_")
