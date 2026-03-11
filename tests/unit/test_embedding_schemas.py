"""Tests for Jina AI embedding Pydantic schemas (FR-1)."""

from __future__ import annotations

import pytest
from src.schemas.embeddings import JinaEmbeddingRequest, JinaEmbeddingResponse


class TestJinaEmbeddingRequest:
    """Tests for the Jina API request schema."""

    def test_default_values(self):
        req = JinaEmbeddingRequest(input=["hello"])
        assert req.model == "jina-embeddings-v3"
        assert req.dimensions == 1024
        assert req.task == "retrieval.passage"
        assert req.input == ["hello"]

    def test_custom_values(self):
        req = JinaEmbeddingRequest(
            model="jina-embeddings-v3",
            task="retrieval.query",
            dimensions=512,
            input=["query text"],
        )
        assert req.task == "retrieval.query"
        assert req.dimensions == 512

    def test_multiple_inputs(self):
        texts = ["text1", "text2", "text3"]
        req = JinaEmbeddingRequest(input=texts)
        assert len(req.input) == 3

    def test_serialization(self):
        req = JinaEmbeddingRequest(input=["test"])
        data = req.model_dump()
        assert "model" in data
        assert "task" in data
        assert "dimensions" in data
        assert "input" in data

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            JinaEmbeddingRequest(input=[])


class TestJinaEmbeddingResponse:
    """Tests for the Jina API response schema."""

    def test_parse_response(self):
        raw = {
            "model": "jina-embeddings-v3",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
            ],
            "usage": {"total_tokens": 10},
        }
        resp = JinaEmbeddingResponse(**raw)
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.1] * 1024
        assert resp.data[0].index == 0
        assert resp.usage.total_tokens == 10

    def test_multiple_embeddings(self):
        raw = {
            "model": "jina-embeddings-v3",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
                {"object": "embedding", "index": 1, "embedding": [0.2] * 1024},
            ],
            "usage": {"total_tokens": 20},
        }
        resp = JinaEmbeddingResponse(**raw)
        assert len(resp.data) == 2
        assert resp.data[1].index == 1

    def test_extract_embeddings(self):
        raw = {
            "model": "jina-embeddings-v3",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.5, 0.6, 0.7]},
            ],
            "usage": {"total_tokens": 5},
        }
        resp = JinaEmbeddingResponse(**raw)
        assert resp.data[0].embedding == [0.5, 0.6, 0.7]
