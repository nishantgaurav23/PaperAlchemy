"""Tests for embedding service factory (FR-5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from src.config import JinaSettings, Settings
from src.exceptions import ConfigurationError
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.embeddings.factory import make_embeddings_client


class TestMakeEmbeddingsClient:
    """FR-5: Factory function creates configured client."""

    def test_creates_client_from_settings(self):
        """Factory uses settings.jina config."""
        settings = Settings(
            jina=JinaSettings(api_key="test-api-key", model="jina-embeddings-v3", dimensions=1024, batch_size=50, timeout=15),
        )
        client = make_embeddings_client(settings)
        assert isinstance(client, JinaEmbeddingsClient)
        assert client._api_key == "test-api-key"
        assert client._dimensions == 1024

    def test_uses_get_settings_when_none(self):
        """When no settings passed, reads from get_settings()."""
        mock_settings = Settings(
            jina=JinaSettings(api_key="env-key"),
        )
        with patch("src.services.embeddings.factory.get_settings", return_value=mock_settings):
            client = make_embeddings_client()
        assert isinstance(client, JinaEmbeddingsClient)
        assert client._api_key == "env-key"

    def test_missing_api_key_raises(self):
        """Empty API key raises ConfigurationError."""
        settings = Settings(
            jina=JinaSettings(api_key=""),
        )
        with pytest.raises(ConfigurationError, match="API key"):
            make_embeddings_client(settings)
