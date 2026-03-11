"""Tests for unified LLM provider (S5.1).

Covers: LLMResponse/UsageMetadata/HealthStatus models, OllamaProvider,
GeminiProvider, and factory functions. All external calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.config import GeminiSettings, OllamaSettings, Settings
from src.exceptions import ConfigurationError, LLMConnectionError, LLMServiceError, LLMTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ollama_settings(**overrides) -> OllamaSettings:
    defaults = dict(
        host="localhost",
        port=11434,
        default_model="llama3.2:1b",
        default_timeout=300,
        default_temperature=0.7,
        default_top_p=0.9,
    )
    defaults.update(overrides)
    return OllamaSettings(**defaults)


def _gemini_settings(**overrides) -> GeminiSettings:
    defaults = dict(api_key="test-api-key", model="gemini-2.0-flash", temperature=0.7, max_output_tokens=4096, timeout=60)
    defaults.update(overrides)
    return GeminiSettings(**defaults)


# ===================================================================
# Response Models
# ===================================================================


class TestResponseModels:
    def test_usage_metadata_creation(self):
        from src.services.llm.provider import UsageMetadata

        usage = UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30, latency_ms=150.5)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert usage.latency_ms == 150.5

    def test_usage_metadata_optional_fields(self):
        from src.services.llm.provider import UsageMetadata

        usage = UsageMetadata()
        assert usage.prompt_tokens is None
        assert usage.total_tokens is None

    def test_llm_response_creation(self):
        from src.services.llm.provider import LLMResponse, UsageMetadata

        resp = LLMResponse(text="Hello world", model="llama3.2", provider="ollama", usage=UsageMetadata(total_tokens=5))
        assert resp.text == "Hello world"
        assert resp.model == "llama3.2"
        assert resp.provider == "ollama"
        assert resp.usage.total_tokens == 5

    def test_llm_response_no_usage(self):
        from src.services.llm.provider import LLMResponse

        resp = LLMResponse(text="hi", model="gemini", provider="gemini")
        assert resp.usage is None

    def test_health_status(self):
        from src.services.llm.provider import HealthStatus

        hs = HealthStatus(healthy=True, provider="ollama", message="OK", version="0.5.0")
        assert hs.healthy is True
        assert hs.version == "0.5.0"

    def test_health_status_no_version(self):
        from src.services.llm.provider import HealthStatus

        hs = HealthStatus(healthy=False, provider="gemini", message="Unreachable")
        assert hs.version is None


# ===================================================================
# OllamaProvider
# ===================================================================


class TestOllamaProvider:
    def _make_provider(self, settings=None):
        from src.services.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings or _ollama_settings())

    @pytest.mark.asyncio
    async def test_generate(self):
        provider = self._make_provider()
        mock_response = httpx.Response(
            200,
            json={
                "response": "The answer is 42",
                "model": "llama3.2:1b",
                "prompt_eval_count": 10,
                "eval_count": 20,
                "total_duration": 500_000_000,
            },
        )
        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate("What is the meaning of life?")

        assert result.text == "The answer is 42"
        assert result.model == "llama3.2:1b"
        assert result.provider == "ollama"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_overrides(self):
        provider = self._make_provider()
        mock_response = httpx.Response(200, json={"response": "ok", "model": "mistral"})
        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await provider.generate("hi", model="mistral", temperature=0.1, max_tokens=100)

        assert result.text == "ok"
        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["model"] == "mistral"
        assert call_json["options"]["temperature"] == 0.1
        assert call_json["options"]["num_predict"] == 100

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        provider = self._make_provider()
        chunks = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = lambda: _async_iter(chunks)
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=mock_stream_response):
            collected = []
            async for chunk in provider.generate_stream("test"):
                collected.append(chunk)

        assert collected == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        provider = self._make_provider()
        mock_response = httpx.Response(200, json={"version": "0.5.1"})
        with patch.object(provider._client, "get", new_callable=AsyncMock, return_value=mock_response):
            status = await provider.health_check()

        assert status.healthy is True
        assert status.provider == "ollama"
        assert status.version == "0.5.1"

    @pytest.mark.asyncio
    async def test_health_check_unreachable(self):
        provider = self._make_provider()
        with (
            patch.object(provider._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")),
            pytest.raises(LLMConnectionError),
        ):
            await provider.health_check()

    @pytest.mark.asyncio
    async def test_generate_timeout(self):
        provider = self._make_provider()
        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timed out")),
            pytest.raises(LLMTimeoutError),
        ):
            await provider.generate("slow prompt")

    @pytest.mark.asyncio
    async def test_generate_generic_error(self):
        provider = self._make_provider()
        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")),
            pytest.raises(LLMConnectionError),
        ):
            await provider.generate("test")

    def test_get_langchain_model(self):
        provider = self._make_provider()
        with patch("src.services.llm.ollama_provider.ChatOllama") as mock_chat:
            model = provider.get_langchain_model(temperature=0.0)
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs
            assert kwargs["temperature"] == 0.0
            assert kwargs["model"] == "llama3.2:1b"
            assert model is mock_chat.return_value

    @pytest.mark.asyncio
    async def test_close(self):
        provider = self._make_provider()
        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()


# ===================================================================
# GeminiProvider
# ===================================================================


class TestGeminiProvider:
    def _make_provider(self, settings=None):
        from src.services.llm.gemini_provider import GeminiProvider

        with patch("src.services.llm.gemini_provider.genai"):
            return GeminiProvider(settings or _gemini_settings())

    @pytest.mark.asyncio
    async def test_generate(self):
        provider = self._make_provider()
        mock_result = MagicMock()
        mock_result.text = "Gemini says hello"
        mock_result.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=10, total_token_count=15)
        provider._client.models.generate_content = MagicMock(return_value=mock_result)

        result = await provider.generate("Hello")

        assert result.text == "Gemini says hello"
        assert result.provider == "gemini"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        provider = self._make_provider()
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        provider._client.models.generate_content_stream = MagicMock(return_value=iter([chunk1, chunk2]))

        collected = []
        async for text in provider.generate_stream("test"):
            collected.append(text)

        assert collected == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        provider = self._make_provider()
        mock_model = MagicMock()
        mock_model.name = "gemini-2.0-flash"
        provider._client.models.list = MagicMock(return_value=iter([mock_model]))

        status = await provider.health_check()
        assert status.healthy is True
        assert status.provider == "gemini"

    @pytest.mark.asyncio
    async def test_health_check_auth_failure(self):
        provider = self._make_provider()
        provider._client.models.list = MagicMock(side_effect=Exception("PERMISSION_DENIED"))

        status = await provider.health_check()
        assert status.healthy is False

    def test_missing_api_key_raises(self):
        from src.services.llm.gemini_provider import GeminiProvider

        with pytest.raises(ConfigurationError, match="GEMINI__API_KEY"):
            GeminiProvider(_gemini_settings(api_key=""))

    def test_get_langchain_model(self):
        provider = self._make_provider()
        with patch("src.services.llm.gemini_provider.ChatGoogleGenerativeAI") as mock_chat:
            model = provider.get_langchain_model(temperature=0.3)
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args.kwargs
            assert kwargs["temperature"] == 0.3
            assert model is mock_chat.return_value

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        provider = self._make_provider()
        provider._client.models.generate_content = MagicMock(side_effect=Exception("API error"))

        with pytest.raises(LLMServiceError):
            await provider.generate("test")

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        provider = self._make_provider()
        await provider.close()  # Should not raise


# ===================================================================
# Factory
# ===================================================================


class TestFactory:
    def test_selects_gemini_when_api_key_set(self):
        from src.services.llm.factory import create_llm_provider
        from src.services.llm.gemini_provider import GeminiProvider

        settings = Settings()
        settings.gemini = _gemini_settings(api_key="real-key")

        with patch("src.services.llm.factory.genai"):
            provider = create_llm_provider(settings)
        assert isinstance(provider, GeminiProvider)

    def test_selects_ollama_when_no_api_key(self):
        from src.services.llm.factory import create_llm_provider
        from src.services.llm.ollama_provider import OllamaProvider

        settings = Settings()
        settings.gemini = _gemini_settings(api_key="")

        provider = create_llm_provider(settings)
        assert isinstance(provider, OllamaProvider)

    def test_multi_provider_both_available(self):
        from src.services.llm.factory import create_llm_providers
        from src.services.llm.gemini_provider import GeminiProvider
        from src.services.llm.ollama_provider import OllamaProvider

        settings = Settings()
        settings.gemini = _gemini_settings(api_key="real-key")

        with patch("src.services.llm.factory.genai"):
            providers = create_llm_providers(settings)

        assert "ollama" in providers
        assert "gemini" in providers
        assert isinstance(providers["ollama"], OllamaProvider)
        assert isinstance(providers["gemini"], GeminiProvider)

    def test_multi_provider_ollama_only(self):
        from src.services.llm.factory import create_llm_providers
        from src.services.llm.ollama_provider import OllamaProvider

        settings = Settings()
        settings.gemini = _gemini_settings(api_key="")

        providers = create_llm_providers(settings)
        assert "ollama" in providers
        assert "gemini" not in providers
        assert isinstance(providers["ollama"], OllamaProvider)


# ===================================================================
# Async iteration helper
# ===================================================================


async def _async_iter(items):
    for item in items:
        yield item
