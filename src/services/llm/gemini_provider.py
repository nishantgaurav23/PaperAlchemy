"""Gemini LLM provider — wraps the Google Generative AI SDK.

Uses google.genai.Client for generation and streaming, and
langchain_google_genai.ChatGoogleGenerativeAI for LangGraph agent nodes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GeminiSettings
from src.exceptions import ConfigurationError, LLMServiceError, LLMTimeoutError
from src.services.llm.provider import HealthStatus, LLMResponse, UsageMetadata

logger = logging.getLogger(__name__)


class GeminiProvider:
    """LLMProvider implementation for Google Gemini cloud inference."""

    def __init__(self, settings: GeminiSettings) -> None:
        if not settings.api_key:
            raise ConfigurationError("GEMINI__API_KEY is required for GeminiProvider")

        self._settings = settings
        self._default_model = settings.model
        self._default_temperature = settings.temperature
        self._max_output_tokens = settings.max_output_tokens
        self._client = genai.Client(api_key=settings.api_key)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model_name = model or self._default_model
        config = genai.types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._default_temperature,
            max_output_tokens=max_tokens or self._max_output_tokens,
        )

        try:
            result = await asyncio.to_thread(
                self._client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=config,
            )
            usage = self._extract_usage(result)
            return LLMResponse(text=result.text or "", model=model_name, provider="gemini", usage=usage)
        except TimeoutError as exc:
            raise LLMTimeoutError(f"Gemini generation timed out: {exc}") from exc
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(f"Gemini generation failed: {exc}") from exc

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        model_name = model or self._default_model
        config = genai.types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._default_temperature,
            max_output_tokens=max_tokens or self._max_output_tokens,
        )

        try:
            stream = await asyncio.to_thread(
                self._client.models.generate_content_stream,
                model=model_name,
                contents=prompt,
                config=config,
            )
            # The stream is a synchronous iterator from the genai SDK;
            # yield chunks via to_thread to avoid blocking the event loop.
            while True:
                chunk = await asyncio.to_thread(next, stream, None)
                if chunk is None:
                    break
                if chunk.text:
                    yield chunk.text
        except TimeoutError as exc:
            raise LLMTimeoutError(f"Gemini streaming timed out: {exc}") from exc
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(f"Gemini streaming failed: {exc}") from exc

    async def health_check(self) -> HealthStatus:
        try:
            models = await asyncio.to_thread(lambda: list(self._client.models.list()))
            return HealthStatus(
                healthy=True,
                provider="gemini",
                message=f"Gemini API accessible ({len(models)} models available)",
            )
        except Exception as exc:
            return HealthStatus(healthy=False, provider="gemini", message=f"Gemini API error: {exc}")

    def get_langchain_model(self, *, model: str | None = None, temperature: float | None = None) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model or self._default_model,
            google_api_key=self._settings.api_key,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_output_tokens=self._max_output_tokens,
        )

    async def close(self) -> None:
        pass  # SDK manages its own connections

    @staticmethod
    def _extract_usage(result) -> UsageMetadata | None:
        meta = getattr(result, "usage_metadata", None)
        if meta is None:
            return None
        return UsageMetadata(
            prompt_tokens=getattr(meta, "prompt_token_count", None),
            completion_tokens=getattr(meta, "candidates_token_count", None),
            total_tokens=getattr(meta, "total_token_count", None),
        )
