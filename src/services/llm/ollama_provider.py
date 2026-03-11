"""Ollama LLM provider — wraps the Ollama HTTP API.

Communicates with Ollama via httpx, maps errors to LLM exception hierarchy,
and provides a LangChain-compatible ChatOllama for agent nodes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from langchain_ollama import ChatOllama
from src.config import OllamaSettings
from src.exceptions import LLMConnectionError, LLMServiceError, LLMTimeoutError
from src.services.llm.provider import HealthStatus, LLMResponse, UsageMetadata

logger = logging.getLogger(__name__)


class OllamaProvider:
    """LLMProvider implementation for local Ollama inference."""

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._base_url = settings.url
        self._default_model = settings.default_model
        self._default_temperature = settings.default_temperature
        self._default_top_p = settings.default_top_p
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(float(settings.default_timeout)))

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = model or self._default_model
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        body = {"model": model, "prompt": prompt, "stream": False}
        if options:
            body["options"] = options

        try:
            response = await self._client.post(f"{self._base_url}/api/generate", json=body)
            if response.status_code != 200:
                raise LLMServiceError(f"Ollama returned HTTP {response.status_code}")
            data = response.json()
            return LLMResponse(
                text=data.get("response", ""),
                model=data.get("model", model),
                provider="ollama",
                usage=self._extract_usage(data),
            )
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Cannot connect to Ollama at {self._base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama generation timed out: {exc}") from exc
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(f"Ollama generation failed: {exc}") from exc

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        model = model or self._default_model
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        body = {"model": model, "prompt": prompt, "stream": True}
        if options:
            body["options"] = options

        try:
            async with self._client.stream("POST", f"{self._base_url}/api/generate", json=body) as response:
                if response.status_code != 200:
                    raise LLMServiceError(f"Ollama streaming failed: HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama chunk: %s", line)
                        continue
                    if chunk.get("done"):
                        break
                    text = chunk.get("response", "")
                    if text:
                        yield text
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Cannot connect to Ollama: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama streaming timed out: {exc}") from exc
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(f"Ollama streaming failed: {exc}") from exc

    async def health_check(self) -> HealthStatus:
        try:
            response = await self._client.get(f"{self._base_url}/api/version")
            if response.status_code == 200:
                data = response.json()
                return HealthStatus(
                    healthy=True,
                    provider="ollama",
                    message="Ollama is running",
                    version=data.get("version"),
                )
            return HealthStatus(healthy=False, provider="ollama", message=f"HTTP {response.status_code}")
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Cannot connect to Ollama: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama health check timed out: {exc}") from exc
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(f"Ollama health check failed: {exc}") from exc

    def get_langchain_model(self, *, model: str | None = None, temperature: float | None = None) -> ChatOllama:
        return ChatOllama(
            base_url=self._base_url,
            model=model or self._default_model,
            temperature=temperature if temperature is not None else self._default_temperature,
            request_timeout=self._client.timeout.read,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _extract_usage(data: dict) -> UsageMetadata | None:
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_duration = data.get("total_duration")

        if prompt_tokens is None and completion_tokens is None:
            return None

        total = (prompt_tokens or 0) + (completion_tokens or 0)
        latency = round(total_duration / 1_000_000, 2) if total_duration else None

        return UsageMetadata(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            latency_ms=latency,
        )
