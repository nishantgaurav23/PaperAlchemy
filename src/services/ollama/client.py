"""
Why It's Needed                                                                                                                  
   
  The Ollama client is the bridge between PaperAlchemy and the local LLM. Without it, every router would need to:                  
  - Manually construct HTTP requests to Ollama                                                                                   
  - Handle connection errors, timeouts, and retries
  - Parse token usage and timing metadata
  - Build RAG prompts from chunks

  This client centralizes all that complexity.

  ---
  What It Does
  ┌──────────────────────────────┬───────────────────────────────────────────────────────┐
  │            Method            │                        Purpose                        │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ health_check()               │ Verifies Ollama is running (used by /health endpoint) │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ list_models()                │ Gets available models (for UI dropdowns)              │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ generate()                   │ Core text generation with usage metadata              │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ generate_stream()            │ Streaming generation for real-time UI                 │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ generate_rag_answer()        │ Main RAG method — prompt + generate + parse           │
  ├──────────────────────────────┼───────────────────────────────────────────────────────┤
  │ generate_rag_answer_stream() │ Streaming RAG for /stream endpoint                    │
  └──────────────────────────────┴───────────────────────────────────────────────────────┘
  ---
  How It Helps

  1. Centralized error handling — Connection, timeout, and general errors map to custom exceptions
  2. Usage metadata — Token counts and latency for cost tracking
  3. Async throughout — Non-blocking for FastAPI
  4. Settings-driven — All defaults from OllamaSettings, overridable per-request


  Ollama client for local LLM inference with RAG support.

  Why it's needed:
      PaperAlchemy uses Ollama to run LLMs locally for answering questions about
      papers. This client wraps the Ollama HTTP API with proper error handling,
      timeout management, and RAG-specific methods for generating answers from
      retrieved paper chunks.

  What it does:
      - health_check(): Verifies Ollama is running and responsive
      - list_models(): Gets available models (useful for UI dropdowns)
      - generate(): Core text generation with usage metadata
      - generate_stream(): Streaming generation for real-time UI updates
      - generate_rag_answer(): Main RAG method - builds prompt, generates, parses
      - generate_rag_answer_stream(): Streaming version for /stream endpoint

  How it helps:
      - Centralized error handling (connection, timeout, general errors)
      - Custom exceptions map to appropriate HTTP responses in routers
      - Usage metadata (tokens, latency) enables cost tracking
      - All methods are async for non-blocking FastAPI integration

Key Optimizations
  ┌──────────────────────────────────────┬─────────────────────────────────┐
  │             Optimization             │             Benefit             │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Single httpx.AsyncClient per request │ Proper connection lifecycle     │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ _extract_usage_metadata helper       │ Avoids code duplication         │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ _build_response_from_chunks helper   │ Separates parsing logic         │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Settings defaults with or fallback   │ 3-tier config hierarchy         │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Early raise for known exceptions     │ Prevents double-wrapping errors │
  └──────────────────────────────────────┴─────────────────────────────────┘

"""
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.base_url = settings.ollama.url
        self.timeout = httpx.Timeout(float(settings.ollama.default_timeout))
        self.default_model = settings.ollama.default_model
        self.default_temperature = settings.ollama.default_temperature
        self.default_top_p = settings.ollama.default_top_p
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama is healthy and responding.
        
        Returns:
            Dictinary with status, message, and version

        Raised:
            OllamaConenctionError: Cannot connect to Ollama
            OllamaTimeoutError: Request timed out
            OllamaException: Other errors
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }
                else:
                    raise OllamaException(f"Ollama returned status {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")
        
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from Ollama.
        
        Returns:
            List of model info dictionaries

        Raises:
            OllamaConnectionError: Cannot connect to Ollama
            OllamaTimeoutError: Request timed out
            OllamaException: Other errors
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                else:
                    raise OllamaException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")
        
    async def generate(
            self,
            model: str,
            prompt: str,
            stream: bool = False,
            **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate text using specified model.
        
        Args:
            model: Model name (e.g, "llama3.2", "mistral:7b")
            prompt: Input promot for generation
            stream: Whether to stream response
            **kwargs: Additional parms (temperature, top+p, format)

        Returns:
            Response dict with 'response' text and 'usage_metadata

        Raises:
            OllamaConnectionError: Cannot connect to Ollama
            OllamaTimeoutError: Generation timed out
            OllamaException: Other errors
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

                logger.info(f"Sending request to Ollama: model={model}, stream={stream}")
                response = await client.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    result = response.json()
                    result["usage_metadata"] = self._extract_usage_metadata(result)
                    return result
                else:
                    raise OllamaException(f"Generation failed: {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")
        
    def _extract_usage_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize usage metadata from Ollama response.
        
        Ollama returns token counts and timing differently than OpenAI.
        This normalizes them from consistent logging.

        Args:
            result: Raw Ollama response

        Returns:
            Normalized usage metadata dict
        """
        usage = {}

        # Token counts
        if "prompt_eval_count" in result:
            usage["prompt_tokens"] = result["prompt_eval_count"]
        if "eval_count" in result:
            usage["completion_tokens"] = result["eval_count"]

        # Total tokekns
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        # Timing (nanoseconds -> milliseconds)
        if "total_duration" in result:
            usage["latency_ms"] = round(result["total_duration"] / 1_000_000, 2)
        if "prompt_eval_duration" in result:
            usage["prompt_eval_duration_ms"] = round(result["prompt_eval_duration"] / 1_000_000, 2)
        if "eval_duration" in result:
            usage["eval_duration_ms"] = round(result["eval_duration"] / 1_000_000, 2)

        return usage
    
    async def generate_stream(
            self,
            model: str,
            prompt: str,
            **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text with streaming response.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Yields:
            JSON chunks with 'response' (partial text) and 'done' flag

        Raise:
            OllamaConnectionError: Cannot connect to Ollama
            OllamaTimeoutError: Timeout during streaming
            OllamaException: Other errors
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": True, **kwargs}

                logger.info(f"Starting streaming generation: model={model}")

                async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                    if response.status_code != 200:
                        raise OllamaException(f"Streaming failed: {response.status_code}")

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {line}")
        

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved chunks with 'chunk_text' and 'arxiv_id'
            model: Model to use (defaults to settings value)
            temperature: Generation temperature (defaults to settings value)
            top_p: Nucleus sampling threshold (defaults to settings value)
            use_structured_output: Use Ollama's JSON format feature

        Returns:
            Dict with 'answer', 'sources', 'confidence', 'citations'

        Raises:
            OllamaException: Generation or parsing failed
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        try:
            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=temperature,
                    top_p=top_p,
                    format=prompt_data["format"],
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                )

            if response and "response" in response:
                answer_text = response["response"]
                logger.debug(f"Raw LLM response: {answer_text[:500]}...")

                if use_structured_output:
                    return self.response_parser.parse_structured_response(answer_text)
                else:
                    return self._build_response_from_chunks(answer_text, chunks)
            else:
                raise OllamaException("No response generated from Ollama")

        except OllamaException:
            raise
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    def _build_response_from_chunks(
        self,
        answer_text: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build structured response from plain text and chunk metadata.

        Args:
            answer_text: Raw LLM output
            chunks: Original context chunks

        Returns:
            Structured response dict
        """
        sources, seen_urls = [], set()
        citations = []

        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id")
            if arxiv_id:
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                if pdf_url not in seen_urls:
                    sources.append(pdf_url)
                    seen_urls.add(pdf_url)
                if arxiv_id not in citations:
                    citations.append(arxiv_id)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": "medium",
            "citations": citations[:5],
        }

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming RAG answer using retrieved chunks.

        Note: Structured output not supported in streaming mode.

        Args:
            query: User's question
            chunks: Retrieved document chunks
            model: Model to use
            temperature: Generation temperature
            top_p: Nucleus sampling threshold

        Yields:
            Streaming response chunks

        Raises:
            OllamaException: Generation failed
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        try:
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
            ):
                yield chunk

        except OllamaException:
            raise
        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")


