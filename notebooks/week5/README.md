# Week 5: Complete RAG Pipeline with LLM Integration

## Overview

Week 5 completes the RAG system by adding **answer generation with a local LLM**. User queries are embedded via Jina AI, searched against OpenSearch using hybrid BM25 + KNN, and the retrieved chunks are fed to Ollama (local LLM) to generate grounded answers with source citations. Both standard and streaming (SSE) endpoints are provided.

## What You'll Learn

- **Ollama integration** — HTTP client for local LLM inference (`/api/generate`)
- **RAG prompt engineering** — System prompt loaded from file, structured output with JSON schema
- **Response parsing** — JSON parse → regex extraction → plain text fallback chain
- **SSE streaming** — Server-Sent Events for real-time token-by-token output
- **Factory pattern** — `make_ollama_client()` for consistent service initialization
- **3-tier config** — Per-request params → environment variables → code defaults
- **Custom exceptions** — `LLMException → OllamaException → OllamaConnectionError/OllamaTimeoutError`

## Architecture

```
User Question
      │
      ▼
  FastAPI Router (/ask or /stream)
      │
      ├──► Jina AI (embed query → 1024-dim vector)
      │         │
      │         ▼
      │    OpenSearch (BM25 + KNN + RRF hybrid search)
      │         │
      │         ▼
      │    Retrieved Chunks + Source URLs
      │
      ▼
  RAGPromptBuilder (system prompt + chunks + query)
      │
      ▼
  Ollama LLM (llama3.2:1b, local inference)
      │
      ├──► /ask: Complete JSON response (answer + sources + metadata)
      └──► /stream: SSE events (metadata → tokens → done)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/services/ollama/client.py` | Ollama HTTP client (health, generate, stream, RAG) |
| `src/services/ollama/prompts.py` | RAGPromptBuilder and ResponseParser |
| `src/services/ollama/prompts/rag_system.txt` | System prompt template |
| `src/services/ollama/factory.py` | `make_ollama_client()` factory |
| `src/services/ollama/__init__.py` | Package exports |
| `src/routers/ask.py` | `/ask` and `/stream` endpoints |
| `src/schemas/api/ask.py` | AskRequest and AskResponse models |
| `src/schemas/ollama.py` | RAGResponse (structured LLM output schema) |
| `src/exceptions.py` | Ollama-specific exception hierarchy |
| `src/config.py` | OllamaSettings (host, port, model, temperature, top_p) |
| `src/dependency.py` | OllamaDep (Annotated + Depends injection) |
| `src/main.py` | Router registration + Ollama lifespan init |
| `compose.yml` | Ollama service container with volume + healthcheck |

## Notebook

**[week5_complete_rag_system.ipynb](week5_complete_rag_system.ipynb)** covers:

1. Environment setup and service URL configuration
2. Service health check (FastAPI, OpenSearch, Ollama)
3. API structure overview (all registered endpoints)
4. Ollama LLM test (list models, simple generation)
5. Hybrid search test (verify indexed data)
6. Complete RAG pipeline — standard `/ask` endpoint
7. Parameter variations (BM25-only, low temperature, category filter)
8. Streaming RAG pipeline — `/stream` with SSE parsing
9. Edge cases (short queries, missing categories, single chunk)
10. System status summary

## Prerequisites

- Docker services running (`docker compose up --build -d`)
- Ollama model pulled (`docker exec paperalchemy-ollama ollama pull llama3.2:1b`)
- Papers indexed in OpenSearch (run Week 2 + Week 3 notebooks first)
- Jina API key in `.env` (`JINA_API_KEY=...`)

## Running

```bash
uv run jupyter lab notebooks/week5/week5_complete_rag_system.ipynb
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | Standard RAG — returns complete answer with sources |
| `/api/v1/stream` | POST | Streaming RAG — SSE real-time token output |

## Request/Response

**Request:**
```json
{
    "query": "What are recent advances in machine learning?",
    "top_k": 3,
    "use_hybrid": true,
    "model": "llama3.2:1b",
    "temperature": 0.7,
    "top_p": 0.9,
    "categories": ["cs.AI"]
}
```

**Response (Standard):**
```json
{
    "query": "What are recent advances in machine learning?",
    "answer": "Generated answer based on retrieved papers...",
    "sources": ["https://arxiv.org/pdf/2508.21263.pdf"],
    "chunks_used": 3,
    "search_mode": "hybrid",
    "model": "llama3.2:1b"
}
```

**Response (Streaming SSE):**
```
data: {"sources": [...], "chunks_used": 3, "search_mode": "hybrid", "model": "llama3.2:1b"}
data: {"chunk": "The"}
data: {"chunk": " answer"}
data: {"chunk": " is..."}
data: {"done": true, "answer": "The answer is..."}
```
