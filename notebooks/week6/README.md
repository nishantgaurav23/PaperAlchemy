# Week 6: Production Monitoring & Caching

## Overview

Week 6 adds **Redis caching** for exact-match RAG response caching (150-400x speedup on repeated queries) and **Langfuse tracing** for full pipeline observability. Both features use graceful degradation — if Redis is down or Langfuse is unconfigured, the RAG pipeline continues working normally.

## What You'll Learn

- **Redis caching** — SHA256 key generation from query parameters, async GET/SET with configurable TTL
- **Cache key isolation** — Different parameters (top_k, model, use_hybrid, categories) produce different cache keys
- **Langfuse tracing** — Per-stage spans for embedding, search, prompt construction, and generation
- **Graceful degradation** — Cache misses and tracing failures are logged but never break the pipeline
- **Factory pattern** — `make_cache_client()` and `make_langfuse_tracer()` for consistent initialization
- **Dependency injection** — `CacheDep` and `LangfuseDep` type aliases for FastAPI

## Architecture

```
User Question
     │
     ▼
AskRequest (validation)
     │
     ▼
Cache Check ──[HIT]──► Instant Response (~50ms)
     │ [MISS]
     ▼
Embed query (Jina) ──[fail]──► BM25 fallback
     │                │
     │     [Langfuse: embedding span]
     ▼
search_unified (OpenSearch)
     │                │
     │     [Langfuse: search span]
     ▼
RAGPromptBuilder
     │                │
     │     [Langfuse: prompt span]
     ▼
OllamaClient.generate_rag_answer()
     │                │
     │     [Langfuse: generation span]
     ▼
Store in Redis (TTL: 24h default)
     │
     ▼
AskResponse (answer + sources)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/services/cache/__init__.py` | Package exports (CacheClient) |
| `src/services/cache/client.py` | CacheClient — SHA256 key generation, async Redis GET/SET |
| `src/services/cache/factory.py` | `make_redis_client()` + `make_cache_client()` factories |
| `src/services/langfuse/__init__.py` | Package exports (LangfuseTracer, RAGTracer) |
| `src/services/langfuse/client.py` | LangfuseTracer — SDK wrapper for traces, spans, generations |
| `src/services/langfuse/tracer.py` | RAGTracer — per-stage tracing (embed, search, prompt, generate) |
| `src/services/langfuse/factory.py` | `make_langfuse_tracer()` factory with disabled fallback |
| `src/config.py` | RedisSettings (host, port, TTL, timeouts) + LangfuseSettings (keys, flush) |
| `src/dependency.py` | CacheDep + LangfuseDep dependency injection |
| `src/main.py` | Lifespan init for cache client + langfuse tracer |
| `src/routers/ask.py` | Cache check/store + Langfuse spans in `/ask` and `/stream` |

## Notebook

**[week6_caching_and_tracing.ipynb](week6_caching_and_tracing.ipynb)** covers:

1. Environment setup and service URL configuration
2. Service health check (FastAPI, OpenSearch, Ollama, Redis)
3. Cache miss — first query runs full RAG pipeline (~15-25s)
4. Cache hit — same query returns instantly (~50-150ms)
5. Cache key isolation — different parameters produce cache misses
6. Streaming + cache — `/stream` endpoint also uses cache
7. Langfuse tracing status — configuration check
8. Redis cache inspection — view cached keys and metadata
9. System status summary

## Prerequisites

- Docker services running (`docker compose up --build -d`)
- Ollama running natively on macOS (`brew install ollama && ollama serve`)
- Model pulled (`ollama pull llama3.2:1b`)
- Papers indexed in OpenSearch (run Week 2-4 notebooks first)
- Jina API key in `.env` (`JINA_API_KEY=...`)

## Running

```bash
uv run jupyter lab notebooks/week6/week6_caching_and_tracing.ipynb
```

## Cache Key Strategy

Cache keys are SHA256 hashes of normalized query parameters:

```python
key_data = {
    "query": query.strip().lower(),
    "model": model,
    "top_k": top_k,
    "use_hybrid": use_hybrid,
    "categories": sorted(categories) if categories else [],
}
# Key format: rag:ask:<sha256_hash>
```

This ensures:
- Same query + same params = cache hit
- Any parameter change = cache miss (new key)
- Keys are fixed-length and safe for Redis

## Configuration

```bash
# Redis (in .env)
REDIS__HOST=localhost
REDIS__PORT=6380          # Host port (6379 inside Docker)
REDIS__TTL_HOURS=24       # Cache expiry

# Langfuse (in .env)
LANGFUSE__ENABLED=true
LANGFUSE__PUBLIC_KEY=pk-...
LANGFUSE__SECRET_KEY=sk-...
LANGFUSE__HOST=http://localhost:3000
```

## Performance

| Scenario | Response Time | Notes |
|----------|--------------|-------|
| Cache MISS (full pipeline) | ~15-25s | Embed + Search + LLM generation |
| Cache HIT | ~50-150ms | Redis lookup only |
| Speedup | 150-400x | Depends on LLM generation time |
