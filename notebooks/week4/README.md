# Week 4: Chunking & Hybrid Search

## Overview

Week 4 adds **semantic search** on top of Week 3's BM25 keyword search. Papers are split into section-aware chunks, embedded via Jina AI into 1024-dimensional vectors, and indexed into OpenSearch alongside the original text. At query time, BM25 and KNN vector search run in parallel, and **Reciprocal Rank Fusion (RRF)** merges both ranked lists into a single result set.

## What You'll Learn

- **Section-aware chunking** — Split papers into overlapping chunks that respect section boundaries (600 words, 100-word overlap)
- **Jina AI embeddings** — Asymmetric encoding (`retrieval.passage` for documents, `retrieval.query` for queries)
- **Hybrid search** — Combine BM25 keyword relevance with KNN semantic similarity
- **RRF fusion** — Merge ranked lists without manual weight tuning
- **Graceful degradation** — If Jina is down, search falls back to BM25 automatically

## Architecture

```
Paper (full text + sections)
        │
        ▼
   TextChunker (section-aware, 600 words, 100 overlap)
        │
        ▼
   Jina AI (embed each chunk → 1024-dim vector)
        │
        ▼
   OpenSearch (bulk index: text + vector per chunk)

Query → Jina AI (embed query) → OpenSearch (BM25 + KNN + RRF) → Ranked Results
```

## Key Files

| File | Purpose |
|------|---------|
| `src/services/indexing/text_chunker.py` | Section-aware text chunking with overlap |
| `src/services/indexing/hybrid_indexer.py` | End-to-end: chunk → embed → index |
| `src/services/indexing/factory.py` | Factory wiring all indexing services |
| `src/services/embeddings/jina_client.py` | Async Jina AI embedding client |
| `src/services/embeddings/factory.py` | Embeddings client factory |
| `src/schemas/indexing/models.py` | ChunkMetadata and TextChunk models |
| `src/schemas/api/search.py` | HybridSearchRequest schema |
| `src/routers/hybrid_search.py` | POST `/api/v1/hybrid-search` endpoint |
| `src/config.py` | ChunkingSettings (env prefix `CHUNKING__`) |

## Notebook

**[week4_hybrid_search.ipynb](week4_hybrid_search.ipynb)** covers:

1. Environment setup and service health checks
2. Fetch papers from PostgreSQL
3. Test TextChunker on a real paper
4. Test Jina AI embeddings
5. Set up OpenSearch hybrid index with RRF pipeline
6. Run the full hybrid indexing pipeline
7. Inspect indexed chunks
8. Compare search modes: BM25 vs Vector vs Hybrid
9. Test the production API endpoint
10. Graceful degradation when embeddings fail

## Prerequisites

- Docker services running (`docker compose up -d`)
- Jina API key in `.env` (`JINA_API_KEY=...`)
- Papers ingested from Week 2

## Running

```bash
uv run jupyter lab notebooks/week4/week4_hybrid_search.ipynb
```

## Search Modes

| Mode | How It Works | When to Use |
|------|-------------|-------------|
| **BM25** | Keyword matching with TF-IDF scoring | Exact term queries |
| **Vector** | KNN cosine similarity on embeddings | Semantic/conceptual queries |
| **Hybrid (RRF)** | Fuses BM25 + KNN ranked lists | Best of both worlds (default) |
