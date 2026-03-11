# PaperAlchemy — Architecture & Design Document

> An **AI Research Assistant** that answers research questions by searching its knowledge base,
> using agents to retrieve and synthesize information, and **always citing papers with links**.

---

## Core Principle: Citation-First Research Assistant

PaperAlchemy is NOT a generic chatbot. It is a research assistant that:

1. **Always searches the knowledge base** — every research question triggers agent tool calls to hybrid search
2. **Always cites papers** — every answer includes paper title, authors, year, and arXiv link
3. **Uses agents with tool calls** — the LangGraph agent MUST invoke retrieval tools before generating
4. **Admits knowledge gaps** — if no papers are found, it says so instead of hallucinating

### Agent Tool Call Flow (Mandatory for Every Research Question)
```
User: "What is a transformer?"
    │
    ▼
[Guardrail] ── Is this a research question? YES
    │
    ▼
[Agent decides tools] ── MUST call at least one:
    ├─ tool: hybrid_search("transformer architecture attention mechanism")
    ├─ tool: hybrid_search("self-attention neural network")
    └─ tool: knowledge_base_lookup(keywords=["transformer", "attention"])
    │
    ▼
[Retrieved Papers] ── e.g., "Attention Is All You Need" (arxiv:1706.03762)
    │
    ▼
[Grade Documents] ── Are retrieved papers relevant? YES
    │
    ▼
[Generate Answer] ── Synthesize answer WITH inline citations [1], [2]
    │                  MUST include source list with arXiv links
    ▼
Response:
  "Transformers are a neural network architecture based on self-attention [1]..."

  **Sources:**
  1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
  2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              Next.js 15 Frontend (TypeScript + Tailwind)             │   │
│  │                                                                      │   │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌─────────────┐  │   │
│  │  │ Search  │ │ Chatbot  │ │Upload  │ │Dashboard │ │ Eval Panel  │  │   │
│  │  │  Page   │ │(follow-up│ │& Analyze│ │& Trends  │ │(Human Eval) │  │   │
│  │  └────┬────┘ └────┬─────┘ └───┬────┘ └────┬─────┘ └──────┬──────┘  │   │
│  │       │           │           │            │              │          │   │
│  │  ┌────┴───────────┴───────────┴────────────┴──────────────┴──────┐  │   │
│  │  │              API Client (fetch + SSE streaming)                │  │   │
│  │  └───────────────────────────┬────────────────────────────────────┘  │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │         Telegram Bot (python-telegram-bot, webhook/polling)         │   │
│  │  /ask → RAG pipeline  │  /search → hybrid search  │  Alerts/Digest │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │ HTTPS                                     │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          API GATEWAY (FastAPI :8000)                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Routers                                                             │   │
│  │  ├─ /api/v1/health          Health check                            │   │
│  │  ├─ /api/v1/search          BM25 keyword search                     │   │
│  │  ├─ /api/v1/hybrid-search   BM25 + KNN + RRF fusion                │   │
│  │  ├─ /api/v1/ask             Standard RAG (JSON)                     │   │
│  │  ├─ /api/v1/stream          Streaming RAG (SSE)                     │   │
│  │  ├─ /api/v1/chat            Chatbot with follow-up (SSE)            │   │
│  │  ├─ /api/v1/ask-agentic     Agentic RAG with reasoning              │   │
│  │  ├─ /api/v1/upload          PDF upload + analysis                    │   │
│  │  ├─ /api/v1/papers          Paper CRUD + detail                      │   │
│  │  ├─ /api/v1/collections     Reading lists management                 │   │
│  │  ├─ /api/v1/compare         Paper comparison                         │   │
│  │  ├─ /api/v1/trends          Research trends data                     │   │
│  │  ├─ /api/v1/export          BibTeX / Markdown export                 │   │
│  │  ├─ /api/v1/eval            Evaluation results & submission          │   │
│  │  └─ /api/v1/ingest/*        Data ingestion endpoints                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Middleware   │  │  Dependency  │  │   Error      │  │ Auth (API     │   │
│  │ (CORS, Log) │  │  Injection   │  │   Handler    │  │  Key based)   │   │
│  └─────────────┘  └──────────────┘  └──────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          SERVICE LAYER                                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT SYSTEM (LangGraph)                          │   │
│  │                                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │   │
│  │  │Guardrail │→ │Retrieve  │→ │ Grade    │→ │ Generate Answer  │    │   │
│  │  │  Node    │  │  Node    │  │Documents │  │  Node (citations)│    │   │
│  │  └──────────┘  └────┬─────┘  └────┬─────┘  └──────────────────┘    │   │
│  │       ↓ reject       ↑            ↓ fail                            │   │
│  │  ┌──────────┐  ┌────┴─────┐                                         │   │
│  │  │Out-of-   │  │ Rewrite  │  (max 3 retries)                       │   │
│  │  │Scope Node│  │Query Node│                                         │   │
│  │  └──────────┘  └──────────┘                                         │   │
│  │                                                                      │   │
│  │  SPECIALIZED AGENTS:                                                 │   │
│  │  ┌────────────┐ ┌─────────────┐ ┌──────────────┐ ┌───────────────┐ │   │
│  │  │ Summarizer │ │Fact-Checker │ │Trend Analyzer│ │Citation Track │ │   │
│  │  └────────────┘ └─────────────┘ └──────────────┘ └───────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    CORE SERVICES                                     │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │   │
│  │  │ LLM      │ │ Embedding│ │ Search   │ │ Cache    │ │ PDF      │ │   │
│  │  │ Client   │ │ Service  │ │ Service  │ │ Service  │ │ Parser   │ │   │
│  │  │(Ollama/  │ │(Jina AI) │ │(Hybrid)  │ │(Redis)   │ │(Docling) │ │   │
│  │  │ Gemini3) │ │          │ │          │ │          │ │          │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │ Chat     │ │ Analysis │ │ ArXiv    │ │ Langfuse │              │   │
│  │  │ Memory   │ │ Engine   │ │ Client   │ │ Tracer   │              │   │
│  │  │(sessions)│ │(summary) │ │(fetcher) │ │(observ.) │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          DATA LAYER                                         │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ PostgreSQL   │  │ OpenSearch   │  │ Redis 7      │  │ Cloud Storage│   │
│  │ 16           │  │ 2.19         │  │              │  │ / MinIO      │   │
│  │              │  │              │  │              │  │              │   │
│  │ • papers     │  │ • BM25 index │  │ • RAG cache  │  │ • PDF files  │   │
│  │ • users      │  │ • KNN vectors│  │ • Chat hist. │  │ • Exports    │   │
│  │ • collections│  │ • 1024-dim   │  │ • Sessions   │  │              │   │
│  │ • annotations│  │              │  │ • Rate limit │  │              │   │
│  │ • eval_results│ │              │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐                                        │
│  │ Airflow 2.10 │  │ Langfuse v3  │                                        │
│  │              │  │              │                                        │
│  │ • Fetch DAGs │  │ • Traces     │                                        │
│  │ • Index DAGs │  │ • Spans      │                                        │
│  │ • Reports    │  │ • Costs      │                                        │
│  └──────────────┘  └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User Flows

### Flow 1: Research Question → Cited Answer (Primary Flow)
1. User asks: "What is a transformer?"
2. Frontend sends POST to `/api/v1/chat` with `session_id`
3. **Agent MUST make tool calls** — searches knowledge base via hybrid search
4. Retrieved papers graded for relevance
5. If insufficient → query rewritten → re-search (up to 3 retries)
6. If no papers found → respond: "I don't have papers on this topic yet"
7. Answer generated with **inline citations [1], [2]** and **source list with arXiv links**
8. SSE streaming to frontend with citation rendering

### Flow 2: Follow-up Questions (Conversational)
1. User asks follow-up: "What about its limitations?"
2. Frontend sends POST to `/api/v1/chat` with same `session_id`
3. Backend loads conversation history from Redis
4. Rewrites query using context: "What are the limitations of transformers?"
5. **Agent searches knowledge base again** — always re-retrieves, never answers from memory
6. Streams cited answer via SSE, stores in conversation memory
7. Citations from follow-up may reference same or new papers

### Flow 3: Search & Explore
1. User types query in Search page
2. Frontend sends POST to `/api/v1/hybrid-search`
3. Backend runs BM25 + KNN + RRF fusion
4. Results displayed as paper cards with title, authors, abstract, arXiv link
5. User clicks "Ask about this" → Chat opens with paper context pre-loaded

### Flow 4: Upload & Analyze Paper
1. User drags PDF onto Upload page
2. Frontend sends multipart POST to `/api/v1/upload`
3. Docling parses PDF → sections, metadata extracted
4. Paper stored in PostgreSQL, chunks indexed in OpenSearch
5. Specialized agents run in parallel:
   - Summarizer → structured summary
   - Highlights → key findings & contributions
   - Methodology → datasets, baselines, results
6. Results displayed in Paper Detail View

### Flow 5: Paper Comparison
1. User selects 2+ papers from Reading List
2. Frontend sends POST to `/api/v1/compare`
3. Comparison agent analyzes: methods, results, contributions
4. Side-by-side comparison rendered with diff highlights

### Flow 6: Evaluation
1. Admin loads evaluation dataset (Q&A pairs with ground truth)
2. System runs RAG pipeline on each question
3. LLM judge scores: faithfulness, relevance, coherence
4. Human evaluator scores via Eval Panel UI
5. Metrics aggregated in Benchmark Dashboard

---

## Data Flow: Chatbot with Conversation Memory

```
User Message (session_id: "abc123")
    │
    ▼
┌─────────────────────────┐
│  Load Conversation      │
│  History from Redis     │
│  (last N messages)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Context-Aware Query    │
│  Rewriting              │
│  "What about its        │
│   limitations?"         │
│      ↓                  │
│  "What are the          │
│   limitations of        │
│   [paper from context]?"│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  RAG Pipeline           │
│  (retrieve → generate)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Store Response in      │
│  Conversation Memory    │
│  Redis (24h TTL)        │
└────────┬────────────────┘
         │
         ▼
    SSE Stream to Client
```

---

## Advanced RAG Retrieval Pipeline

The retrieval pipeline goes far beyond basic vector search. It combines multiple
advanced techniques to maximize retrieval quality before the LLM ever sees the context.

```
User Query: "How do transformers handle long sequences?"
    │
    ▼
┌───────────────────────────────────────────────────┐
│  STAGE 1: QUERY EXPANSION                         │
│                                                    │
│  ┌─────────────────┐  ┌────────────────────────┐  │
│  │ Multi-Query     │  │ HyDE                   │  │
│  │ Generation      │  │ (Hypothetical Doc      │  │
│  │                 │  │  Embeddings)            │  │
│  │ LLM generates   │  │                        │  │
│  │ 3-5 variations: │  │ LLM generates a        │  │
│  │ • "transformer  │  │ hypothetical answer,   │  │
│  │   long context" │  │ then embeds THAT       │  │
│  │ • "attention    │  │ instead of the query.  │  │
│  │   sequence      │  │ Better semantic match  │  │
│  │   length limit" │  │ with actual papers.    │  │
│  │ • "positional   │  │                        │  │
│  │   encoding      │  │                        │  │
│  │   scaling"      │  │                        │  │
│  └────────┬────────┘  └───────────┬────────────┘  │
│           │                       │                │
│           └───────────┬───────────┘                │
│                       │ all query variants         │
└───────────────────────┼───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  STAGE 2: HYBRID RETRIEVAL                        │
│                                                    │
│  For EACH query variant, run in parallel:          │
│                                                    │
│  ┌──────────────┐    ┌──────────────────────────┐ │
│  │ BM25 Search  │    │ KNN Vector Search        │ │
│  │ (keyword     │    │ (Jina 1024-dim           │ │
│  │  matching)   │    │  cosine similarity)      │ │
│  └──────┬───────┘    └────────────┬─────────────┘ │
│         │                         │                │
│         └────────────┬────────────┘                │
│                      │                             │
│              ┌───────▼───────┐                     │
│              │ RRF Fusion    │                     │
│              │ (Reciprocal   │                     │
│              │  Rank Fusion) │                     │
│              │ Combines BM25 │                     │
│              │  + KNN scores │                     │
│              └───────┬───────┘                     │
│                      │                             │
│              Deduplicate across all query variants  │
│              → Top 20 unique chunks                │
└──────────────────────┼────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  STAGE 3: RE-RANKING (Cross-Encoder)              │
│                                                    │
│  Cross-encoder model (ms-marco-MiniLM-L-12-v2)   │
│  scores each (query, chunk) pair with full         │
│  attention — much more accurate than bi-encoder.   │
│                                                    │
│  Input:  Top 20 chunks from Stage 2               │
│  Output: Re-scored + sorted → Top 5 most relevant │
│                                                    │
│  Alternative (cloud): Cohere Rerank API            │
└──────────────────────┼────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  STAGE 4: PARENT EXPANSION                        │
│                                                    │
│  For each top-5 chunk, retrieve its PARENT         │
│  (full section) to give the LLM more context.     │
│                                                    │
│  Index: small chunks (200 words) for precision     │
│  Retrieve: parent section (600+ words) for context │
│                                                    │
│  Result: 5 rich context windows with metadata      │
│  (arxiv_id, title, authors, section_name, pdf_url) │
└──────────────────────┼────────────────────────────┘
                       │
                       ▼
            Top 5 Contextual Chunks
            (ready for LLM generation
             with full citation metadata)
```

### Pipeline Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `multi_query_count` | 3 | Number of query variations to generate |
| `hyde_enabled` | true | Whether to use HyDE for one query variant |
| `retrieval_top_k` | 20 | Chunks retrieved before re-ranking |
| `rerank_top_k` | 5 | Chunks after re-ranking |
| `rerank_model` | ms-marco-MiniLM-L-12-v2 | Cross-encoder model |
| `parent_expansion` | true | Expand to parent sections |
| `child_chunk_size` | 200 | Small chunk size for precision |
| `parent_chunk_size` | 600 | Parent section size for context |
| `rrf_k` | 60 | RRF fusion constant |

---

## Operations Guide (Quick Reference)

### Prerequisites
- Docker Desktop (4GB+ RAM allocated)
- Python 3.12+ with UV package manager
- Node.js 20+ with pnpm (for frontend)
- Ollama installed locally (for dev LLM)

### Start Everything
```bash
# 1. Clone and setup
git clone <repo> && cd PaperAlchemy
cp .env.example .env          # Edit with your API keys

# 2. Start infrastructure (PostgreSQL, OpenSearch, Redis, etc.)
make start                     # docker compose up --build -d

# 3. Wait for services to be healthy
make health                    # Check all service health

# 4. Pull Ollama model (first time only)
ollama pull llama3.2

# 5. Start API server (development)
make dev                       # uvicorn src.main:app --reload

# 6. Start frontend (parallel terminal)
cd frontend && pnpm dev        # Next.js dev server on :3000

# 7. Start Gradio UI (optional dev UI)
make gradio                    # Gradio on :7861
```

### Stop Everything
```bash
make stop                      # docker compose down
# To also remove volumes (DELETES DATA):
make clean                     # docker compose down -v
```

### Key URLs (Development)
| Service | URL |
|---------|-----|
| API docs (Swagger) | http://localhost:8000/docs |
| Frontend (Next.js) | http://localhost:3000 |
| Gradio UI | http://localhost:7861 |
| pgAdmin | http://localhost:5050 |
| OpenSearch Dashboards | http://localhost:5602 |
| Airflow | http://localhost:8080 |
| Langfuse | http://localhost:3001 |

### Makefile Commands
| Command | Action |
|---------|--------|
| `make start` | Start all Docker services |
| `make stop` | Stop all Docker services |
| `make restart` | Rebuild and restart |
| `make status` | Show service status |
| `make logs` | Tail all service logs |
| `make health` | Health check all services |
| `make dev` | Start API in dev mode (hot reload) |
| `make setup` | Install Python dependencies (uv sync) |
| `make format` | Format code (ruff format) |
| `make lint` | Lint + type check (ruff + mypy) |
| `make test` | Run all tests (pytest) |
| `make test-cov` | Tests with coverage report |
| `make gradio` | Start Gradio dev UI |
| `make clean` | Remove containers + volumes + caches |

### Troubleshooting
| Problem | Solution |
|---------|----------|
| OpenSearch won't start | Increase Docker memory to 4GB+; check `vm.max_map_count` on Linux |
| Ollama connection refused | Ensure Ollama is running: `ollama serve` |
| Jina embedding fails | Check `JINA_API_KEY` in .env |
| Redis connection error | Check Redis is healthy: `make health` |
| Port already in use | Stop conflicting services or change ports in compose.yml |

---

## Tech Stack Rationale

| Choice | Why | Alternatives Considered |
|--------|-----|------------------------|
| **Gemini 3 Flash** (cloud) | Best cost/perf for research, free tier, fast | Claude API (costly), GPT-4o (costly) |
| **Ollama** (local) | Free, no API keys needed for dev | vLLM (heavier), llama.cpp (lower-level) |
| **Next.js 15** | App Router, SSR, streaming, React Server Components | SvelteKit (smaller ecosystem), Remix |
| **shadcn/ui** | Accessible, customizable, Tailwind-native | MUI (heavy), Chakra (opinionated) |
| **OpenSearch** | Hybrid BM25+KNN in one engine, open-source | Qdrant (vector-only), Weaviate (less BM25) |
| **LangGraph** | State-machine agents, decision graphs, retries | CrewAI (less control), AutoGen (complex) |
| **GCP Cloud Run** | Scale to zero, generous free tier, container-native | AWS Lambda (cold starts), Fly.io (less free) |
| **RAGAS** | Standard RAG evaluation framework | TruLens (heavier), custom (more work) |
| **Docling** | Section-aware PDF parsing, scientific papers | PyMuPDF (no sections), GROBID (Java) |
| **Jina AI** | Free embeddings API, high quality 1024-dim | OpenAI (costly), Cohere (rate limits) |
| **python-telegram-bot** | Mature async Telegram lib, webhook + polling | Aiogram (less docs), Telethon (MTProto, overkill) |

---

## Security Design

| Concern | Mitigation |
|---------|-----------|
| API Authentication | API key-based auth for production endpoints |
| File Upload | 50MB limit, PDF-only validation, content-type check |
| Secrets | All via environment variables, GCP Secret Manager in prod |
| CORS | Strict origin allowlist in production |
| Rate Limiting | Redis-based per-IP rate limiting |
| SQL Injection | SQLAlchemy ORM parameterized queries |
| XSS | Next.js auto-escaping, CSP headers |
| Prompt Injection | Guardrail node filters malicious queries |

---

## Deployment Architecture (GCP)

```
┌─────────────────────────────────────────────────┐
│                   GCP Project                    │
│                                                  │
│  ┌───────────────┐    ┌───────────────────────┐ │
│  │ Cloud Run     │    │ Cloud Run             │ │
│  │ (API)         │    │ (Frontend - Next.js)  │ │
│  │ Scale: 0-3    │    │ Scale: 0-2            │ │
│  │ Memory: 1GB   │    │ Memory: 512MB         │ │
│  └───────┬───────┘    └───────────────────────┘ │
│          │                                       │
│  ┌───────┴───────┐    ┌───────────────────────┐ │
│  │ Cloud SQL     │    │ Cloud Storage         │ │
│  │ (PostgreSQL)  │    │ (PDFs + exports)      │ │
│  └───────────────┘    └───────────────────────┘ │
│                                                  │
│  ┌───────────────┐    ┌───────────────────────┐ │
│  │ Upstash Redis │    │ GCP Secret Manager    │ │
│  │ (free tier)   │    │ (API keys, DB creds)  │ │
│  └───────────────┘    └───────────────────────┘ │
│                                                  │
│  External:                                       │
│  • Gemini 3 Flash API (free tier)               │
│  • Jina Embeddings API (free tier)              │
│  • OpenSearch (self-hosted on Cloud Run or       │
│    use managed Aiven free trial)                │
└─────────────────────────────────────────────────┘
```

---

## Evaluation Framework Design

### Datasets
- **SciQ**: Scientific question answering dataset
- **QASPER**: Q&A on research papers
- **Custom curated**: 100+ domain-specific Q&A pairs from indexed papers

### Metrics
| Metric | Type | Description |
|--------|------|-------------|
| **Faithfulness** | RAGAS | Answer grounded in retrieved context |
| **Answer Relevance** | RAGAS | Answer addresses the question |
| **Context Precision** | RAGAS | Retrieved docs relevant to query |
| **Context Recall** | RAGAS | Retrieved docs cover ground truth |
| **Hallucination Rate** | Custom | % of claims not in context |
| **Citation Accuracy** | Custom | % of citations pointing to correct source |
| **Response Latency** | System | P50/P95/P99 response times |
| **User Satisfaction** | Human | 1-5 Likert scale from eval UI |

### Judge Models
- **Primary**: Gemini 3 Flash (cost-effective, research-capable)
- **Cross-validation**: Compare scores across multiple judge models
- **Human baseline**: Subset evaluated by human annotators for calibration

---

## Cost Estimates (Monthly)

| Component | Free Tier | Over Free Tier |
|-----------|-----------|----------------|
| Cloud Run (API) | 2M requests | $0.00004/req |
| Cloud Run (Frontend) | Included above | — |
| Cloud SQL (PostgreSQL) | $0 (trial) | ~$7/mo smallest |
| Upstash Redis | 10K cmd/day | $0.20/100K cmd |
| Cloud Storage | 5GB | $0.02/GB |
| Gemini 3 Flash | 15 RPM free | $0.075/1M tokens |
| Jina Embeddings | 1M tokens/mo | — |
| **Estimated Total** | **$0** | **$7-15/mo** |
