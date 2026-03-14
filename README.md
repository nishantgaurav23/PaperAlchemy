<p align="center">
  <img src="docs/architecture.svg" alt="PaperAlchemy Architecture" width="100%"/>
</p>

<h1 align="center">PaperAlchemy</h1>

<p align="center">
  <strong>AI Research Assistant that answers questions grounded in real papers — every answer cited, every source linked</strong>
</p>

<p align="center">
  <a href="#what-is-paperalchemy">About</a> &bull;
  <a href="#key-capabilities">Capabilities</a> &bull;
  <a href="#tech-stack">Stack</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#getting-started">Setup</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="#development">Dev</a> &bull;
  <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2+-orange" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/OpenSearch-2.19-005EB8?logo=opensearch&logoColor=white" alt="OpenSearch"/>
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Tests-1845-brightgreen" alt="Tests"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Gemini-Research%20QA-4285F4?logo=googlegemini&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/Claude-Code%20Gen-CC785C?logo=anthropic&logoColor=white" alt="Claude"/>
  <img src="https://img.shields.io/badge/Ollama-Local%20Dev-lightgrey" alt="Ollama"/>
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white" alt="Redis"/>
  <img src="https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?logo=googlecloud&logoColor=white" alt="GCP"/>
</p>

---

## What is PaperAlchemy?

PaperAlchemy is **not** a generic chatbot. It is a research assistant built from scratch with spec-driven TDD (67 specs, 1,845 tests) that:

- **Searches multiple sources in parallel** — knowledge base, arXiv API, and web (DuckDuckGo) simultaneously
- **Always cites papers** with title, authors, year, and arXiv link — no hallucinated references
- **Uses agentic RAG** — LangGraph agents with a 4-stage retrieval pipeline (query expansion → hybrid search → re-ranking → parent expansion)
- **Routes to the right LLM** — Gemini for research, Claude for code, Ollama for local dev, with automatic fallback
- **Admits knowledge gaps** — when no relevant papers exist in the knowledge base, it says so honestly

```
You: "What are the key innovations in transformer architectures?"

PaperAlchemy:
  Transformers replaced recurrence with multi-head self-attention,
  enabling parallel sequence processing [1]. BERT later introduced
  bidirectional pre-training for transfer learning [2]...

  Sources:
  1. Attention Is All You Need — Vaswani et al., 2017
     https://arxiv.org/abs/1706.03762
  2. BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al., 2018
     https://arxiv.org/abs/1810.04805
```

---

## Key Capabilities

### Citation-First Research Q&A
Every response includes inline `[1][2]` references linked to arXiv. The system uses multi-source retrieval (KB + arXiv API + DuckDuckGo) in parallel, with conversational follow-ups that resolve coreference and re-retrieve context every turn. Session memory is Redis-backed with a sliding window of 20 messages.

### 4-Stage Advanced RAG Pipeline

| Stage | Method | Result |
|-------|--------|--------|
| **Query Expansion** | Multi-query (3-5 variations) + HyDE (hypothetical doc embeddings) | Expanded query set |
| **Hybrid Retrieval** | BM25 + KNN vector search + RRF fusion | Top 20 chunks |
| **Re-ranking** | Cross-encoder (ms-marco-MiniLM-L-12-v2) | Top 5 most relevant |
| **Parent Expansion** | Retrieve full parent sections for richer context | 5 context windows |

### Agentic RAG (LangGraph)

```
Query → [Guardrail] → [Retrieve] → [Grade Documents]
                             ↑              |
                             |         relevant?
                        [Rewrite] ← no (max 3)
                                              |
                        [Web Search] ← KB insufficient
                                              |
                                             yes
                                              ↓
                                     [Generate Answer]
                                     with citations [1][2]
```

Nodes: Guardrail (domain scoring 0-100), Retrieve, Grade (binary relevance), Rewrite (synonym expansion), Web Search (DuckDuckGo fallback), Generate (citation-enforcing). Specialized agents: Summarizer, Fact-Checker, Trend Analyzer, Citation Tracker.

### Paper Upload & AI Analysis
Upload PDFs (50MB limit, magic byte validation) through a full pipeline: parse → chunk → embed → index → analyze. Get structured AI summaries, key highlights, methodology deep-dives, and side-by-side comparison of 2-5 papers.

### Multi-Provider LLM Routing

| Provider | Use Case | Features |
|----------|----------|----------|
| **Google Gemini** | Research Q&A, summarization | `google.genai` SDK, configurable model |
| **Anthropic Claude** | Code generation | `anthropic.AsyncAnthropic`, Opus/Sonnet/Haiku |
| **Ollama** | Local development | No API key, free inference |

Task-based routing across 6 task types (RESEARCH_QA, CODE_GENERATION, SUMMARIZATION, GRADING, QUERY_REWRITE, GENERAL) with configurable fallback chains and per-provider cost tracking.

### Modern Frontend (Next.js 16 + React 19)

| Area | Features |
|------|----------|
| **Landing** | Animated mesh gradient hero, feature grid, scroll-triggered stats, use cases |
| **Search** | Dual-mode (KB + arXiv), autocomplete, category chips, filter pills, shimmer skeletons |
| **Chat** | SSE streaming, Markdown/GFM/syntax highlighting, Mermaid diagrams, follow-up chips, citation cards |
| **Upload** | Drag-and-drop, progress tracking, tabbed analysis results |
| **Papers** | Full sections view, on-demand AI analysis, APA citation copy, filterable list |
| **Dashboard** | Stats cards, category/timeline charts, hot papers, trending topics |
| **Navigation** | Sidebar + Cmd+K command palette + keyboard shortcuts + mobile bottom nav |
| **Design** | Indigo/violet oklch palette, glassmorphism, Inter + Plus Jakarta Sans, 6-tier shadows |
| **Mobile** | Bottom nav, swipe gestures, pull-to-refresh, 44px touch targets, clamp() typography |
| **Auth** | Login/signup/forgot-password (infrastructure ready), Zustand store, JWT interceptor |

### Knowledge Base & Ingestion
Automated arXiv ingestion via Airflow DAGs (Mon-Fri 6am UTC). Section-aware PDF parsing with Docling + PyMuPDF fallback. Parent-child chunking (200w chunks for precision, 600w parents for context).

### Database & Migrations
PostgreSQL 16 with async SQLAlchemy 2.0, Alembic migrations, UUID PKs, JSONB columns. Paper and Collection models with many-to-many relationships.

### Testing: 1,845 Tests

| Suite | Count | Framework |
|-------|-------|-----------|
| Backend | ~1,147 | pytest + pytest-asyncio |
| Frontend | ~698 | Vitest + React Testing Library |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.12 + FastAPI | Async API, auto-docs, SSE streaming |
| **Frontend** | Next.js 16 + React 19 + TypeScript + Tailwind v4 + shadcn/ui | App Router, RSC, streaming |
| **Database** | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic | Async ORM, migrations, JSONB |
| **Search** | OpenSearch 2.19 | BM25 + KNN hybrid in one engine |
| **Embeddings** | Jina AI v3 (1024-dim) | Free API, high quality |
| **Re-ranking** | Cross-encoder (ms-marco-MiniLM-L-12-v2) | Local second-stage scoring |
| **LLM (cloud)** | Google Gemini (gemini-2.5-flash) | Research Q&A, summarization |
| **LLM (code)** | Anthropic Claude (claude-sonnet-4) | Code generation |
| **LLM (local)** | Ollama (llama3.2:1b) | Free local inference |
| **Agents** | LangGraph 0.2+ | State-machine agent graphs |
| **Web Search** | DuckDuckGo (ddgs) | Fallback when KB insufficient |
| **Caching** | Redis 7 | Response + session cache, 24h TTL |
| **PDF Parsing** | Docling 2.43+ / PyMuPDF | Section-aware scientific PDFs |
| **Observability** | Langfuse v3 | LLM tracing + cost tracking |
| **Orchestration** | Apache Airflow 2.10 | Scheduled data ingestion |
| **Object Storage** | MinIO | PDFs, artifacts (dev) |
| **Testing** | pytest (1,147) + Vitest (698) | Full-stack TDD |
| **Cloud** | GCP Cloud Run | Scale to zero, free tier |

---

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="System Architecture" width="95%"/>
</p>

### System Layers

```
Frontend          Next.js 16 + React 19 (TypeScript, Tailwind v4, shadcn/ui)
    |
    | HTTPS / SSE
    v
API Gateway       FastAPI :8000 — 8 routers, DI, validation, middleware
    |
    v
Service Layer     Agentic RAG (LangGraph) + 4-stage retrieval + 3 LLM providers
    |
    v
Data Layer        PostgreSQL · OpenSearch · Redis · MinIO · Jina Embeddings
    |
    v
External          arXiv API · DuckDuckGo · Airflow · Langfuse · GCP
```

### RAG Flow (Every Research Question)

```
Query → [Multi-Source Parallel Retrieval]
              |           |            |
              v           v            v
          [KB Search]  [arXiv API]  [Web Search]
              |           |            |
              +-----merge + dedupe-----+
                          |
                    [Re-rank top 5]
                          |
                   [Generate Answer]
                   with citations [1][2]
```

---

## Getting Started

### Prerequisites
- **Docker Desktop** (4GB+ RAM)
- **Python 3.12+** with [UV](https://docs.astral.sh/uv/)
- **Node.js 20+** with [pnpm](https://pnpm.io/)
- **Ollama** (for local LLM)

### Quick Start

```bash
# 1. Clone & configure
git clone https://github.com/nishantgaurav23/PaperAlchemy.git
cd PaperAlchemy
cp .env.example .env          # Add your JINA_API_KEY, GEMINI__API_KEY

# 2. Start infrastructure
make start                     # PostgreSQL, OpenSearch, Redis, MinIO
make health                    # Wait for healthy services

# 3. Setup backend
ollama pull llama3.2           # First time only
make setup                     # uv sync
make db-upgrade                # Alembic migrations

# 4. Start API
make dev                       # uvicorn on :8000

# 5. Start frontend (new terminal)
cd frontend && pnpm install && pnpm dev   # Next.js on :3000
```

### Key URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| OpenSearch Dashboards | http://localhost:5602 |
| Airflow | http://localhost:8080 |
| pgAdmin | http://localhost:5050 |
| Langfuse | http://localhost:3001 |

---

## API Reference

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ping` | Health check with version |

### Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Hybrid search (BM25 + KNN + RRF) |
| `POST` | `/api/v1/search/arxiv` | Live arXiv API search |
| `GET` | `/api/v1/search/health` | OpenSearch health check |

### Research Q&A
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ask` | Single-turn question (JSON or SSE) |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Multi-turn with session management |
| `GET` | `/api/v1/chat/sessions/{id}/history` | Conversation history |
| `DELETE` | `/api/v1/chat/sessions/{id}` | Clear session |

### Papers
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/papers` | List (filter by query, category) |
| `GET` | `/api/v1/papers/{id}` | Get by UUID |
| `GET` | `/api/v1/papers/by-arxiv/{arxiv_id}` | Get by arXiv ID |

### Upload & Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | PDF → parse → chunk → embed → index → analyze |
| `POST` | `/api/v1/papers/{id}/summary` | AI summary |
| `POST` | `/api/v1/papers/{id}/highlights` | Key highlights |
| `POST` | `/api/v1/papers/{id}/methodology` | Methodology analysis |
| `POST` | `/api/v1/papers/compare` | Compare 2-5 papers |

### Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest/fetch` | Fetch from arXiv → full pipeline |
| `POST` | `/api/v1/ingest/reparse` | Re-parse pending/failed papers |

### Collections
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/collections` | List collections |
| `POST` | `/api/v1/collections` | Create collection |
| `GET/PUT/DELETE` | `/api/v1/collections/{id}` | CRUD |
| `POST` | `/api/v1/collections/{id}/papers` | Add paper |
| `DELETE` | `/api/v1/collections/{id}/papers/{paper_id}` | Remove paper |

Interactive docs at `/docs` when running.

---

## Development

### Commands

```bash
make start          # Start Docker services
make stop           # Stop Docker services
make dev            # API server (hot reload)
make setup          # Install Python deps (uv sync)
make test           # Run all backend tests
make test-cov       # Tests with coverage report
make lint           # Ruff lint + check
make format         # Ruff format
make health         # Health check services
make logs           # Tail service logs
make clean          # Remove containers + volumes
make db-migrate     # Generate Alembic migration
make db-upgrade     # Apply migrations
make db-downgrade   # Rollback last migration
```

### Spec-Driven Development

Every feature follows a spec lifecycle with TDD:

```
/start-spec-dev S{x}.{y} {slug}    # Full: create → deps → implement → verify
/create-spec S{x}.{y} {slug}       # Generate spec.md + checklist.md
/check-spec-deps S{x}.{y}          # Verify prerequisites
/implement-spec S{x}.{y}           # TDD: Red → Green → Refactor
/verify-spec S{x}.{y}              # Audit + test verification
```

### Running Tests

```bash
# Backend (~1,147 tests)
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/unit/test_rag_chain.py -v

# Frontend (~698 tests)
cd frontend && pnpm test
```

### Project Structure

```
PaperAlchemy/
├── src/                         # Python backend
│   ├── main.py                  # FastAPI app with lifespan
│   ├── config.py                # 15+ Pydantic settings classes
│   ├── dependency.py            # Annotated[] DI singletons
│   ├── routers/                 # search, chat, upload, analysis, papers, collections, ingest
│   ├── models/                  # SQLAlchemy ORM (Paper, Collection)
│   ├── repositories/            # Data access layer
│   ├── schemas/                 # Pydantic request/response models
│   └── services/
│       ├── agents/              # LangGraph agentic RAG + specialized agents
│       ├── rag/                 # Multi-source RAG chain + citation enforcement
│       ├── retrieval/           # HyDE, multi-query, unified pipeline
│       ├── llm/                 # Gemini + Claude + Ollama + LLMRouter
│       ├── analysis/            # Summarizer, Highlights, Methodology, Comparator
│       ├── chat/                # Session memory + follow-up detection
│       ├── opensearch/          # Hybrid search client
│       ├── upload/              # PDF upload orchestration
│       ├── web_search/          # DuckDuckGo fallback
│       ├── arxiv/               # arXiv API client
│       ├── pdf_parser/          # Docling + PyMuPDF fallback
│       ├── embeddings/          # Jina AI client
│       ├── reranking/           # Cross-encoder
│       ├── indexing/            # Text chunking + parent-child
│       ├── cache/               # Redis caching
│       └── langfuse/            # LLM tracing
├── frontend/                    # Next.js 16 + React 19
│   └── src/
│       ├── app/                 # Pages: landing, search, chat, upload, papers, dashboard, auth
│       ├── components/          # landing, layout, chat, search, paper, upload, dashboard, ui
│       ├── lib/                 # API client, auth store, hooks
│       └── types/               # TypeScript types
├── tests/                       # ~1,147 pytest tests
├── specs/                       # 67 spec folders (spec.md + checklist.md)
├── alembic/                     # Database migrations
├── airflow/dags/                # arXiv ingestion DAG
├── docs/                        # Architecture SVG, ops guide, spec summaries
├── compose.yml                  # Docker: PostgreSQL, OpenSearch, Redis, MinIO, Ollama, Airflow, Langfuse
└── Makefile                     # Dev commands
```

---

## Roadmap

**67/168 specs completed** across 13 phases:

| Phase | Name | Specs | Status |
|-------|------|-------|--------|
| P1 | Project Foundation | 4/4 | Done |
| P2 | Backend Core | 4/4 | Done |
| P3 | Data Layer | 4/4 | Done |
| P4 | Search & Retrieval | 4/4 | Done |
| P4b | Advanced RAG | 5/5 | Done |
| P5 | RAG Pipeline | 5/5 | Done |
| P6 | Agent System | 8/8 | Done |
| P7 | Chatbot & Conversations | 3/3 | Done |
| P8 | Paper Upload & Analysis | 5/5 | Done |
| P9 | Frontend (Next.js) | 9/9 | Done |
| P9b | Platform Foundation | 7/7 | Done |
| P5b | Extended LLM Providers | 2/2 | Done |
| P13 | UI Enhancement & Design | 7/7 | Done |
| P10 | Evaluation Framework | 0/5 | Next |
| P11 | Observability & Deploy | 0/5 | Planned |
| P12 | Telegram Bot | 0/4 | Planned |
| P14–P23 | Advanced Features | 0/82 | Planned |

Full details in [roadmap.md](roadmap.md).

---

## Environment Variables

```env
# Database
POSTGRES__HOST=localhost
POSTGRES__PORT=5433
POSTGRES__USER=paperalchemy
POSTGRES__PASSWORD=paperalchemy_secret
POSTGRES__DB=paperalchemy

# Search
OPENSEARCH__HOST=http://localhost:9201

# LLM — Multi-Provider
GEMINI__API_KEY=your-key
GEMINI__MODEL=gemini-2.5-flash
ANTHROPIC__API_KEY=your-key
ANTHROPIC__MODEL=claude-sonnet-4-20250514
OLLAMA__HOST=localhost
OLLAMA__MODEL=llama3.2:1b

# LLM Routing
LLM_ROUTING__RESEARCH_QA=gemini
LLM_ROUTING__CODE_GENERATION=anthropic
LLM_ROUTING__FALLBACK_ORDER=gemini,anthropic,ollama

# Embeddings & Re-ranking
JINA_API_KEY=your-key
RERANKER__MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER__TOP_K=5

# Cache
REDIS__HOST=localhost
REDIS__PORT=6380
REDIS__TTL_HOURS=24
```

See `.env.example` for all options.

---

## Deployment

```
GCP Cloud Run
├── Cloud Run (API)          scale 0→3, 1GB RAM
├── Cloud Run (Frontend)     scale 0→2, 512MB RAM
├── Cloud SQL (PostgreSQL)   smallest instance
├── Upstash Redis            free tier
├── Cloud Storage            PDFs, artifacts (replaces MinIO)
└── Secret Manager           API keys, credentials

External Services
├── Gemini API               free tier (15 RPM)
├── Anthropic Claude API     code generation
├── Jina Embeddings API      free tier (1M tokens/mo)
└── OpenSearch               self-hosted or Aiven
```

**Estimated cost: $0-7/mo** with free tier usage.

---

## Author

**Nishant Gaurav** — [@nishantgaurav23](https://github.com/nishantgaurav23)

## Acknowledgments

Architecture inspired by [arxiv-paper-curator](https://github.com/jamwithai/arxiv-paper-curator) by [Jam With AI](https://jamwithai.substack.com/). Rebuilt from scratch with spec-driven TDD, advanced RAG, and a modern Next.js frontend.

## License

This project is for educational and research purposes.

---

<p align="center">
  <sub>Every answer grounded in real papers. Every citation linked to arXiv.</sub>
  <br/>
  <sub>Built with FastAPI + LangGraph + Next.js + OpenSearch + Gemini + Claude</sub>
</p>
