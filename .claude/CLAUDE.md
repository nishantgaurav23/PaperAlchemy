# PaperAlchemy — Claude Code Context

## Project Summary
PaperAlchemy is an **AI Research Assistant** that answers research questions by searching its knowledge base of academic papers, using agents to retrieve and synthesize information, and **always citing sources with paper titles, authors, and arXiv links**. It is NOT a generic chatbot — every answer must be grounded in real papers.

## Core Identity: Research Assistant
- **Every answer MUST cite papers** — title, authors, year, and arXiv link (e.g., `https://arxiv.org/abs/1706.03762`)
- **Knowledge Base first** — agents always search the indexed paper collection before answering
- **Tool calls are mandatory** — the agent system MUST invoke retrieval tools (hybrid search, knowledge base lookup) for every research question. Never answer from LLM memory alone.
- **"I don't have papers on that topic"** — if no relevant papers are found in the knowledge base, say so honestly rather than hallucinating citations
- **Response format**: Answer + inline citations + source list with links at the bottom

### Example Response Format
```
Transformers are a neural network architecture based on self-attention mechanisms,
introduced by Vaswani et al. [1]. The key innovation is replacing recurrence with
multi-head attention, allowing parallel processing of sequences...

**Sources:**
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
```

## Development Strategy: Clean Slate
- Building from scratch with spec-driven TDD — NOT reusing old code
- Old code moved to `_reference/` (gitignored) — study patterns, don't copy blindly
- Reference: `_reference/src/services/`, `_reference/src/routers/`, `_reference/src/config.py`
- Reference notebooks: `_reference/notebooks/week1-7/`
- Configs kept in root: `.env`, `.env.example`, `.env.test`, `compose.yml`, `Dockerfile`, `Makefile`, `pyproject.toml`

## Key Rules
- **NEVER** answer research questions without searching the knowledge base first
- **ALWAYS** cite papers with title, authors, and arXiv link in every response
- **ALWAYS** use agent tool calls (retrieval, search) before generating answers
- **NEVER** write code without a spec (spec-driven development)
- **ALWAYS** write tests FIRST (TDD: Red → Green → Refactor)
- **NEVER** hardcode secrets — all config via environment variables
- **ALWAYS** use async/await for I/O operations
- **NEVER** commit .env files or credentials
- **ALWAYS** update roadmap.md status after completing a spec
- **ALWAYS** append a spec summary to `docs/spec-summaries.md` after verification passes
- **ALWAYS** mock external services in tests (arXiv, Jina, Ollama, Gemini, Cohere)
- **NEVER** skip the checklist — update progressively as you implement
- **ALWAYS** ensure each spec is testable via UI (Next.js frontend)

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Backend | Python 3.12 + FastAPI | Async, auto-docs |
| Frontend | Next.js 16 + React 19 + TypeScript + Tailwind v4 + shadcn/ui | App Router |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic | Async engine, migrations |
| Search | OpenSearch 2.19 | BM25 + KNN hybrid |
| Embeddings | Jina AI v3 (1024-dim) | Free API |
| Re-ranking | Cross-encoder (ms-marco-MiniLM-L-12-v2) | Second-stage relevance scoring |
| LLM (local) | Ollama (llama3.2:1b) | Dev only |
| LLM (cloud) | Google Gemini (gemini-2.5-flash) | Research Q&A, summarization |
| LLM (code) | Anthropic Claude (claude-sonnet-4) | Code generation, multi-provider routing |
| Agents | LangGraph 0.2+ | State-machine flows |
| Web Search | DuckDuckGo (ddgs) | Fallback when KB insufficient |
| Caching | Redis 7 | 24h TTL |
| PDF Parsing | Docling 2.43+ / PyMuPDF | Section-aware, fallback |
| Object Storage | MinIO | Audio files, code artifacts (dev) |
| Observability | Langfuse v3 | LLM tracing (optional) |
| Orchestration | Apache Airflow 2.10 | Data ingestion |
| Testing | pytest (~303 tests), Vitest (~693 tests) | Full-stack TDD |
| Linting | Ruff (Python, line-length: 130), ESLint + Prettier (TS) | |
| Package Mgr | UV (Python), pnpm (Node) | Fast |
| Cloud | GCP Cloud Run | Scale to zero |
| Telegram | python-telegram-bot 21+ | Mobile-first research assistant (planned) |
| Evaluation | RAGAS + custom LLM judges | Quality benchmarks (planned) |

## Project Structure

```
PaperAlchemy/
├── .claude/
│   ├── CLAUDE.md                    # This file
│   └── commands/                    # Spec-driven dev commands
├── specs/                           # All spec folders
│   └── spec-S{x}.{y}-{slug}/
│       ├── spec.md                  # Requirements
│       └── checklist.md             # Implementation tracker
├── _reference/                      # Old code (gitignored, study only)
│   ├── src/                         # Week 1-7 implementation
│   ├── notebooks/                   # Learning notebooks
│   ├── airflow/                     # Old DAGs
│   └── gradio_launcher.py          # Old Gradio entry point
├── src/                             # Python backend (CLEAN SLATE)
│   ├── main.py                      # FastAPI entry point
│   ├── config.py                    # Pydantic settings
│   ├── dependency.py                # DI container
│   ├── exceptions.py                # Custom exceptions
│   ├── middlewares.py               # Request logging, CORS
│   ├── routers/                     # API endpoints
│   ├── models/                      # SQLAlchemy ORM
│   ├── repositories/                # Data access layer
│   ├── schemas/                     # Pydantic models
│   ├── services/                    # Business logic
│   │   ├── agents/                  # LangGraph agentic RAG
│   │   │   ├── nodes/              # Individual agent nodes
│   │   │   └── specialized/        # Summarizer, Fact-Checker, etc.
│   │   ├── arxiv/                   # ArXiv API client
│   │   ├── cache/                   # Redis caching
│   │   ├── chat/                    # Conversation memory + follow-up
│   │   ├── embeddings/              # Jina AI client
│   │   ├── reranking/              # Cross-encoder re-ranking
│   │   ├── retrieval/              # Advanced: HyDE, multi-query, pipeline
│   │   ├── indexing/                # Text chunking + parent-child indexing
│   │   ├── langfuse/                # Tracing
│   │   ├── llm/                     # Unified LLM client (Ollama/Gemini)
│   │   ├── rag/                     # RAG chain + citation enforcement
│   │   ├── analysis/               # Paper summary, highlights, comparison
│   │   ├── opensearch/              # Search engine client
│   │   ├── pdf_parser/              # Docling parser
│   │   └── telegram/               # Telegram bot (commands, handlers, notifications)
│   └── db/                          # Database engine + sessions
├── frontend/                        # Next.js 16 + React 19 app
│   ├── src/
│   │   ├── app/                     # App Router pages
│   │   ├── components/              # Reusable UI components
│   │   ├── lib/                     # API client, utils
│   │   └── types/                   # TypeScript types
│   ├── package.json
│   └── tsconfig.json
├── evals/                           # Evaluation framework
│   ├── datasets/                    # Q&A evaluation sets
│   ├── judges/                      # LLM-as-judge evaluators
│   └── metrics/                     # RAGAS + custom metrics
├── notebooks/
│   └── week1-7/                     # REFERENCE: learning notebooks (old)
├── tests/                           # Python tests (pytest)
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── airflow/dags/                    # Data ingestion DAGs
├── data/                            # Cached PDFs
├── docs/
│   ├── ops-guide.md                 # Operations guide (start/stop/deploy)
│   └── spec-summaries.md            # Developer log: what/how/why per completed spec
├── compose.yml                      # Docker Compose (dev)
├── Dockerfile                       # API container
├── pyproject.toml                   # Python deps (UV)
├── roadmap.md                       # Master plan
├── design.md                        # Architecture doc
└── Makefile                         # Dev commands
```

## Spec Folder Convention

```
specs/spec-S{phase}.{number}-{slug}/
  spec.md       ← Requirements, outcomes, TDD notes
  checklist.md  ← Phase-by-phase implementation tracker
```

## Spec-Driven Development Commands

| Command | Input | Action |
|---------|-------|--------|
| `/start-spec-dev` | S{x}.{y} {slug} | **Full lifecycle**: create → check deps → implement (TDD) → verify. Runs all 4 steps automatically. |
| `/create-spec` | S{x}.{y} {slug} | Read roadmap, create spec.md + checklist.md |
| `/check-spec-deps` | S{x}.{y} | Verify prerequisites are "done" |
| `/implement-spec` | S{x}.{y} | Load spec, TDD (Red→Green→Refactor) |
| `/verify-spec` | S{x}.{y} | Run tests, lint, audit outcomes |

### Post-Phase E2E Gate
**After completing the last spec of any phase**, you MUST run a full end-to-end smoke test before moving to the next phase. This ensures new components haven't broken existing flows.

1. **Backend**: `make test` (all unit + integration tests must pass)
2. **Frontend**: `cd frontend && pnpm test` (all Vitest tests must pass)
3. **Lint**: `make lint` (backend) + `cd frontend && pnpm lint` (frontend)
4. **Docker compose up**: Verify services start cleanly (`docker compose up -d` → check health endpoints)
5. **Manual smoke**: Hit key API endpoints (e.g., `/health`, `/api/v1/chat`, `/api/v1/search`) and verify no regressions

If any E2E check fails, **fix before proceeding** — do not start the next phase with broken flows. Log any issues found and fixed in the relevant spec's checklist or in `docs/spec-summaries.md`.

## Code Standards

### Python Backend
- **Async by default**: `async def`, `await`, non-blocking I/O
- **Type hints**: All function signatures typed
- **Pydantic validation**: All API inputs/outputs via Pydantic models
- **Error handling**: Custom exception classes, global handler
- **Dependency injection**: FastAPI `Depends()` pattern
- **Line length**: 130 characters (Ruff)
- **Imports**: `from __future__ import annotations` where needed

### TypeScript Frontend
- **Strict mode**: `strict: true` in tsconfig
- **Server Components**: Default to RSC, `"use client"` only when needed
- **Zod validation**: API response validation
- **Fetch + SSE**: Native fetch for API calls, EventSource for streaming

## Testing Conventions

### Backend (pytest)
- Test files: `tests/unit/test_{module}.py`
- Async tests: `@pytest.mark.asyncio`
- Fixtures: Shared in `conftest.py`
- Mocking: `unittest.mock.AsyncMock` for external services
- Coverage: `pytest-cov` with HTML reports

### Frontend (Vitest)
- Test files: `*.test.ts` / `*.test.tsx` co-located with components
- React Testing Library for component tests
- MSW for API mocking

## Advanced RAG Pipeline (Reference)
The retrieval pipeline uses 4 stages:
1. **Query Expansion**: Multi-query (3-5 variations) + HyDE (hypothetical doc embedding)
2. **Hybrid Retrieval**: BM25 + KNN vector search + RRF fusion → top 20 chunks
3. **Re-ranking**: Cross-encoder (ms-marco-MiniLM-L-12-v2) re-scores top 20 → top 5
4. **Parent Expansion**: Retrieve parent sections for richer LLM context

See `design.md` for the full pipeline diagram.

## Environment Variables (Key)

```
# Database
POSTGRES__HOST, POSTGRES__PORT, POSTGRES__USER, POSTGRES__PASSWORD, POSTGRES__DB

# Search
OPENSEARCH__HOST, OPENSEARCH__PORT

# LLM — Multi-Provider
OLLAMA__HOST, OLLAMA__PORT, OLLAMA__MODEL=llama3.2:1b
GEMINI__API_KEY, GEMINI__MODEL=gemini-2.5-flash
ANTHROPIC__API_KEY, ANTHROPIC__MODEL=claude-sonnet-4-20250514

# LLM Routing
LLM_ROUTING__RESEARCH_QA=gemini, LLM_ROUTING__CODE_GENERATION=anthropic
LLM_ROUTING__FALLBACK_ORDER=gemini,anthropic,ollama

# Re-ranking
RERANKER__MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER__TOP_K=5

# Embeddings
JINA_API_KEY

# Cache
REDIS__HOST, REDIS__PORT, REDIS__TTL_HOURS=24

# Observability
LANGFUSE__PUBLIC_KEY, LANGFUSE__SECRET_KEY, LANGFUSE__HOST

# Telegram Bot
TELEGRAM__BOT_TOKEN, TELEGRAM__WEBHOOK_URL (optional)
```
