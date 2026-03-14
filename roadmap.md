# PaperAlchemy — Roadmap

> Spec-driven, test-driven development roadmap.
> Inspired by arxiv-paper-curator, evolved into a full research intelligence platform.

---

## Development Strategy: Clean Slate + Reference

We are building PaperAlchemy **from scratch** using spec-driven TDD.
The existing Week 1-7 code (13,000+ lines) serves as **reference only** — we DO NOT reuse it directly.

### Why Clean Slate?
- Proper spec coverage for every module
- 100% test coverage from day one
- Clean architecture incorporating advanced RAG from the start
- Citation-first design baked in (not bolted on)
- Gemini integration designed in (not patched)

### Reference Code (in `_reference/`, gitignored)
Old Week 1-7 code moved to `_reference/` — study patterns, don't copy blindly:
- `_reference/src/services/agents/` → agent node patterns
- `_reference/src/services/opensearch/` → OpenSearch query patterns
- `_reference/src/routers/` → FastAPI endpoint patterns
- `_reference/src/config.py` → Pydantic settings patterns
- `_reference/notebooks/week1-7/` → step-by-step learning notebooks
- `_reference/airflow/` → Airflow DAG patterns

### Configs Kept in Root (reusable)
- `.env`, `.env.example`, `.env.test` → API keys, DB creds
- `compose.yml` → Docker service definitions
- `Dockerfile` → Multi-stage build
- `Makefile` → Dev commands
- `pyproject.toml` → Dependencies (will be updated per S1.1)

### Status Flow
```
pending → spec-written → done
```
- `pending`: No spec yet
- `spec-written`: spec.md + checklist.md created
- `done`: Code + tests passing, notebook verified, roadmap updated

### Testing Convention
- **Every backend spec** gets a notebook: `notebooks/specs/S{x}.{y}_{slug}.ipynb`
- **Every backend spec** gets pytest tests: `tests/unit/test_{module}.py`
- **Every frontend spec** gets Vitest tests: `*.test.tsx` co-located
- **Every spec** must be testable via UI (Next.js frontend)

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.12 | Async ecosystem, ML/AI libraries |
| **Backend** | FastAPI + Uvicorn | Async, auto-docs, streaming SSE |
| **Database** | PostgreSQL 16 | Robust relational storage, JSONB support |
| **ORM** | SQLAlchemy 2.0+ | Type-safe async ORM |
| **Search Engine** | OpenSearch 2.19 | BM25 + KNN hybrid search |
| **Vector Embeddings** | Jina AI v3 (1024-dim) | Free API, high quality |
| **Re-ranking** | Cross-encoder (ms-marco-MiniLM-L-12-v2) | Second-stage relevance scoring after retrieval |
| **LLM (Local)** | Ollama (llama3.2:1b) | Free local inference for dev |
| **LLM (Cloud)** | Google Gemini (gemini-2.5-flash) | Research Q&A, summarization |
| **LLM (Code)** | Anthropic Claude (claude-sonnet-4) | Code generation, multi-provider routing |
| **Agent Framework** | LangGraph 0.2+ | State-machine agents, decision graphs |
| **LLM Framework** | LangChain 0.3+ | Chains, tools, prompt templates |
| **Web Search** | DuckDuckGo (ddgs) | Fallback when KB results insufficient |
| **PDF Parsing** | Docling 2.43+ / PyMuPDF | Section-aware scientific PDF parsing |
| **Caching** | Redis 7 | Sub-ms response caching, LRU eviction |
| **Observability** | Langfuse v3 | Per-node LLM tracing, cost tracking |
| **Workflow** | Apache Airflow 2.10 | Scheduled data ingestion DAGs |
| **Frontend** | Next.js 16 + React 19 + TypeScript + Tailwind v4 + shadcn/ui | App Router, SSR, streaming |
| **Object Storage** | MinIO | Audio files, code artifacts (dev) |
| **Testing** | pytest + pytest-asyncio + Vitest | Backend + frontend testing |
| **Linting** | Ruff (Python, line-length: 130), ESLint + Prettier (TS) | Fast, consistent |
| **Containerization** | Docker + Compose | Reproducible multi-service dev |
| **Cloud** | GCP Cloud Run | Scale-to-zero, 2M free requests/month |
| **Evaluation** | RAGAS + custom LLM judges | Research quality benchmarks |
| **Package Manager** | UV (Python), pnpm (Node) | Fast, reliable |

---

## Budget Estimate (GCP Free/Near-Free)

| Service | Cost | Notes |
|---------|------|-------|
| Cloud Run | $0 | 2M requests free, scale to zero |
| Cloud SQL (PostgreSQL) | $0-7/mo | Free trial, then smallest instance |
| Memorystore (Redis) | $0 | Use Upstash free tier (10K cmd/day) |
| Cloud Storage | $0 | 5GB free for PDFs |
| Secret Manager | $0 | 6 active versions free |
| Artifact Registry | $0 | 500MB free |
| Gemini API | $0 | Free tier: 15 RPM |
| Jina Embeddings API | $0 | Free tier: 1M tokens/month |
| **Total** | **$0-7/mo** | |

---

## Phases Overview

| Phase | Name | Spec Count | Key Output | Parallel |
|-------|------|-----------|------------|----------|
| P1 | Project Foundation | 4 | Clean deps, config, Docker, CI | — |
| P2 | Backend Core | 4 | FastAPI app, DB, middleware, DI | — |
| P3 | Data Layer | 4 | Paper model, arXiv client, PDF parser, ingestion | — |
| P4 | Search & Retrieval | 4 | OpenSearch, chunking, embeddings, hybrid search | — |
| P4b | Advanced RAG Retrieval | 5 | Re-ranking, HyDE, multi-query, parent-child, pipeline | — |
| P5 | RAG Pipeline | 5 | LLM client, RAG chain, streaming, caching, citations | — |
| P6 | Agent System | 8 | LangGraph nodes, orchestrator, specialized agents | — |
| P7 | Chatbot & Conversations | 3 | Follow-up Q&A, session memory, chat API | — |
| P8 | Paper Upload & Analysis | 5 | Upload, summary, highlights, methodology, comparison | — |
| P9 | Frontend (Next.js) | 9 | Full UI: search, chat, upload, dashboard, export | ✅ P3+ |
| P9b | Platform Foundation | 7 | Alembic migrations, auth deps, Anthropic SDK, WebSocket, TTS, frontend infra | ✅ P9 done |
| P5b | Extended LLM Providers | 2 | Anthropic Claude provider, unified multi-provider switching | ✅ P9b+ |
| P10 | Evaluation Framework | 5 | Datasets, LLM judges, human eval, RAGAS, dashboard | ✅ P6+ |
| P10b | Extended Evaluation | 4 | Eval for overviews, code gen, audio, recommendations | ✅ P10+, P21+ |
| P11 | Observability, Ops & Deploy | 5 | Tracing, monitoring, ops guide, GCP deploy, CI/CD | — |
| P11b | Platform Ops Extension | 3 | Ops for new services (MinIO, sandbox, TTS), auth monitoring, platform deploy config | ✅ P11+, P9b+ |
| P12 | Telegram Bot | 4 | Mobile-first research assistant via Telegram | ✅ P7+ |
| P12b | Telegram Feature Extensions | 4 | /overview, /trending, /code, /listen commands + community notifications | ✅ P12+, P21+ |
| P3b | Ingestion Pipeline Extensions | 3 | Auto-trigger overview gen, trending updates, recommendation refresh on new papers | ✅ P3+, P16+, P21+ |
| P6b | Agent Inference Optimization | 4 | Parallel grading, per-node timeouts, smart grading bypass, streaming grading | ✅ P6 done |
| P5c | LLM & Cache Optimization | 5 | Semantic caching, embedding cache, multi-tier LLM routing, concurrency control, batch calls | ✅ P5 done |
| P2b | Infrastructure Hardening | 4 | Async OpenSearch, DB pool tuning, service pre-warming, capacity-aware health checks | ✅ P2 done |
| P10c | Quality Assurance & A/B Testing | 3 | Response quality monitoring, A/B model comparison, regression detection | ✅ P10+, P11+ |
| P13 | UI Enhancement & Design Polish | 7 | Premium visual design, animations, landing page, UX refinements | ✅ P9b+ |
| P14 | Social & Community (AlphaXiv-inspired) | 6 | Inline comments, voting, author profiles, discussions, ORCID | ✅ P9b+ |
| P15 | Paper Annotations & Notebooks | 5 | Inline highlights, private notes, research notebook, snippets | ✅ P14+ |
| P16 | Discovery & Intelligence | 6 | Recommendations, weekly digest, citation graph, research timeline, trending feed | ✅ P4+ |
| P17 | Collaboration & Sharing | 4 | Shared collections, team comments, shareable summaries, public profiles | ✅ P14+ |
| P18 | Workflow Integration | 5 | Zotero/Mendeley import, Notion/Obsidian export, API access, browser extension, BibTeX sync | ✅ P15+ |
| P19 | Advanced AI Features | 5 | Multi-paper Q&A, literature review generator, slide deck generation, research gap finder, AI paper assistant | ✅ P6+ |
| P20 | Interactive Labs | 3 | Geo-trends visualization, research playground, paper relationship explorer | ✅ P16+ |
| P21 | AI Paper Blog & Overview | 6 | Auto-generated paper overviews, tabbed paper view, similar papers, resources hub | ✅ P8+ |
| P22 | Paper-to-Code Implementation | 5 | AI feasibility analysis, code generation agent, sandbox execution, GitHub export | ✅ P21+ |
| P23 | AI Audio & Podcast Generation | 5 | Paper-to-podcast with multi-voice dialogue, transcript, audio player, customizable avatars | ✅ P21+ |

**Total: 168 specs**

---

## Phase 1: Project Foundation

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S1.1 | specs/spec-S1.1-dependency-declaration/ | — | pyproject.toml | Clean dependency declaration | UV; include cross-encoder, google-genai, ragas deps | done |
| S1.2 | specs/spec-S1.2-environment-config/ | S1.1 | src/config.py, .env.example | Pydantic settings, env validation | Nested: postgres, opensearch, ollama, gemini, redis, jina, reranker, langfuse, arxiv, chunking, app | done |
| S1.3 | specs/spec-S1.3-docker-infrastructure/ | S1.1 | compose.yml, Dockerfile | Multi-service Docker Compose | PostgreSQL, OpenSearch, Redis, Ollama, Airflow, Langfuse stack | done |
| S1.4 | specs/spec-S1.4-ci-cd-setup/ | S1.1 | .github/workflows/ | GitHub Actions: lint, test, build | Ruff + pytest + Docker build + type check | done |

## Phase 2: Backend Core

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S2.1 | specs/spec-S2.1-fastapi-app-factory/ | S1.2 | src/main.py | App factory with lifespan | Router registration, CORS, startup/shutdown, health check | done |
| S2.2 | specs/spec-S2.2-database-layer/ | S1.2, S1.3 | src/db/ | SQLAlchemy engine + sessions | Connection pooling, health check, async sessions | done |
| S2.3 | specs/spec-S2.3-error-handling/ | S2.1 | src/exceptions.py, src/middlewares.py | Global error handling, request logging | Custom exception classes, structured error responses | done |
| S2.4 | specs/spec-S2.4-dependency-injection/ | S2.1, S2.2 | src/dependency.py | FastAPI Depends() pattern | Typed Annotated[] service injection | done |

## Phase 3: Data Layer

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S3.1 | specs/spec-S3.1-paper-model/ | S2.2 | src/models/paper.py, src/repositories/paper.py | Paper ORM model + repository | UUID PK, JSONB fields, CRUD, upsert, query by date/status | done |
| S3.2 | specs/spec-S3.2-arxiv-client/ | S1.2 | src/services/arxiv/ | ArXiv API client | Rate-limited (3s), retry with backoff, category filter, PDF download | done |
| S3.3 | specs/spec-S3.3-pdf-parser/ | S1.2 | src/services/pdf_parser/ | Docling PDF parser | Section-aware, 30-page limit, 50MB max, OCR optional | done |
| S3.4 | specs/spec-S3.4-ingestion-pipeline/ | S3.1, S3.2, S3.3 | airflow/dags/ | Airflow DAG: fetch → parse → store → index | Mon-Fri 6am UTC, idempotent, 2 retries | done |

## Phase 4: Search & Retrieval

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S4.1 | specs/spec-S4.1-opensearch-client/ | S1.3 | src/services/opensearch/ | OpenSearch client + index config | BM25 + KNN mappings, RRF pipeline, bulk indexing | done |
| S4.2 | specs/spec-S4.2-text-chunker/ | S3.3 | src/services/indexing/text_chunker.py | Section-aware chunking | 600-word chunks, 100 overlap, min 100 words, metadata preserved | done |
| S4.3 | specs/spec-S4.3-embedding-service/ | S1.2 | src/services/embeddings/ | Jina AI embeddings client | 1024-dim, batch (100), retrieval.passage/retrieval.query tasks | done |
| S4.4 | specs/spec-S4.4-hybrid-search/ | S4.1, S4.2, S4.3 | src/routers/search.py, src/schemas/api/search.py | BM25 + KNN + RRF fusion | Unified search endpoint, graceful fallback BM25 on embed failure | done |

## Phase 4b: Advanced RAG Retrieval

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S4b.1 | specs/spec-S4b.1-reranker/ | S4.4 | src/services/reranking/ | Cross-encoder re-ranking | ms-marco-MiniLM-L-12-v2 (local) or Cohere Rerank (cloud); re-score top-20 → return top-5 | done |
| S4b.2 | specs/spec-S4b.2-hyde/ | S4.4, S5.1 | src/services/retrieval/hyde.py | HyDE (Hypothetical Document Embeddings) | LLM generates hypothetical answer → embed → search for similar real docs | done |
| S4b.3 | specs/spec-S4b.3-multi-query/ | S4.4, S5.1 | src/services/retrieval/multi_query.py | Multi-query retrieval | LLM generates 3-5 query variations → parallel search → deduplicate + fuse | done |
| S4b.4 | specs/spec-S4b.4-parent-child-chunks/ | S4.2 | src/services/indexing/parent_child.py | Parent-child chunk retrieval | Index small chunks (200w), retrieve parent section for context window | done |
| S4b.5 | specs/spec-S4b.5-retrieval-pipeline/ | S4b.1-S4b.4 | src/services/retrieval/pipeline.py | Unified advanced retrieval pipeline | Orchestrates: multi-query → hybrid search → re-rank → parent expansion → top-K | done |

## Phase 5: RAG Pipeline

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S5.1 | specs/spec-S5.1-llm-client/ | S1.2 | src/services/llm/ | Unified LLM client (Ollama + Gemini) | Provider abstraction: LLMProvider interface, OllamaProvider, GeminiProvider, model switching | done |
| S5.2 | specs/spec-S5.2-rag-chain/ | S5.1, S4b.5 | src/services/rag/ | RAG pipeline: retrieve → prompt → generate | Uses advanced retrieval pipeline; citation-enforcing prompts | done |
| S5.3 | specs/spec-S5.3-streaming-responses/ | S5.2 | src/routers/ask.py | SSE streaming endpoint | Token-by-token streaming, source metadata at end | done |
| S5.4 | specs/spec-S5.4-response-caching/ | S5.2 | src/services/cache/ | Redis response caching | SHA256 keys, 24h TTL, cache invalidation, graceful fallback | done |
| S5.5 | specs/spec-S5.5-citation-enforcement/ | S5.2 | src/services/rag/citation.py | Citation-first response format | Parse LLM output → validate inline [1],[2] refs → append source list with title, authors, arXiv link | done |

## Phase 6: Agent System

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S6.1 | specs/spec-S6.1-agent-state/ | S5.1 | src/services/agents/state.py, context.py | Agent state schema + runtime context | TypedDict, message history, source tracking, citation list | done |
| S6.2 | specs/spec-S6.2-guardrail-node/ | S6.1 | src/services/agents/nodes/guardrail_node.py | Domain relevance check (0-100) | Reject off-topic queries, structured LLM output | done |
| S6.3 | specs/spec-S6.3-retrieval-node/ | S6.1, S4b.5 | src/services/agents/nodes/retrieve_node.py | Tool-based document retrieval | MANDATORY tool call; uses advanced retrieval pipeline (rerank + HyDE + multi-query) | done |
| S6.4 | specs/spec-S6.4-grading-node/ | S6.1 | src/services/agents/nodes/grade_documents_node.py | Binary relevance scoring | LLM-based grading, threshold 0.5 | done |
| S6.5 | specs/spec-S6.5-rewrite-node/ | S6.1 | src/services/agents/nodes/rewrite_query_node.py | Query optimization (max 3 retries) | Synonym expansion, context refinement, rewrite → re-retrieve loop | done |
| S6.6 | specs/spec-S6.6-generation-node/ | S6.1, S5.5 | src/services/agents/nodes/generate_answer_node.py | Citation-backed answer generation | MUST include inline [1],[2] refs + source list with title, authors, arXiv link | done |
| S6.7 | specs/spec-S6.7-agent-orchestrator/ | S6.2-S6.6 | src/services/agents/agentic_rag.py | LangGraph StateGraph compilation | Decision edges, conditional routing, compiled once at startup | done |
| S6.8 | specs/spec-S6.8-specialized-agents/ | S6.7 | src/services/agents/specialized/ | Summarizer, Fact-Checker, Trend Analyzer, Citation Tracker | Multi-agent collaboration for paper analysis | done |

## Phase 7: Chatbot & Conversations

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S7.1 | specs/spec-S7.1-conversation-memory/ | S5.4 | src/services/chat/memory.py | Session-based conversation history | Redis-backed, sliding window (last 20 messages), 24h TTL | done |
| S7.2 | specs/spec-S7.2-follow-up-handler/ | S7.1, S5.2 | src/services/chat/follow_up.py | Context-aware follow-up Q&A | Coreference resolution ("What about its limitations?" → full query), re-retrieves every time | done |
| S7.3 | specs/spec-S7.3-chat-api/ | S7.1, S7.2 | src/routers/chat.py | Chat endpoint with session management | POST /api/v1/chat, session_id, streaming SSE, citation-backed responses | done |

## Phase 8: Paper Upload & Analysis

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S8.1 | specs/spec-S8.1-pdf-upload/ | S3.3, S2.4 | src/routers/upload.py | PDF upload endpoint | Multipart form, 50MB limit, PDF-only validation, stores + indexes | done |
| S8.2 | specs/spec-S8.2-paper-summary/ | S8.1, S5.1 | src/services/analysis/summarizer.py | AI-generated paper summary | Structured: objective, method, key findings, contribution, limitations | done |
| S8.3 | specs/spec-S8.3-key-highlights/ | S8.1, S5.1 | src/services/analysis/highlights.py | Extract key highlights & insights | Novel contributions, important findings, practical implications | done |
| S8.4 | specs/spec-S8.4-methodology-analysis/ | S8.1, S5.1 | src/services/analysis/methodology.py | Methodology & findings deep-dive | Datasets used, baselines compared, results tables, statistical significance | done |
| S8.5 | specs/spec-S8.5-paper-comparison/ | S8.2 | src/services/analysis/comparator.py | Side-by-side paper comparison | Compare 2+ papers: methods, results, contributions, limitations | done |

## Phase 9: Frontend (Next.js) — PARALLEL from P3+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S9.1 | specs/spec-S9.1-nextjs-setup/ | — | frontend/ | Next.js 16 + React 19 + TS + Tailwind v4 + shadcn/ui | App Router, pnpm, ESLint, Vitest, dark mode | done |
| S9.2 | specs/spec-S9.2-layout-navigation/ | S9.1 | frontend/src/app/ | App shell: sidebar, header, theme toggle | Responsive, collapsible sidebar, breadcrumbs | done |
| S9.3 | specs/spec-S9.3-search-interface/ | S9.2 | frontend/src/app/search/ | Search page with filters & results | Category filter, sort, pagination, paper cards with arXiv links | done |
| S9.4 | specs/spec-S9.4-chat-interface/ | S9.2 | frontend/src/app/chat/ | RAG chatbot with streaming | SSE consumption, follow-up Q&A, message history, citation rendering with clickable arXiv links | done |
| S9.5 | specs/spec-S9.5-paper-upload-ui/ | S9.2 | frontend/src/app/upload/ | Drag-and-drop PDF upload | Progress bar, summary + highlights + methodology display | done |
| S9.6 | specs/spec-S9.6-paper-detail-view/ | S9.2 | frontend/src/app/papers/[id]/ | Full paper view with highlights | Sections, annotations, metadata, arXiv link, related papers | done |
| S9.7 | specs/spec-S9.7-reading-lists/ | S9.2 | frontend/src/app/collections/ | Save & organize papers into collections | CRUD collections, drag-and-drop, share link | done |
| S9.8 | specs/spec-S9.8-trends-dashboard/ | S9.2 | frontend/src/app/dashboard/ | Research trends & analytics | Charts (recharts), trending topics, hot papers, category breakdown | done |
| S9.9 | specs/spec-S9.9-export-citations/ | S9.6 | frontend/src/components/export/ | BibTeX, Markdown, slide export | Copy-to-clipboard, download file | done |

## Phase 9b: Platform Foundation — Bridge to P13-P23

> Critical infrastructure that the existing codebase (P1-P9) is missing but all new phases (P13-P23) require. This phase bridges the research assistant into a full platform. **Must be completed before P13-P14 can start.**

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S9b.1 | specs/spec-S9b.1-alembic-migrations/ | S2.2 | alembic/, src/db/ | Alembic migration setup | Initialize Alembic, auto-generate initial migration from Paper model, configure async driver, add `make db-migrate`, `make db-upgrade`, `make db-downgrade` commands; required for all future models (User, Comment, Vote, Note, Avatar, etc.) | done |
| S9b.2 | specs/spec-S9b.2-platform-dependencies/ | S1.1 | pyproject.toml, .env.example | Platform dependency declaration | Add: anthropic SDK (for P22 code gen), python-jose[cryptography] + passlib[bcrypt] (for P14 auth), websockets (for P14 real-time comments), edge-tts or google-cloud-texttospeech (for P23 audio), python-pptx (for P19 slides); update .env.example with ANTHROPIC__API_KEY, AUTH__SECRET_KEY, AUTH__ALGORITHM, TTS__PROVIDER, TTS__API_KEY | done |
| S9b.3 | specs/spec-S9b.3-frontend-infra-deps/ | S9.1 | frontend/package.json | Frontend infrastructure dependencies | Add: react-markdown + remark-gfm + rehype-highlight (P13 chat markdown), framer-motion (P13 animations), zustand (global state for auth/notifications), react-hot-toast or sonner (toast notifications), cmdk (P13 command palette), react-hook-form + zod (form validation for P14 auth forms); add to Vitest mocks | done |
| S9b.4 | specs/spec-S9b.4-frontend-auth-infra/ | S9b.3 | frontend/src/lib/auth/, frontend/src/components/auth/ | Frontend auth infrastructure | Auth context provider (useAuth hook), token storage (httpOnly cookie or sessionStorage), API client interceptor (auto-attach Bearer token, refresh on 401), ProtectedRoute wrapper component, login/signup/forgot-password page shells (/login, /signup), auth types (User, AuthState, LoginRequest, SignupRequest) | done |
| S9b.5 | specs/spec-S9b.5-frontend-ui-primitives/ | S9b.3 | frontend/src/components/ui/ | Missing UI primitives | Add shadcn/ui components needed by P13-P23: Dialog/Modal, Dropdown Menu, Popover, Tabs, Textarea, Checkbox, Tooltip, Avatar, Toast container, Command (for Cmd+K palette), Sheet (mobile drawers); ensure all have Vitest tests | done |
| S9b.6 | specs/spec-S9b.6-collections-backend/ | S2.4, S3.1 | src/models/collection.py, src/repositories/collection.py, src/routers/collections.py | Collections backend API | Migrate collections from frontend localStorage to backend: Collection model (id, name, description, user_id nullable for now, papers M2M, created_at), CollectionRepository (CRUD, add/remove paper), REST API (GET/POST/PUT/DELETE /api/v1/collections), update frontend to call API instead of localStorage; required for P15, P17 shared collections | done |
| S9b.7 | specs/spec-S9b.7-docker-platform-services/ | S1.3 | compose.yml | Docker services for platform features | Add services: E2B or sandboxed Docker-in-Docker (for P22 code sandbox), MinIO/S3 (for P23 audio file storage, P22 code artifacts); add ANTHROPIC__API_KEY to API service env; add new env vars for TTS, auth secret | done |

## Phase 5b: Extended LLM Providers — AFTER P9b

> Extend the unified LLM client (S5.1) to support Anthropic Claude models, enabling P22 code generation and future multi-provider AI features.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S5b.1 | specs/spec-S5b.1-anthropic-provider/ | S5.1, S9b.2 | src/services/llm/anthropic_provider.py | Anthropic Claude LLM provider | Add AnthropicProvider implementing LLMProvider protocol; supports Claude Opus/Sonnet/Haiku; streaming + non-streaming; tool use support for agentic code gen; config via ANTHROPIC__API_KEY, ANTHROPIC__MODEL; register in factory | done |
| S5b.2 | specs/spec-S5b.2-multi-provider-routing/ | S5b.1 | src/services/llm/router.py | Multi-provider LLM routing | Route different tasks to optimal provider: research Q&A → Gemini, code generation → Claude, local dev → Ollama; configurable per-task routing table; fallback chain (primary fails → try secondary); cost tracking per provider | done |

## Phase 10: Evaluation Framework — PARALLEL from P6+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S10.1 | specs/spec-S10.1-eval-dataset/ | S5.2 | evals/datasets/ | Curated Q&A evaluation dataset | 100+ Q&A triples from indexed papers + SciQ + QASPER subsets | pending |
| S10.2 | specs/spec-S10.2-llm-judge/ | S10.1, S5.1 | evals/judges/llm_judge.py | LLM-as-judge evaluator | Faithfulness, relevance, coherence, citation accuracy scoring via Gemini | pending |
| S10.3 | specs/spec-S10.3-human-eval-ui/ | S10.1, S9.1 | frontend/src/app/eval/ | Human evaluation interface | Side-by-side comparison, Likert scales (1-5), inter-annotator agreement | pending |
| S10.4 | specs/spec-S10.4-quality-metrics/ | S10.2 | evals/metrics/ | RAGAS metrics + custom metrics | Answer correctness, context precision/recall, hallucination rate, citation accuracy | pending |
| S10.5 | specs/spec-S10.5-benchmark-dashboard/ | S10.4, S9.1 | frontend/src/app/benchmarks/ | Evaluation results dashboard | Model comparison, metric trends over time, regression alerts | pending |

## Phase 10b: Extended Evaluation — AFTER P10+, P21+

> Evaluate the quality of new AI features: paper overviews, code generation, audio/podcast scripts, and recommendations.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S10b.1 | specs/spec-S10b.1-overview-eval/ | S10.2, S21.1 | evals/judges/overview_judge.py | Paper overview quality evaluation | LLM judge for overview quality: factual accuracy vs paper, completeness (all key sections covered), readability, technical depth; compare generated overview against paper abstract + sections; score 1-5 on faithfulness, coverage, clarity | pending |
| S10b.2 | specs/spec-S10b.2-code-gen-eval/ | S10.2, S22.2 | evals/judges/code_gen_judge.py | Code generation quality evaluation | Evaluate generated code: does it compile/run? Does it match the paper's algorithm? Test coverage? Code quality (linting)? Compare against reference implementations if available; metrics: compilation rate, test pass rate, algorithmic fidelity | pending |
| S10b.3 | specs/spec-S10b.3-audio-eval/ | S10.2, S23.1 | evals/judges/audio_judge.py | Podcast script quality evaluation | Evaluate generated scripts: factual accuracy, conversational naturalness, concept explanation quality, coverage of key paper points; score dialogue flow, technical accuracy, engagement quality; does the Q&A format address likely reader questions? | pending |
| S10b.4 | specs/spec-S10b.4-recommendation-eval/ | S10.2, S16.1 | evals/judges/recommendation_judge.py | Recommendation quality evaluation | Evaluate recommendation engine: precision@k (are recommended papers relevant?), diversity (not all same topic), novelty (not just popular papers), serendipity; A/B test framework for comparing recommendation strategies | pending |

## Phase 11: Observability, Ops & Deployment

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S11.1 | specs/spec-S11.1-langfuse-tracing/ | S5.2 | src/services/langfuse/ | Per-node LLM tracing | Cost tracking, latency analysis, per-request traces with spans | pending |
| S11.2 | specs/spec-S11.2-monitoring/ | S2.1 | src/services/monitoring/ | Health checks, metrics, alerts | Prometheus-compatible /metrics endpoint | pending |
| S11.3 | specs/spec-S11.3-ops-documentation/ | S1.3 | docs/ops-guide.md | Operations guide | Start/stop, Docker commands, Makefile, env setup, troubleshooting, architecture | pending |
| S11.4 | specs/spec-S11.4-gcp-deployment/ | ALL | deploy/ | GCP Cloud Run deployment | Terraform/gcloud, auto-scaling, secrets | pending |
| S11.5 | specs/spec-S11.5-production-ci-cd/ | S11.4, S1.4 | .github/workflows/ | Full CI/CD: test → build → deploy | Staging + production environments | pending |

## Phase 11b: Platform Ops Extension — AFTER P11+, P9b+

> Extend ops and deployment to cover new platform services: MinIO, sandbox, TTS, auth, and multi-provider LLM monitoring.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S11b.1 | specs/spec-S11b.1-platform-monitoring/ | S11.2, S9b.7 | src/services/monitoring/ | Platform service health monitoring | Extend health checks for: MinIO/S3 connectivity, sandbox service status, TTS provider health, Anthropic API quota/usage, auth service (JWT issuer), WebSocket connection count; add to /metrics endpoint | pending |
| S11b.2 | specs/spec-S11b.2-platform-ops-guide/ | S11.3, S9b.7 | docs/ops-guide.md | Extended ops documentation | Add sections for: MinIO admin, sandbox lifecycle management, TTS provider switching, auth troubleshooting (JWT debugging, OAuth callback setup), Anthropic API key rotation, audio file storage cleanup, code artifact retention policies | pending |
| S11b.3 | specs/spec-S11b.3-platform-deploy/ | S11.4, S9b.7 | deploy/ | Platform deployment config | Extend GCP deploy for: Cloud Storage (replaces MinIO in prod), Cloud Run jobs (for code sandbox), Secret Manager (auth keys, Anthropic key, TTS key), CDN for audio files, WebSocket via Cloud Run streaming; update Terraform/gcloud scripts | pending |

## Phase 12: Telegram Bot — PARALLEL from P7+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S12.1 | specs/spec-S12.1-telegram-bot-setup/ | S1.2 | src/services/telegram/ | Telegram bot core setup | python-telegram-bot, webhook/polling modes, /start /help commands, bot token config | pending |
| S12.2 | specs/spec-S12.2-telegram-rag-integration/ | S12.1, S7.3 | src/services/telegram/handlers/ | RAG-powered message handling | /ask command → agentic RAG pipeline, citation-backed answers in Telegram, conversation context per chat_id | pending |
| S12.3 | specs/spec-S12.3-telegram-search/ | S12.1, S4.4 | src/services/telegram/handlers/ | Paper search via Telegram | /search command, inline query mode, paper result cards with arXiv links, pagination via callback buttons | pending |
| S12.4 | specs/spec-S12.4-telegram-notifications/ | S12.1, S3.4 | src/services/telegram/notifications.py | Paper alert notifications | New paper alerts (daily digest), topic subscriptions per user, scheduled via Airflow or background task | pending |

## Phase 12b: Telegram Feature Extensions — AFTER P12+, P21+

> Extend the Telegram bot with commands for new platform features: overviews, trending, code, audio, and community notifications.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S12b.1 | specs/spec-S12b.1-telegram-overview/ | S12.1, S21.1 | src/services/telegram/handlers/overview.py | /overview command | `/overview <arxiv_id>` → sends AI-generated paper overview (condensed for Telegram, 500-word summary), links to full overview on web; inline button "Read full overview" | pending |
| S12b.2 | specs/spec-S12b.2-telegram-trending/ | S12.1, S16.5 | src/services/telegram/handlers/trending.py | /trending command | `/trending [category]` → shows top 5 hot/trending papers with vote counts, visit counts; inline buttons for category switching (cs.AI, cs.CL, cs.CV); paginate with "Next 5" button | pending |
| S12b.3 | specs/spec-S12b.3-telegram-code/ | S12.1, S22.2 | src/services/telegram/handlers/code.py | /code command | `/code <arxiv_id>` → triggers code generation for paper, sends feasibility assessment first, then "Generate code?" confirmation button; delivers code as downloadable file or GitHub Gist link | pending |
| S12b.4 | specs/spec-S12b.4-telegram-community/ | S12.1, S14.3, S14.4 | src/services/telegram/handlers/community.py | Community notification integration | Push notifications for: new comments on papers you follow, votes on your comments, new papers from followed authors, weekly digest summary; configurable notification preferences per user | pending |

## Phase 3b: Ingestion Pipeline Extensions — AFTER P3+, P16+, P21+

> Extend the ingestion pipeline to auto-trigger new platform features when papers are ingested.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S3b.1 | specs/spec-S3b.1-auto-overview-generation/ | S3.4, S21.1 | airflow/dags/, src/services/blog/ | Auto-generate overviews on ingest | After paper is fetched + parsed + indexed, trigger overview generation as background task; queue-based (Redis) to avoid overloading LLM; batch process during off-peak hours; mark papers with overview_status (pending/generated/failed) | pending |
| S3b.2 | specs/spec-S3b.2-trending-score-update/ | S3.4, S16.5 | airflow/dags/, src/services/discovery/ | Update trending scores on ingest | Recalculate trending/hot scores when new papers arrive; update category counts for dashboard; refresh recommendation embeddings; run as Airflow task after ingestion DAG completes | pending |
| S3b.3 | specs/spec-S3b.3-resource-auto-detection/ | S3.4, S21.5 | src/services/blog/resources.py | Auto-detect paper resources on ingest | After paper is parsed, scan for: GitHub URLs in abstract/content, project page links, dataset references (HuggingFace, Kaggle), video links (YouTube); auto-populate resources tab; run as post-ingestion task | pending |

## Phase 6b: Agent Inference Optimization — AFTER P6 done

> Optimize the agent pipeline for lower latency and smarter resource usage. Reduces end-to-end response time from ~7s to ~3-4s.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S6b.1 | specs/spec-S6b.1-parallel-grading/ | S6.4 | src/services/agents/nodes/grade_documents_node.py | Parallel document grading | Replace sequential grading loop with `asyncio.gather()` — grade all 5 docs concurrently instead of sequentially; reduces grading time from ~5s to ~1s; configurable max_concurrent_grades (default 5); error isolation per doc (one failure doesn't block others) | pending |
| S6b.2 | specs/spec-S6b.2-node-timeouts/ | S6.7 | src/services/agents/nodes/, src/services/agents/agentic_rag.py | Per-node timeouts & circuit breakers | Wrap each agent node with `asyncio.wait_for(node(), timeout=N)`; configurable per-node: guardrail=5s, grading=10s, generation=30s, rewrite=10s; circuit breaker pattern: if a node fails N times in M seconds, skip it with fallback; track timeout metrics for observability | pending |
| S6b.3 | specs/spec-S6b.3-smart-grading/ | S6b.1, S4b.1 | src/services/agents/nodes/grade_documents_node.py | Smart grading bypass using reranker scores | Skip LLM grading for high-confidence docs: if cross-encoder reranker score > 0.8, auto-grade as relevant (skip LLM call); if score < 0.2, auto-grade as irrelevant; only LLM-grade borderline docs (0.2-0.8); reduces LLM calls from 5 to ~1-2 per request; configurable thresholds | pending |
| S6b.4 | specs/spec-S6b.4-streaming-grading/ | S6b.1, S6.6 | src/services/agents/agentic_rag.py | Early generation with streaming grading | Start answer generation as soon as first relevant doc is found, don't wait for all grading to complete; use `asyncio.as_completed()` to process grading results; feed additional relevant docs into generation context as they're graded; reduces time-to-first-token by ~2-3s | pending |

## Phase 5c: LLM & Cache Optimization — AFTER P5 done

> Optimize LLM usage, reduce redundant calls, and add intelligent caching for faster repeat/similar queries.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S5c.1 | specs/spec-S5c.1-semantic-caching/ | S5.4, S4.3 | src/services/cache/semantic_cache.py | Semantic caching with embedding similarity | Replace exact-match SHA256 cache with embedding-based similarity lookup; embed incoming query via Jina → cosine similarity against cached query embeddings → if similarity > 0.92, return cached response; store cache entries in Redis with TTL + OpenSearch vector index for lookup; "What are transformers?" and "Explain transformer architecture" hit same cache; configurable similarity threshold | pending |
| S5c.2 | specs/spec-S5c.2-embedding-cache/ | S5.4, S4.3 | src/services/cache/embedding_cache.py | Query embedding cache (Redis) | Cache computed embeddings in Redis (key: SHA256 of text, value: embedding vector); TTL: 7 days; avoids redundant Jina API calls for repeated/similar sub-queries; batch-aware: cache individual embeddings from batch calls; reduces embedding latency from ~300ms to ~1ms on cache hit; track hit/miss rate | pending |
| S5c.3 | specs/spec-S5c.3-multi-tier-llm/ | S5.1, S5b.2 | src/services/llm/tier_router.py | Multi-tier LLM routing by task complexity | Route to different model tiers based on task: guardrail/grading → lightweight model (gemini-2.0-flash-lite or llama3.2:1b); query rewrite/HyDE → mid-tier (gemini-2.0-flash); answer generation → full model (gemini-2.0-flash or Claude); configurable task→model mapping; reduces cost and latency for simple tasks by 50-70% | pending |
| S5c.4 | specs/spec-S5c.4-llm-concurrency/ | S5.1 | src/services/llm/concurrency.py | LLM call concurrency control | Add `asyncio.Semaphore(max_concurrent=10)` wrapper around all LLM calls; prevents overwhelming LLM providers under load; configurable per-provider limits (Gemini: 10, Ollama: 1, Claude: 5); request queue with backpressure — return 429 when queue full; track queue depth and wait times for monitoring | pending |
| S5c.5 | specs/spec-S5c.5-batch-llm-calls/ | S5.1, S6b.1 | src/services/llm/batch.py | Batch LLM calls for grading | Send multiple grading prompts as a single batched request where provider supports it; Gemini batch API: send 5 grading prompts in one call; reduces HTTP overhead and API rate consumption; fallback to parallel individual calls if batch not supported; configurable batch_size limit | pending |

## Phase 2b: Infrastructure Hardening — AFTER P2 done

> Harden the infrastructure layer for production-grade concurrency, connection management, and operational readiness.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S2b.1 | specs/spec-S2b.1-async-opensearch/ | S4.1 | src/services/opensearch/client.py | Async OpenSearch client migration | Replace synchronous `opensearch-py` with `opensearch-py[async]` (AsyncOpenSearch); eliminates blocking calls in async event loop; connection pooling with configurable pool size (default 20); retry logic with exponential backoff; proper async context manager lifecycle | pending |
| S2b.2 | specs/spec-S2b.2-db-pool-tuning/ | S2.2 | src/db/engine.py, src/config.py | Database connection pool tuning | Increase pool_size from 5 to 20; add max_overflow=10; configure pool_timeout=30s; pool_recycle=1800s (30min); pool_pre_ping=True for stale connection detection; add pool size to health endpoint; configurable via env vars: POSTGRES__POOL_SIZE, POSTGRES__MAX_OVERFLOW | pending |
| S2b.3 | specs/spec-S2b.3-service-prewarming/ | S2.1, S4b.1 | src/main.py, src/services/reranking/service.py | Service pre-warming & startup optimization | Load cross-encoder model during FastAPI startup (lifespan), not on first request; pre-warm OpenSearch connection pool; verify Redis connectivity; validate LLM provider health; log startup timing per service; add readiness probe endpoint (/api/v1/ready) distinct from liveness (/api/v1/ping) | pending |
| S2b.4 | specs/spec-S2b.4-capacity-health/ | S2b.1, S2b.2, S5c.4 | src/routers/health.py, src/services/monitoring/ | Capacity-aware health endpoint | Extend health endpoint to report: current concurrent requests, LLM semaphore queue depth, DB pool active/idle connections, OpenSearch pool utilization, Redis connection status, cross-encoder model loaded status; return HTTP 503 when at capacity; load balancer compatible (GCP Cloud Run) | pending |

## Phase 10c: Quality Assurance & A/B Testing — AFTER P10+, P11+

> Continuous quality monitoring, A/B model comparison, and regression detection for production-grade response quality.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S10c.1 | specs/spec-S10c.1-response-quality-monitor/ | S10.4, S11.1 | src/services/monitoring/quality.py | Real-time response quality monitoring | Track per-response metrics: citation count, citation coverage %, retrieval stage timings, LLM token usage, grading pass rate, guardrail scores; aggregate into rolling 1h/24h/7d windows; emit to Langfuse + Prometheus; alert if citation coverage drops below 80% or avg latency exceeds 10s | pending |
| S10c.2 | specs/spec-S10c.2-ab-testing/ | S10c.1, S5b.2 | src/services/evaluation/ab_test.py, evals/ | A/B model comparison framework | Route % of traffic to different model configs (e.g., 50% gemini-flash vs 50% gemini-flash-lite for grading); track quality metrics per variant; statistical significance testing (chi-squared, t-test); experiment config via admin API; results dashboard showing per-variant: latency p50/p95/p99, citation accuracy, user satisfaction proxy (retry rate) | pending |
| S10c.3 | specs/spec-S10c.3-regression-detection/ | S10c.1, S10.1 | src/services/evaluation/regression.py, evals/ | Quality regression detection | Nightly automated eval run against S10.1 dataset; compare scores against baseline (last release); alert if any metric drops >5% from baseline; track metrics over time: faithfulness, relevance, citation accuracy, hallucination rate; auto-generate regression report; block deployment if critical metrics fail | pending |

## Phase 13: UI Enhancement & Design Polish — AFTER P9b done

> Transform PaperAlchemy from functional to visually stunning. Design inspiration: Semantic Scholar, Elicit, Vercel, Linear, Arc Browser.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S13.1 | specs/spec-S13.1-landing-page/ | S9.2 | frontend/src/app/page.tsx | Premium landing page | Animated hero with mesh gradient background, feature showcase grid (6 cards with icons + hover lift), live stats counter (papers indexed, queries answered), prominent CTA buttons, testimonial/use-case section, footer with links | done |
| S13.2 | specs/spec-S13.2-design-system/ | S9.1 | frontend/src/components/ui/, globals.css | Design system overhaul | Refined color palette (rich indigo/violet primary), glassmorphism cards (backdrop-blur + border opacity), 8px spacing grid, typography scale (Inter/Plus Jakarta Sans), gradient accents on primary surfaces, elevation system (shadow-sm → shadow-2xl), focus rings, smooth 200ms transitions on all interactive elements | done |
| S13.3 | specs/spec-S13.3-chat-ux-polish/ | S9.4, S9b.3 | frontend/src/components/chat/ | Premium chat experience | Markdown rendering (react-markdown + syntax highlighting via shiki), copy-to-clipboard on code blocks, rich citation cards (thumbnail, title, authors, journal, year), suggested follow-up chips after each response, animated typing indicator, message timestamps, smooth scroll animations, empty state with gradient illustration | done |
| S13.4 | specs/spec-S13.4-search-discovery/ | S9.3 | frontend/src/components/search/ | Search & discovery polish | Autocomplete/typeahead with recent searches, rich paper cards (abstract preview on hover, citation count badge, category color chips, bookmark icon), staggered fade-in animations, active filter pills with remove button, "no results" illustration + suggestions, skeleton shimmer loading states | done |
| S13.5 | specs/spec-S13.5-sidebar-navigation/ | S9.2, S9b.5 | frontend/src/components/layout/ | Premium navigation & layout | Sidebar with gradient logo mark, active item accent indicator (left border + subtle bg), smooth collapse animation with icon tooltips, keyboard shortcut hints, global command palette (Cmd+K) with fuzzy search across papers/collections/actions, breadcrumb trail with dropdown, notification bell with badge count | done |
| S13.6 | specs/spec-S13.6-micro-interactions/ | S9.1, S9b.3 | frontend/src/ | Animations & micro-interactions | Page transition animations (framer-motion), skeleton shimmer loaders on all data pages, button press feedback (scale-95), hover card previews for paper links, toast notifications with slide-in animation, progress indicators for long operations, scroll-triggered fade-in for dashboard cards, smooth number counting animations for stats | done |
| S13.7 | specs/spec-S13.7-responsive-mobile/ | S9.2 | frontend/src/ | Mobile-first responsive polish | Bottom navigation bar on mobile (replacing sidebar), swipe gestures for chat, pull-to-refresh on search, touch-optimized hit targets (min 44px), adaptive layouts (single → multi-column), sheet/drawer pattern for mobile filters, responsive typography (clamp-based fluid sizing) | done |

## Phase 14: Social & Community (AlphaXiv-inspired) — AFTER P9b done

> Bring academic discourse directly into PaperAlchemy. Inspired by [AlphaXiv](https://www.alphaxiv.org/)'s inline comments, voting, and community features.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S14.1 | specs/spec-S14.1-user-auth/ | S2.4, S9b.1, S9b.2, S9b.4 | src/services/auth/, src/models/user.py | User authentication & profiles | Email/OAuth sign-up (Google, GitHub, ORCID), JWT sessions, user profile (name, affiliation, interests, avatar), public profile page | pending |
| S14.2 | specs/spec-S14.2-paper-comments/ | S14.1, S9.6 | src/services/community/comments.py, src/routers/comments.py | Inline paper comments & discussions | Line-by-line comments on paper sections (like AlphaXiv), threaded replies, markdown support, @mentions, comment moderation (report/flag), real-time updates via WebSocket | pending |
| S14.3 | specs/spec-S14.3-voting-system/ | S14.1, S3.1 | src/services/community/voting.py | Paper voting & engagement metrics | Upvote/downvote papers, hot/trending ranking algorithm (time-decay + votes), visit counters, engagement analytics, prevent duplicate votes per user | pending |
| S14.4 | specs/spec-S14.4-author-profiles/ | S14.1, S3.1 | src/services/community/authors.py | Author profiles & follow system | Auto-generated author pages (papers, h-index, co-authors), follow authors for new paper alerts, ORCID integration for identity verification, author claim workflow | pending |
| S14.5 | specs/spec-S14.5-paper-discussions/ | S14.2 | src/services/community/discussions.py | Paper discussion threads | Top-level discussion threads per paper (separate from inline comments), Q&A format, accepted answers, topic tags, pinned discussions by paper authors | pending |
| S14.6 | specs/spec-S14.6-community-frontend/ | S14.1-S14.5, S9.6, S9b.5 | frontend/src/components/community/ | Community UI components | Comment sidebar on paper view, voting buttons with animation, author profile pages, discussion thread UI, user avatars, notification dropdown, activity feed | pending |

## Phase 15: Paper Annotations & Research Notebooks — AFTER P14+

> Personal knowledge management for researchers — highlight, annotate, collect, and synthesize.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S15.1 | specs/spec-S15.1-inline-annotations/ | S14.1, S9.6 | src/services/annotations/ | Inline paper highlights & annotations | Highlight text in paper view, add private notes to highlights, color-coded categories (key finding, methodology, limitation, question), persist per user | pending |
| S15.2 | specs/spec-S15.2-private-notes/ | S14.1, S3.1 | src/services/notes/ | Private paper notes | Personal notes per paper (markdown editor), separate from public comments, searchable across all notes, tag system | pending |
| S15.3 | specs/spec-S15.3-research-notebook/ | S15.1, S15.2 | src/services/notebook/ | Research notebook / scratchpad | Collect snippets from across papers into notebooks, drag-and-drop organization, auto-link back to source paper + section, markdown + LaTeX support | pending |
| S15.4 | specs/spec-S15.4-reading-progress/ | S14.1, S3.1 | src/services/reading/ | Reading progress tracking | Mark papers as to-read/in-progress/read, reading history timeline, time spent per paper, reading streaks & stats, resume where you left off | pending |
| S15.5 | specs/spec-S15.5-annotations-frontend/ | S15.1-S15.4, S9.6 | frontend/src/components/annotations/ | Annotations & notebook UI | Highlight overlay on paper text, annotation sidebar, notebook editor page, reading progress indicators, cross-paper snippet collector | pending |

## Phase 16: Discovery & Intelligence — AFTER P4+

> Go beyond search — proactively surface relevant papers, visualize connections, and track field evolution.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S16.1 | specs/spec-S16.1-paper-recommendations/ | S4b.5, S14.1 | src/services/discovery/recommender.py | Paper recommendations engine | "Similar to papers you've read" via embedding similarity, collaborative filtering (users who read X also read Y), recency boost, diversification to avoid echo chamber | pending |
| S16.2 | specs/spec-S16.2-weekly-digest/ | S16.1, S3.4 | src/services/discovery/digest.py | Weekly research digest | Auto-generated summary of new papers in user's topics, personalized based on reading history + followed authors, email + in-app delivery, Airflow scheduled job | pending |
| S16.3 | specs/spec-S16.3-citation-graph/ | S3.1, S4.3 | src/services/discovery/citation_graph.py | Citation graph visualization | Interactive graph (D3.js/react-force-graph) showing how papers cite each other, cluster by topic, highlight influential nodes, click-to-navigate, zoom/pan, export as image | pending |
| S16.4 | specs/spec-S16.4-research-timeline/ | S3.1 | src/services/discovery/timeline.py | Research timeline & field evolution | Visual timeline of a research topic's evolution, key milestones, paradigm shifts (e.g., RNNs → Transformers → LLMs), auto-generated from paper dates + citations | pending |
| S16.5 | specs/spec-S16.5-trending-feed/ | S14.3, S3.1 | src/services/discovery/trending.py | Trending & hot papers feed | AlphaXiv-style hot/trending sorting, category filters (cs.AI, cs.CL, cs.CV, etc.), organization filter (Meta, Google, OpenAI), time-decay ranking, personalized feed option | pending |
| S16.6 | specs/spec-S16.6-discovery-frontend/ | S16.1-S16.5 | frontend/src/app/discover/ | Discovery UI pages | Recommendations carousel, weekly digest page, interactive citation graph viewer, research timeline visualization, trending feed with sort/filter tabs, "Explore" landing page | pending |

## Phase 17: Collaboration & Sharing — AFTER P14+

> Enable team research workflows — share collections, discuss papers, and collaborate on literature reviews.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S17.1 | specs/spec-S17.1-shared-collections/ | S14.1, S9b.6 | src/services/collaboration/shared_collections.py | Shared collections with team | Invite collaborators by email, role-based access (viewer/editor/admin), real-time sync, activity log (who added/removed what), public/private toggle | pending |
| S17.2 | specs/spec-S17.2-team-comments/ | S14.2, S17.1 | src/services/collaboration/team_comments.py | Team discussion threads | Collection-level discussion threads, @mention team members, resolve/unresolve comments, link comments to specific papers in collection | pending |
| S17.3 | specs/spec-S17.3-shareable-summaries/ | S8.2, S14.1 | src/services/collaboration/share.py | Shareable research summaries | Generate public share link for AI summaries, embed-friendly format, branded PaperAlchemy card with paper metadata, social media preview (Open Graph tags) | pending |
| S17.4 | specs/spec-S17.4-collaboration-frontend/ | S17.1-S17.3 | frontend/src/components/collaboration/ | Collaboration UI | Team member avatars on collections, invite modal, shared collection activity feed, shareable summary preview/copy, public profile pages with published collections | pending |

## Phase 18: Workflow Integration — AFTER P15+

> Connect PaperAlchemy with existing research tools and workflows.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S18.1 | specs/spec-S18.1-bibtex-import/ | S3.1, S14.1 | src/services/integrations/bibtex.py | BibTeX & Zotero/Mendeley import | Import .bib files, parse BibTeX entries, match to existing papers in KB, bulk import from Zotero/Mendeley export, auto-resolve arXiv IDs | pending |
| S18.2 | specs/spec-S18.2-knowledge-export/ | S15.2, S9.9 | src/services/integrations/export.py | Notion & Obsidian export | Export paper notes + annotations as Markdown (Obsidian vault format), Notion API integration for direct push, preserve backlinks and tags, bulk export collections | pending |
| S18.3 | specs/spec-S18.3-public-api/ | S2.1, S14.1 | src/routers/api_v2.py | Public REST API with API keys | Programmatic access to search, papers, collections, AI queries; API key management, rate limiting, usage dashboard, OpenAPI docs, SDKs (Python/JS) | pending |
| S18.4 | specs/spec-S18.4-browser-extension/ | S18.3 | extension/ | Chrome/Firefox browser extension | Detect arXiv pages, one-click save to PaperAlchemy, inline AI summary popup, add to collection from browser, keyboard shortcut (Alt+P), badge with unread count | pending |
| S18.5 | specs/spec-S18.5-integration-frontend/ | S18.1-S18.4 | frontend/src/app/settings/integrations/ | Integrations settings UI | Import/export management page, API key generation UI, connected services list, browser extension install prompt, usage stats | pending |

## Phase 19: Advanced AI Features — AFTER P6+

> Push AI capabilities beyond single-paper Q&A into multi-paper synthesis and generation.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S19.1 | specs/spec-S19.1-multi-paper-qa/ | S6.7, S8.5 | src/services/ai/multi_paper_qa.py | Multi-paper Q&A | "Compare methodology across these 5 papers", cross-paper reasoning, unified citation across multiple sources, context window management for large paper sets | pending |
| S19.2 | specs/spec-S19.2-literature-review/ | S19.1 | src/services/ai/literature_review.py | Literature review generator | Auto-generate survey section from a collection of papers, structured output (introduction, methodology comparison, findings synthesis, gaps, future directions), editable draft | pending |
| S19.3 | specs/spec-S19.3-slide-generation/ | S8.2, S5.1 | src/services/ai/slide_generator.py | Paper-to-slide deck generation | Generate presentation slides from paper summary, key figures referenced, speaker notes, export as PPTX/PDF/Google Slides, customizable templates | pending |
| S19.4 | specs/spec-S19.4-research-gaps/ | S19.1, S16.3 | src/services/ai/gap_finder.py | Research gap identifier | Analyze a set of papers to find unexplored areas, suggest future research directions, identify contradictions between papers, confidence scoring | pending |
| S19.5 | specs/spec-S19.5-ai-paper-assistant/ | S6.7, S9.6 | src/services/ai/paper_assistant.py | In-context AI paper assistant | AlphaXiv-style: chat with AI directly on top of paper, highlight text → ask questions about it, @reference other papers for comparison, paragraph-level citations with page numbers | pending |

## Phase 20: Interactive Labs — AFTER P16+

> Experimental visualizations and interactive tools for exploring the research landscape.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S20.1 | specs/spec-S20.1-geo-trends/ | S16.5, S3.1 | src/services/labs/geo_trends.py, frontend/src/app/labs/geo/ | Geo-trends visualization | World map showing research output by country/institution, filter by topic/year, animate over time, heatmap overlay, drill-down to institution papers | pending |
| S20.2 | specs/spec-S20.2-research-playground/ | S6.7 | frontend/src/app/labs/playground/ | Research playground | Interactive sandbox: compose custom RAG queries, tune retrieval parameters (top-k, rerank threshold, HyDE on/off), see side-by-side results, save configurations | pending |
| S20.3 | specs/spec-S20.3-paper-explorer/ | S16.3 | frontend/src/app/labs/explorer/ | Paper relationship explorer | 3D force-directed graph of paper relationships, cluster by topic/methodology, time slider to see field evolution, click paper node for preview card, fullscreen mode | pending |

## Phase 21: AI Paper Blog & Overview — AFTER P8+

> Auto-generate rich, structured blog-style overviews for every paper — like [AlphaXiv's overview pages](https://www.alphaxiv.org/overview/2603.07685). Transform dense PDFs into readable, shareable deep-dives.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S21.1 | specs/spec-S21.1-paper-overview-generator/ | S8.2, S5.1 | src/services/blog/overview_generator.py | AI paper overview generator | Auto-generate structured blog-style overview from paper: Introduction, Core Challenges, Architecture, Methodology, Results, Significance; 2000-5000 word deep-dive; markdown output with headers, bold terms, code formatting for technical identifiers; cached per paper | pending |
| S21.2 | specs/spec-S21.2-tabbed-paper-view/ | S21.1, S9.6 | frontend/src/app/papers/[id]/ | Tabbed paper view (Paper / Blog / Resources) | AlphaXiv-style tabs: "Paper" (original PDF/sections), "Blog" (AI overview), "Resources" (GitHub repos, datasets, slides); tab state persisted in URL; smooth tab transitions | pending |
| S21.3 | specs/spec-S21.3-paper-assistant-tab/ | S19.5, S21.2 | frontend/src/components/paper/assistant-tab.tsx | In-paper AI assistant tab | "Assistant" tab on paper view: conversational AI that answers questions strictly from the paper content, paragraph-level citations with page numbers, highlight-to-ask, bold formatting, image content interpretation | pending |
| S21.4 | specs/spec-S21.4-similar-papers/ | S16.1, S21.2 | src/services/blog/similar.py, frontend/src/components/paper/similar-tab.tsx | Similar papers tab | "Similar" tab: embedding-based similarity search, ranked list with relevance score, paper cards with abstract preview, one-click add to collection, filter by year/category | pending |
| S21.5 | specs/spec-S21.5-paper-resources/ | S3.1, S21.2 | src/services/blog/resources.py | Paper resources aggregator | Auto-detect and link: GitHub repos (with star counts), datasets, project pages, video presentations, blog posts, slides; manual resource submission by users; resource type icons | pending |
| S21.6 | specs/spec-S21.6-overview-social/ | S14.3, S21.1 | src/services/blog/social.py, frontend/ | Overview engagement & sharing | View counter (total + last 24h), upvote/downvote on overview, social sharing (Twitter/LinkedIn card with auto-generated preview image + title + authors), SEO-optimized meta tags (Open Graph, Schema.org), permalink for each overview | pending |

## Phase 22: Paper-to-Code Implementation — AFTER P21+

> AI agent that analyzes whether a paper's method is implementable, then generates a working code implementation. Uses Claude or other frontier models for code generation.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S22.1 | specs/spec-S22.1-implementability-analysis/ | S8.2, S5.1 | src/services/code_gen/feasibility.py | Paper implementability analysis | AI analyzes paper to determine: is the method implementable from the paper alone? Scores feasibility (high/medium/low/not-possible), identifies required: datasets, hardware, dependencies, missing details; outputs structured assessment with reasoning | pending |
| S22.2 | specs/spec-S22.2-code-generation-agent/ | S22.1, S6.7, S9b.2 | src/services/code_gen/agent.py | Paper-to-code generation agent | LangGraph agent that reads paper sections (method, algorithm, architecture) and generates implementation; uses Claude Opus/Sonnet via Anthropic API or configurable frontier model; iterative: generate → self-review → fix → test; outputs structured project (main code, requirements, README, tests) | pending |
| S22.3 | specs/spec-S22.3-code-sandbox/ | S22.2, S9b.7 | src/services/code_gen/sandbox.py | Code execution sandbox | Secure sandboxed execution environment (Docker container or E2B) for running generated code; captures stdout/stderr, test results, error traces; timeout limits, resource caps; auto-retry with error feedback to agent | pending |
| S22.4 | specs/spec-S22.4-code-export/ | S22.2, S14.1 | src/services/code_gen/export.py | Code export & GitHub integration | Export generated code as: downloadable ZIP, GitHub Gist, or auto-create GitHub repo; includes README with paper citation, requirements.txt, license; version history of generation attempts | pending |
| S22.5 | specs/spec-S22.5-code-gen-frontend/ | S22.1-S22.4, S21.2 | frontend/src/components/paper/code-tab.tsx | Paper-to-code UI | "Code" tab on paper view: feasibility badge (implementable/partial/not-possible), "Generate Implementation" button with progress steps, live code preview with syntax highlighting, run-in-sandbox button, download/export options, generation history | pending |

## Phase 23: AI Audio & Podcast Generation — AFTER P21+

> Generate engaging audio discussions about papers — seminar-style dialogues with multiple AI voices, like NotebookLM's audio overviews. Configurable speakers/avatars.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S23.1 | specs/spec-S23.1-podcast-script-generator/ | S21.1, S5.1 | src/services/audio/script_generator.py | Podcast script generator | AI generates seminar-style dialogue script from paper overview; configurable format: professor + student Q&A, two researchers debating, solo narrator; covers: motivation, core concepts explained simply, technical details, significance, open questions; markdown script with speaker labels, natural conversation flow, analogies for complex concepts | pending |
| S23.2 | specs/spec-S23.2-text-to-speech/ | S23.1, S9b.2, S9b.7 | src/services/audio/tts.py | Multi-voice text-to-speech engine | Convert dialogue script to audio with distinct voices per speaker; TTS provider abstraction (Google Cloud TTS, ElevenLabs, OpenAI TTS, or edge-tts for free tier); voice profile per avatar (name, voice ID, speaking style); SSML support for emphasis and pauses; output MP3/WAV; audio files stored in MinIO/S3 | pending |
| S23.3 | specs/spec-S23.3-audio-avatars/ | S23.1, S14.1 | src/services/audio/avatars.py, src/models/avatar.py | Configurable speaker avatars | Customizable speaker profiles: name, role (professor, student, researcher, narrator), voice selection, avatar image; preset avatars (e.g., "Professor John", "Student Noah"); user-created custom avatars; avatar used in transcript UI and audio player | pending |
| S23.4 | specs/spec-S23.4-transcript-viewer/ | S23.1, S23.2, S23.3 | src/services/audio/transcript.py | Synchronized transcript | Full transcript with speaker labels + avatar icons; auto-scroll synchronized with audio playback; click-to-seek on any transcript line; highlight current speaking line; searchable transcript; export as text/SRT/VTT | pending |
| S23.5 | specs/spec-S23.5-audio-frontend/ | S23.1-S23.4, S21.2 | frontend/src/components/paper/audio-tab.tsx | Audio player & podcast UI | "Listen" tab on paper view: embedded audio player (play/pause, seek, speed control 0.5x-2x), speaker avatars displayed during playback, synchronized transcript panel, "Generate Podcast" button with style selector (seminar/debate/narrative), download MP3, share audio link, queue for offline listening | pending |

---

## Master Spec Index

| Spec | Feature | Phase | Notebook | Status |
|------|---------|-------|----------|--------|
| S1.1 | Dependency declaration | P1 | notebooks/specs/S1.1_dependency.ipynb | done |
| S1.2 | Environment configuration | P1 | notebooks/specs/S1.2_config.ipynb | done |
| S1.3 | Docker infrastructure | P1 | notebooks/specs/S1.3_docker.ipynb | done |
| S1.4 | CI/CD setup | P1 | notebooks/specs/S1.4_cicd.ipynb | done |
| S2.1 | FastAPI app factory | P2 | notebooks/specs/S2.1_app_factory.ipynb | done |
| S2.2 | Database layer | P2 | notebooks/specs/S2.2_database.ipynb | done |
| S2.3 | Error handling & middleware | P2 | notebooks/specs/S2.3_error_handling.ipynb | done |
| S2.4 | Dependency injection | P2 | notebooks/specs/S2.4_di.ipynb | done |
| S3.1 | Paper model & repository | P3 | notebooks/specs/S3.1_paper_model.ipynb | done |
| S3.2 | ArXiv client | P3 | notebooks/specs/S3.2_arxiv_client.ipynb | done |
| S3.3 | PDF parser (Docling) | P3 | notebooks/specs/S3.3_pdf_parser.ipynb | done |
| S3.4 | Ingestion pipeline (Airflow) | P3 | notebooks/specs/S3.4_ingestion.ipynb | done |
| S4.1 | OpenSearch client | P4 | notebooks/specs/S4.1_opensearch.ipynb | done |
| S4.2 | Text chunker | P4 | notebooks/specs/S4.2_chunker.ipynb | done |
| S4.3 | Embedding service (Jina) | P4 | notebooks/specs/S4.3_embeddings.ipynb | done |
| S4.4 | Hybrid search (BM25+KNN+RRF) | P4 | notebooks/specs/S4.4_hybrid_search.ipynb | done |
| S4b.1 | Cross-encoder re-ranking | P4b | notebooks/specs/S4b.1_reranker.ipynb | done |
| S4b.2 | HyDE retrieval | P4b | notebooks/specs/S4b.2_hyde.ipynb | done |
| S4b.3 | Multi-query retrieval | P4b | notebooks/specs/S4b.3_multi_query.ipynb | done |
| S4b.4 | Parent-child chunk retrieval | P4b | notebooks/specs/S4b.4_parent_child.ipynb | done |
| S4b.5 | Unified retrieval pipeline | P4b | notebooks/specs/S4b.5_retrieval_pipeline.ipynb | done |
| S5.1 | LLM client (Ollama + Gemini) | P5 | notebooks/specs/S5.1_llm_client.ipynb | done |
| S5.2 | RAG chain | P5 | notebooks/specs/S5.2_rag_chain.ipynb | done |
| S5.3 | Streaming responses (SSE) | P5 | notebooks/specs/S5.3_streaming.ipynb | done |
| S5.4 | Response caching (Redis) | P5 | notebooks/specs/S5.4_caching.ipynb | done |
| S5.5 | Citation enforcement | P5 | notebooks/specs/S5.5_citations.ipynb | done |
| S6.1 | Agent state & context | P6 | notebooks/specs/S6.1_agent_state.ipynb | done |
| S6.2 | Guardrail node | P6 | notebooks/specs/S6.2_guardrail.ipynb | done |
| S6.3 | Retrieval node | P6 | notebooks/specs/S6.3_retrieval.ipynb | done |
| S6.4 | Document grading node | P6 | notebooks/specs/S6.4_grading.ipynb | done |
| S6.5 | Query rewrite node | P6 | notebooks/specs/S6.5_rewrite.ipynb | done |
| S6.6 | Answer generation node | P6 | notebooks/specs/S6.6_generation.ipynb | done |
| S6.7 | Agent orchestrator (LangGraph) | P6 | notebooks/specs/S6.7_orchestrator.ipynb | done |
| S6.8 | Specialized agents | P6 | notebooks/specs/S6.8_specialized.ipynb | done |
| S7.1 | Conversation memory | P7 | notebooks/specs/S7.1_chat_memory.ipynb | done |
| S7.2 | Follow-up handler | P7 | notebooks/specs/S7.2_follow_up.ipynb | done |
| S7.3 | Chat API | P7 | notebooks/specs/S7.3_chat_api.ipynb | done |
| S8.1 | PDF upload endpoint | P8 | notebooks/specs/S8.1_upload.ipynb | done |
| S8.2 | Paper summary generation | P8 | notebooks/specs/S8.2_summary.ipynb | done |
| S8.3 | Key highlights extraction | P8 | notebooks/specs/S8.3_highlights.ipynb | done |
| S8.4 | Methodology analysis | P8 | notebooks/specs/S8.4_methodology.ipynb | done |
| S8.5 | Paper comparison | P8 | notebooks/specs/S8.5_comparison.ipynb | done |
| S9b.1 | Alembic migration setup | P9b | notebooks/specs/S9b.1_alembic.ipynb | done |
| S9b.2 | Platform dependency declaration | P9b | notebooks/specs/S9b.2_platform_deps.ipynb | done |
| S9b.3 | Frontend infrastructure dependencies | P9b | — | done |
| S9b.4 | Frontend auth infrastructure | P9b | — | done |
| S9b.5 | Missing UI primitives (Dialog, Tabs, etc.) | P9b | — | done |
| S9b.6 | Collections backend API | P9b | notebooks/specs/S9b.6_collections_backend.ipynb | done |
| S9b.7 | Docker platform services (S3, sandbox) | P9b | — | done |
| S5b.1 | Anthropic Claude LLM provider | P5b | — | done |
| S5b.2 | Multi-provider LLM routing | P5b | notebooks/specs/S5b.2_multi_provider.ipynb | done |
| S9.1 | Next.js project setup | P9 | — (Vitest) | done |
| S9.2 | Layout & navigation | P9 | — | done |
| S9.3 | Search interface | P9 | — | done |
| S9.4 | Chat interface (streaming) | P9 | — | done |
| S9.5 | Paper upload UI | P9 | — | done |
| S9.6 | Paper detail view | P9 | — | done |
| S9.7 | Reading lists & collections | P9 | — | done |
| S9.8 | Trends dashboard | P9 | — | done |
| S9.9 | Export & citations | P9 | — | done |
| S10.1 | Evaluation dataset | P10 | notebooks/specs/S10.1_eval_dataset.ipynb | pending |
| S10.2 | LLM-as-judge evaluator | P10 | notebooks/specs/S10.2_llm_judge.ipynb | pending |
| S10.3 | Human evaluation UI | P10 | — (frontend) | pending |
| S10.4 | Quality metrics (RAGAS) | P10 | notebooks/specs/S10.4_metrics.ipynb | pending |
| S10.5 | Benchmark dashboard | P10 | — (frontend) | pending |
| S10b.1 | Paper overview quality evaluation | P10b | notebooks/specs/S10b.1_overview_eval.ipynb | pending |
| S10b.2 | Code generation quality evaluation | P10b | notebooks/specs/S10b.2_code_gen_eval.ipynb | pending |
| S10b.3 | Podcast script quality evaluation | P10b | notebooks/specs/S10b.3_audio_eval.ipynb | pending |
| S10b.4 | Recommendation quality evaluation | P10b | notebooks/specs/S10b.4_recommendation_eval.ipynb | pending |
| S11.1 | Langfuse tracing | P11 | notebooks/specs/S11.1_tracing.ipynb | pending |
| S11.2 | Monitoring & alerts | P11 | notebooks/specs/S11.2_monitoring.ipynb | pending |
| S11.3 | Ops documentation | P11 | — (docs) | pending |
| S11.4 | GCP Cloud Run deployment | P11 | — (deploy) | pending |
| S11.5 | Production CI/CD | P11 | — (workflows) | pending |
| S12.1 | Telegram bot setup | P12 | notebooks/specs/S12.1_telegram_setup.ipynb | pending |
| S12.2 | Telegram RAG integration | P12 | notebooks/specs/S12.2_telegram_rag.ipynb | pending |
| S12.3 | Telegram search | P12 | notebooks/specs/S12.3_telegram_search.ipynb | pending |
| S12.4 | Telegram notifications | P12 | notebooks/specs/S12.4_telegram_notifications.ipynb | pending |
| S13.1 | Premium landing page | P13 | — | done |
| S13.2 | Design system overhaul | P13 | — | done |
| S13.3 | Chat UX polish | P13 | — | done |
| S13.4 | Search & discovery polish | P13 | — | done |
| S13.5 | Premium navigation & command palette | P13 | — | done |
| S13.6 | Animations & micro-interactions | P13 | — | done |
| S13.7 | Mobile-first responsive polish | P13 | — | done |
| S14.1 | User authentication & profiles | P14 | notebooks/specs/S14.1_user_auth.ipynb | pending |
| S14.2 | Inline paper comments & discussions | P14 | notebooks/specs/S14.2_paper_comments.ipynb | pending |
| S14.3 | Paper voting & engagement metrics | P14 | notebooks/specs/S14.3_voting.ipynb | pending |
| S14.4 | Author profiles & follow system | P14 | notebooks/specs/S14.4_author_profiles.ipynb | pending |
| S14.5 | Paper discussion threads | P14 | notebooks/specs/S14.5_discussions.ipynb | pending |
| S14.6 | Community UI components | P14 | — | pending |
| S15.1 | Inline paper highlights & annotations | P15 | notebooks/specs/S15.1_annotations.ipynb | pending |
| S15.2 | Private paper notes | P15 | notebooks/specs/S15.2_private_notes.ipynb | pending |
| S15.3 | Research notebook / scratchpad | P15 | notebooks/specs/S15.3_research_notebook.ipynb | pending |
| S15.4 | Reading progress tracking | P15 | notebooks/specs/S15.4_reading_progress.ipynb | pending |
| S15.5 | Annotations & notebook UI | P15 | — | pending |
| S16.1 | Paper recommendations engine | P16 | notebooks/specs/S16.1_recommendations.ipynb | pending |
| S16.2 | Weekly research digest | P16 | notebooks/specs/S16.2_weekly_digest.ipynb | pending |
| S16.3 | Citation graph visualization | P16 | notebooks/specs/S16.3_citation_graph.ipynb | pending |
| S16.4 | Research timeline & field evolution | P16 | notebooks/specs/S16.4_research_timeline.ipynb | pending |
| S16.5 | Trending & hot papers feed | P16 | notebooks/specs/S16.5_trending_feed.ipynb | pending |
| S16.6 | Discovery UI pages | P16 | — | pending |
| S17.1 | Shared collections with team | P17 | notebooks/specs/S17.1_shared_collections.ipynb | pending |
| S17.2 | Team discussion threads | P17 | notebooks/specs/S17.2_team_comments.ipynb | pending |
| S17.3 | Shareable research summaries | P17 | notebooks/specs/S17.3_shareable_summaries.ipynb | pending |
| S17.4 | Collaboration UI | P17 | — | pending |
| S18.1 | BibTeX & Zotero/Mendeley import | P18 | notebooks/specs/S18.1_bibtex_import.ipynb | pending |
| S18.2 | Notion & Obsidian export | P18 | notebooks/specs/S18.2_knowledge_export.ipynb | pending |
| S18.3 | Public REST API with API keys | P18 | notebooks/specs/S18.3_public_api.ipynb | pending |
| S18.4 | Chrome/Firefox browser extension | P18 | — | pending |
| S18.5 | Integrations settings UI | P18 | — | pending |
| S19.1 | Multi-paper Q&A | P19 | notebooks/specs/S19.1_multi_paper_qa.ipynb | pending |
| S19.2 | Literature review generator | P19 | notebooks/specs/S19.2_literature_review.ipynb | pending |
| S19.3 | Paper-to-slide deck generation | P19 | notebooks/specs/S19.3_slide_generation.ipynb | pending |
| S19.4 | Research gap identifier | P19 | notebooks/specs/S19.4_research_gaps.ipynb | pending |
| S19.5 | In-context AI paper assistant | P19 | notebooks/specs/S19.5_ai_paper_assistant.ipynb | pending |
| S20.1 | Geo-trends visualization | P20 | — | pending |
| S20.2 | Research playground | P20 | — | pending |
| S20.3 | Paper relationship explorer | P20 | — | pending |
| S21.1 | AI paper overview generator | P21 | notebooks/specs/S21.1_overview_generator.ipynb | pending |
| S21.2 | Tabbed paper view (Paper/Blog/Resources) | P21 | — | pending |
| S21.3 | In-paper AI assistant tab | P21 | — | pending |
| S21.4 | Similar papers tab | P21 | — | pending |
| S21.5 | Paper resources aggregator | P21 | notebooks/specs/S21.5_paper_resources.ipynb | pending |
| S21.6 | Overview engagement & sharing | P21 | — | pending |
| S22.1 | Paper implementability analysis | P22 | notebooks/specs/S22.1_feasibility.ipynb | pending |
| S22.2 | Paper-to-code generation agent | P22 | notebooks/specs/S22.2_code_gen_agent.ipynb | pending |
| S22.3 | Code execution sandbox | P22 | notebooks/specs/S22.3_code_sandbox.ipynb | pending |
| S22.4 | Code export & GitHub integration | P22 | notebooks/specs/S22.4_code_export.ipynb | pending |
| S22.5 | Paper-to-code UI | P22 | — | pending |
| S23.1 | Podcast script generator | P23 | notebooks/specs/S23.1_podcast_script.ipynb | pending |
| S23.2 | Multi-voice text-to-speech | P23 | notebooks/specs/S23.2_tts.ipynb | pending |
| S23.3 | Configurable speaker avatars | P23 | notebooks/specs/S23.3_avatars.ipynb | pending |
| S23.4 | Synchronized transcript viewer | P23 | notebooks/specs/S23.4_transcript.ipynb | pending |
| S23.5 | Audio player & podcast UI | P23 | — | pending |
| S11b.1 | Platform monitoring (new services) | P11b | notebooks/specs/S11b.1_platform_monitoring.ipynb | pending |
| S11b.2 | Platform ops runbooks | P11b | — (docs) | pending |
| S11b.3 | Extended Cloud Run deployment | P11b | — (deploy) | pending |
| S12b.1 | Telegram paper overview sharing | P12b | notebooks/specs/S12b.1_telegram_overview.ipynb | pending |
| S12b.2 | Telegram trending papers feed | P12b | notebooks/specs/S12b.2_telegram_trending.ipynb | pending |
| S12b.3 | Telegram code snippet sharing | P12b | notebooks/specs/S12b.3_telegram_code.ipynb | pending |
| S12b.4 | Telegram community notifications | P12b | notebooks/specs/S12b.4_telegram_community.ipynb | pending |
| S3b.1 | Auto-trigger overview generation | P3b | notebooks/specs/S3b.1_auto_overview.ipynb | pending |
| S3b.2 | Trending score computation | P3b | notebooks/specs/S3b.2_trending_scores.ipynb | pending |
| S3b.3 | Resource auto-detection pipeline | P3b | notebooks/specs/S3b.3_resource_detection.ipynb | pending |
| S6b.1 | Parallel document grading | P6b | notebooks/specs/S6b.1_parallel_grading.ipynb | pending |
| S6b.2 | Per-node timeouts & circuit breakers | P6b | notebooks/specs/S6b.2_node_timeouts.ipynb | pending |
| S6b.3 | Smart grading bypass (reranker scores) | P6b | notebooks/specs/S6b.3_smart_grading.ipynb | pending |
| S6b.4 | Streaming grading with early generation | P6b | notebooks/specs/S6b.4_streaming_grading.ipynb | pending |
| S5c.1 | Semantic caching (embedding similarity) | P5c | notebooks/specs/S5c.1_semantic_cache.ipynb | pending |
| S5c.2 | Query embedding cache (Redis) | P5c | notebooks/specs/S5c.2_embedding_cache.ipynb | pending |
| S5c.3 | Multi-tier LLM routing | P5c | notebooks/specs/S5c.3_multi_tier_llm.ipynb | pending |
| S5c.4 | LLM concurrency control (semaphores) | P5c | notebooks/specs/S5c.4_llm_concurrency.ipynb | pending |
| S5c.5 | Batch LLM calls for grading | P5c | notebooks/specs/S5c.5_batch_llm.ipynb | pending |
| S2b.1 | Async OpenSearch client | P2b | notebooks/specs/S2b.1_async_opensearch.ipynb | pending |
| S2b.2 | Database pool tuning | P2b | notebooks/specs/S2b.2_db_pool.ipynb | pending |
| S2b.3 | Service pre-warming & startup | P2b | notebooks/specs/S2b.3_prewarming.ipynb | pending |
| S2b.4 | Capacity-aware health endpoint | P2b | notebooks/specs/S2b.4_capacity_health.ipynb | pending |
| S10c.1 | Response quality monitoring | P10c | notebooks/specs/S10c.1_quality_monitor.ipynb | pending |
| S10c.2 | A/B model comparison framework | P10c | notebooks/specs/S10c.2_ab_testing.ipynb | pending |
| S10c.3 | Quality regression detection | P10c | notebooks/specs/S10c.3_regression.ipynb | pending |
