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
- Gemini 3 Flash integration designed in (not patched)

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
- **Every spec** must be testable via UI (Gradio initially, Next.js later)

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
| **LLM (Local)** | Ollama (llama3.2, mistral) | Free local inference for dev |
| **LLM (Cloud)** | Google Gemini 3 Flash | Best research perf/cost, generous free tier |
| **Agent Framework** | LangGraph 0.2+ | State-machine agents, decision graphs |
| **LLM Framework** | LangChain 0.3+ | Chains, tools, prompt templates |
| **PDF Parsing** | Docling 2.43+ | Section-aware scientific PDF parsing |
| **Caching** | Redis 7 | Sub-ms response caching, LRU eviction |
| **Observability** | Langfuse v3 | Per-node LLM tracing, cost tracking |
| **Workflow** | Apache Airflow 2.10 | Scheduled data ingestion DAGs |
| **Frontend** | Next.js 15 + TypeScript + Tailwind + shadcn/ui | SSR, streaming, modern DX |
| **Fallback UI** | Gradio 4.0+ | Quick dev/testing UI (built per-spec) |
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
| Gemini 3 Flash API | $0 | Free tier: 15 RPM |
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
| P10 | Evaluation Framework | 5 | Datasets, LLM judges, human eval, RAGAS, dashboard | ✅ P6+ |
| P11 | Observability, Ops & Deploy | 5 | Tracing, monitoring, ops guide, GCP deploy, CI/CD | — |
| P12 | Telegram Bot | 4 | Mobile-first research assistant via Telegram | ✅ P7+ |

**Total: 65 specs**

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
| S5.1 | specs/spec-S5.1-llm-client/ | S1.2 | src/services/llm/ | Unified LLM client (Ollama + Gemini 3 Flash) | Provider abstraction: LLMProvider interface, OllamaProvider, GeminiProvider, model switching | done |
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
| S8.1 | specs/spec-S8.1-pdf-upload/ | S3.3, S2.4 | src/routers/upload.py | PDF upload endpoint | Multipart form, 50MB limit, PDF-only validation, stores + indexes | pending |
| S8.2 | specs/spec-S8.2-paper-summary/ | S8.1, S5.1 | src/services/analysis/summarizer.py | AI-generated paper summary | Structured: objective, method, key findings, contribution, limitations | pending |
| S8.3 | specs/spec-S8.3-key-highlights/ | S8.1, S5.1 | src/services/analysis/highlights.py | Extract key highlights & insights | Novel contributions, important findings, practical implications | pending |
| S8.4 | specs/spec-S8.4-methodology-analysis/ | S8.1, S5.1 | src/services/analysis/methodology.py | Methodology & findings deep-dive | Datasets used, baselines compared, results tables, statistical significance | pending |
| S8.5 | specs/spec-S8.5-paper-comparison/ | S8.2 | src/services/analysis/comparator.py | Side-by-side paper comparison | Compare 2+ papers: methods, results, contributions, limitations | pending |

## Phase 9: Frontend (Next.js) — PARALLEL from P3+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S9.1 | specs/spec-S9.1-nextjs-setup/ | — | frontend/ | Next.js 15 + TS + Tailwind + shadcn/ui | App Router, pnpm, ESLint, Vitest, dark mode | done |
| S9.2 | specs/spec-S9.2-layout-navigation/ | S9.1 | frontend/src/app/ | App shell: sidebar, header, theme toggle | Responsive, collapsible sidebar, breadcrumbs | done |
| S9.3 | specs/spec-S9.3-search-interface/ | S9.2 | frontend/src/app/search/ | Search page with filters & results | Category filter, sort, pagination, paper cards with arXiv links | done |
| S9.4 | specs/spec-S9.4-chat-interface/ | S9.2 | frontend/src/app/chat/ | RAG chatbot with streaming | SSE consumption, follow-up Q&A, message history, citation rendering with clickable arXiv links | done |
| S9.5 | specs/spec-S9.5-paper-upload-ui/ | S9.2 | frontend/src/app/upload/ | Drag-and-drop PDF upload | Progress bar, summary + highlights + methodology display | done |
| S9.6 | specs/spec-S9.6-paper-detail-view/ | S9.2 | frontend/src/app/papers/[id]/ | Full paper view with highlights | Sections, annotations, metadata, arXiv link, related papers | done |
| S9.7 | specs/spec-S9.7-reading-lists/ | S9.2 | frontend/src/app/collections/ | Save & organize papers into collections | CRUD collections, drag-and-drop, share link | done |
| S9.8 | specs/spec-S9.8-trends-dashboard/ | S9.2 | frontend/src/app/dashboard/ | Research trends & analytics | Charts (recharts), trending topics, hot papers, category breakdown | done |
| S9.9 | specs/spec-S9.9-export-citations/ | S9.6 | frontend/src/components/export/ | BibTeX, Markdown, slide export | Copy-to-clipboard, download file | done |

## Phase 10: Evaluation Framework — PARALLEL from P6+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S10.1 | specs/spec-S10.1-eval-dataset/ | S5.2 | evals/datasets/ | Curated Q&A evaluation dataset | 100+ Q&A triples from indexed papers + SciQ + QASPER subsets | pending |
| S10.2 | specs/spec-S10.2-llm-judge/ | S10.1, S5.1 | evals/judges/llm_judge.py | LLM-as-judge evaluator | Faithfulness, relevance, coherence, citation accuracy scoring via Gemini 3 Flash | pending |
| S10.3 | specs/spec-S10.3-human-eval-ui/ | S10.1, S9.1 | frontend/src/app/eval/ | Human evaluation interface | Side-by-side comparison, Likert scales (1-5), inter-annotator agreement | pending |
| S10.4 | specs/spec-S10.4-quality-metrics/ | S10.2 | evals/metrics/ | RAGAS metrics + custom metrics | Answer correctness, context precision/recall, hallucination rate, citation accuracy | pending |
| S10.5 | specs/spec-S10.5-benchmark-dashboard/ | S10.4, S9.1 | frontend/src/app/benchmarks/ | Evaluation results dashboard | Model comparison, metric trends over time, regression alerts | pending |

## Phase 11: Observability, Ops & Deployment

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S11.1 | specs/spec-S11.1-langfuse-tracing/ | S5.2 | src/services/langfuse/ | Per-node LLM tracing | Cost tracking, latency analysis, per-request traces with spans | pending |
| S11.2 | specs/spec-S11.2-monitoring/ | S2.1 | src/services/monitoring/ | Health checks, metrics, alerts | Prometheus-compatible /metrics endpoint | pending |
| S11.3 | specs/spec-S11.3-ops-documentation/ | S1.3 | docs/ops-guide.md | Operations guide | Start/stop, Docker commands, Makefile, env setup, troubleshooting, architecture | pending |
| S11.4 | specs/spec-S11.4-gcp-deployment/ | ALL | deploy/ | GCP Cloud Run deployment | Terraform/gcloud, auto-scaling, secrets | pending |
| S11.5 | specs/spec-S11.5-production-ci-cd/ | S11.4, S1.4 | .github/workflows/ | Full CI/CD: test → build → deploy | Staging + production environments | pending |

## Phase 12: Telegram Bot — PARALLEL from P7+

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S12.1 | specs/spec-S12.1-telegram-bot-setup/ | S1.2 | src/services/telegram/ | Telegram bot core setup | python-telegram-bot, webhook/polling modes, /start /help commands, bot token config | pending |
| S12.2 | specs/spec-S12.2-telegram-rag-integration/ | S12.1, S7.3 | src/services/telegram/handlers/ | RAG-powered message handling | /ask command → agentic RAG pipeline, citation-backed answers in Telegram, conversation context per chat_id | pending |
| S12.3 | specs/spec-S12.3-telegram-search/ | S12.1, S4.4 | src/services/telegram/handlers/ | Paper search via Telegram | /search command, inline query mode, paper result cards with arXiv links, pagination via callback buttons | pending |
| S12.4 | specs/spec-S12.4-telegram-notifications/ | S12.1, S3.4 | src/services/telegram/notifications.py | Paper alert notifications | New paper alerts (daily digest), topic subscriptions per user, scheduled via Airflow or background task | pending |

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
| S5.1 | LLM client (Ollama + Gemini 3 Flash) | P5 | notebooks/specs/S5.1_llm_client.ipynb | done |
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
| S8.1 | PDF upload endpoint | P8 | notebooks/specs/S8.1_upload.ipynb | pending |
| S8.2 | Paper summary generation | P8 | notebooks/specs/S8.2_summary.ipynb | pending |
| S8.3 | Key highlights extraction | P8 | notebooks/specs/S8.3_highlights.ipynb | pending |
| S8.4 | Methodology analysis | P8 | notebooks/specs/S8.4_methodology.ipynb | pending |
| S8.5 | Paper comparison | P8 | notebooks/specs/S8.5_comparison.ipynb | pending |
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
| S11.1 | Langfuse tracing | P11 | notebooks/specs/S11.1_tracing.ipynb | pending |
| S11.2 | Monitoring & alerts | P11 | notebooks/specs/S11.2_monitoring.ipynb | pending |
| S11.3 | Ops documentation | P11 | — (docs) | pending |
| S11.4 | GCP Cloud Run deployment | P11 | — (deploy) | pending |
| S11.5 | Production CI/CD | P11 | — (workflows) | pending |
| S12.1 | Telegram bot setup | P12 | notebooks/specs/S12.1_telegram_setup.ipynb | pending |
| S12.2 | Telegram RAG integration | P12 | notebooks/specs/S12.2_telegram_rag.ipynb | pending |
| S12.3 | Telegram search | P12 | notebooks/specs/S12.3_telegram_search.ipynb | pending |
| S12.4 | Telegram notifications | P12 | notebooks/specs/S12.4_telegram_notifications.ipynb | pending |
