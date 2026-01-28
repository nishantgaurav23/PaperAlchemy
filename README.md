# PaperAlchemy

## Transform Academic Papers into Knowledge Gold

<div align="center">
  <h3>AI-Powered Production RAG System for Academic Research</h3>
  <p>Build a complete RAG pipeline from infrastructure to agentic AI</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.19-orange.svg" alt="OpenSearch">
  <img src="https://img.shields.io/badge/PostgreSQL-17-blue.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Airflow-2.10-red.svg" alt="Airflow">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Status-Week%202%20Complete-green.svg" alt="Status">
</p>

---

## Overview

PaperAlchemy is a **production-grade RAG (Retrieval-Augmented Generation) system** designed to help researchers discover, understand, and synthesize academic papers from arXiv. This project follows a structured 7-week curriculum, building from infrastructure basics to advanced agentic AI capabilities.

### Why PaperAlchemy?

- **Learn by Building**: Each week adds new capabilities to a real, working system
- **Production-Ready**: Industry best practices, not just tutorials
- **Keyword Search First**: Master BM25 before adding vectors (the professional path)
- **Full Observability**: Langfuse tracing for every RAG interaction
- **Agentic AI**: Intelligent decision-making with LangGraph

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PaperAlchemy System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   arXiv API  │───▶│  Airflow DAG │───▶│   PostgreSQL (17)    │   │
│  │   (CS.AI)    │    │  (Scheduler) │    │   Paper Metadata     │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                             │                       │                │
│                             ▼                       ▼                │
│                    ┌──────────────┐        ┌──────────────┐         │
│                    │   Docling    │        │  OpenSearch  │         │
│                    │  PDF Parser  │        │  (BM25/KNN)  │         │
│                    └──────────────┘        └──────────────┘         │
│                                                    │                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Application                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │ /search │  │  /ask   │  │ /stream │  │ /agentic-ask    │  │   │
│  │  │  BM25   │  │   RAG   │  │   SSE   │  │   LangGraph     │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                             │                                        │
│           ┌─────────────────┼─────────────────┐                     │
│           ▼                 ▼                 ▼                      │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
│    │  Ollama  │      │  Redis   │      │ Langfuse │                 │
│    │   LLM    │      │  Cache   │      │ Tracing  │                 │
│    └──────────┘      └──────────┘      └──────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **API Framework** | FastAPI | 0.115+ | REST API with async support |
| **Database** | PostgreSQL | 17 | Paper metadata storage |
| **Search Engine** | OpenSearch | 2.19 | Hybrid search (BM25 + Vector) |
| **Workflow** | Apache Airflow | 2.10 | Data pipeline orchestration |
| **LLM** | Ollama | Latest | Local LLM serving |
| **Embeddings** | Jina AI | v3 | 1024-dim vector embeddings |
| **Cache** | Redis | 7+ | Query caching |
| **Monitoring** | Langfuse | 3 | RAG pipeline tracing |
| **PDF Parser** | Docling | Latest | Scientific PDF extraction |
| **UI** | Gradio | Latest | Interactive chat interface |
| **Agents** | LangGraph | Latest | Agentic RAG workflows |
| **Bot** | Telegram | - | Mobile access |

---

## Project Structure

```
PaperAlchemy/
├── src/                          # Main application code
│   ├── config.py                 # Centralized configuration (Pydantic Settings)
│   ├── main.py                   # FastAPI application entry point
│   ├── dependencies.py           # Dependency injection
│   ├── db/                       # Database layer
│   │   ├── database.py           # SQLAlchemy engine & session
│   │   └── factory.py            # Database factory
│   ├── models/                   # SQLAlchemy ORM models
│   │   ├── base.py               # Base model with timestamps
│   │   └── paper.py              # Paper model
│   ├── schemas/                  # Pydantic validation schemas
│   │   ├── api/                  # API request/response schemas
│   │   └── arxiv/                # arXiv paper schemas
│   ├── repositories/             # Data access layer (Repository pattern)
│   │   └── paper.py              # Paper CRUD operations
│   ├── services/                 # Business logic services
│   │   ├── arxiv/                # arXiv API client
│   │   ├── pdf_parser/           # Docling PDF parser
│   │   ├── opensearch/           # OpenSearch client (Week 3)
│   │   ├── embeddings/           # Jina embeddings (Week 4)
│   │   ├── ollama/               # LLM client (Week 5)
│   │   ├── cache/                # Redis caching (Week 6)
│   │   └── langfuse/             # Observability (Week 6)
│   └── routers/                  # API route handlers
│       ├── ping.py               # Health checks
│       ├── hybrid_search.py      # Search endpoints
│       └── ask.py                # RAG endpoints
├── notebooks/                    # Weekly experimentation notebooks
│   ├── week1/                    # Infrastructure setup
│   ├── week2/                    # arXiv integration & PDF parsing
│   ├── week3/                    # BM25 keyword search
│   ├── week4/                    # Chunking & hybrid search
│   ├── week5/                    # Complete RAG + LLM
│   ├── week6/                    # Monitoring & caching
│   └── week7/                    # Agentic RAG & Telegram
├── airflow/                      # Workflow orchestration
│   └── dags/                     # Airflow DAG definitions
├── tests/                        # Test suite
├── data/                         # Local data storage
│   └── arxiv_pdfs/               # Downloaded PDF cache
├── compose.yml                   # Docker services (12 containers)
├── Dockerfile                    # Application container
├── pyproject.toml                # Python dependencies (UV)
├── Makefile                      # Common commands
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## Development Roadmap

### Week 1: Infrastructure Foundation ✅ COMPLETE
- [x] Docker Compose setup with 12 services
- [x] FastAPI application with health checks
- [x] PostgreSQL 17 database
- [x] OpenSearch 2.19 cluster
- [x] Apache Airflow 2.10 scheduler
- [x] Ollama LLM server
- [x] Redis cache
- [x] Langfuse observability
- [x] OpenSearch Dashboards
- [x] ClickHouse analytics

### Week 2: Data Ingestion Pipeline ✅ COMPLETE
- [x] arXiv API client with rate limiting (3s delay)
- [x] Retry logic with exponential backoff
- [x] PDF download with caching
- [x] Docling PDF parser integration
- [x] SQLAlchemy ORM models (Paper)
- [x] Repository pattern for data access
- [x] Pydantic schemas for validation
- [x] Factory pattern for services
- [x] MetadataFetcher pipeline orchestrator

### Week 3: Keyword Search (BM25) 🔄 IN PROGRESS
- [x] OpenSearch client setup
- [x] Index configuration with analyzers
- [x] Query builder with field boosting
- [ ] BM25 search implementation
- [ ] Search API endpoints
- [ ] Query DSL and filtering
- [ ] Airflow DAG for indexing

### Week 4: Chunking & Hybrid Search
- [ ] Section-based text chunking
- [ ] Jina AI embeddings integration
- [ ] Vector indexing in OpenSearch
- [ ] Hybrid search with RRF fusion
- [ ] Unified search API

### Week 5: Complete RAG Pipeline
- [ ] Ollama LLM integration
- [ ] Prompt engineering
- [ ] Streaming responses (SSE)
- [ ] Gradio chat interface
- [ ] Context window optimization

### Week 6: Production Monitoring
- [ ] Langfuse tracing integration
- [ ] Redis caching layer
- [ ] Performance dashboards
- [ ] Cost analysis & optimization
- [ ] Error tracking

### Week 7: Agentic RAG
- [ ] LangGraph workflows
- [ ] Guardrails & query validation
- [ ] Document grading & re-ranking
- [ ] Adaptive retrieval strategies
- [ ] Telegram bot integration

---

## Quick Start

### Prerequisites

- **Docker Desktop** (with Docker Compose v2)
- **Python 3.12+**
- **UV Package Manager** (`pip install uv`)
- **8GB+ RAM**, 20GB+ disk space
- **macOS / Linux** (Windows with WSL2)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nishantgaurav23/PaperAlchemy.git
cd PaperAlchemy

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (optional - defaults work)

# 3. Install Python dependencies
uv sync

# 4. Start all services
docker compose up --build -d

# 5. Wait for services to initialize (1-2 minutes)
sleep 60

# 6. Verify health
curl http://localhost:8000/health
```

### Running Notebooks

```bash
# Start Jupyter with UV environment
uv run jupyter lab notebooks/
```

---

## Service Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **FastAPI Health** | http://localhost:8000/health | - |
| **Airflow UI** | http://localhost:8080 | `admin` / `admin` |
| **OpenSearch Dashboards** | http://localhost:5602 | No auth |
| **Langfuse** | http://localhost:3001 | Setup on first visit |
| **Ollama API** | http://localhost:11434 | - |
| **PostgreSQL** | localhost:5433 | `paperalchemy` / `paperalchemy_secret` |
| **Redis** | localhost:6380 | - |
| **OpenSearch API** | http://localhost:9201 | No auth |
| **MinIO Console** | http://localhost:9091 | `minio` / `minio123` |

---

## Docker Services

PaperAlchemy runs **12 containerized services**:

```bash
# View all services
docker compose ps

# Service logs
docker compose logs -f api          # FastAPI logs
docker compose logs -f airflow      # Airflow logs
docker compose logs -f opensearch   # OpenSearch logs

# Restart a service
docker compose restart api
```

| Container | Image | Purpose |
|-----------|-------|---------|
| `paperalchemy-api` | Custom | FastAPI application |
| `paperalchemy-postgres` | postgres:17 | Main database |
| `paperalchemy-opensearch` | opensearchproject/opensearch:2.19 | Search engine |
| `paperalchemy-airflow` | apache/airflow:2.10.4 | Workflow scheduler |
| `paperalchemy-ollama` | ollama/ollama | LLM server |
| `paperalchemy-redis` | redis:7-alpine | Caching |
| `paperalchemy-clickhouse` | clickhouse/clickhouse-server | Analytics |
| `paperalchemy-dashboards` | opensearchproject/opensearch-dashboards | Search UI |
| `paperalchemy-langfuse` | langfuse/langfuse:3 | Observability |
| `paperalchemy-langfuse-postgres` | postgres:17 | Langfuse DB |
| `paperalchemy-langfuse-redis` | redis:7-alpine | Langfuse cache |
| `paperalchemy-langfuse-minio` | minio/minio | Langfuse storage |

---

## Make Commands

```bash
make start       # Start all services
make stop        # Stop all services
make restart     # Restart all services
make health      # Check service health
make logs        # View API logs
make test        # Run test suite
make clean       # Remove containers and volumes
make shell       # Open shell in API container
```

---

## Key Features

### 1. arXiv Paper Ingestion
- Fetches CS.AI papers from arXiv API
- Rate-limited requests (3s delay)
- PDF download with local caching
- Docling-powered PDF parsing

### 2. BM25 Keyword Search
- OpenSearch with custom analyzers
- Multi-field search (title^3, abstract^2, authors)
- Fuzzy matching and highlighting
- Category filtering

### 3. Hybrid Search (Coming Week 4)
- Jina AI embeddings (1024 dimensions)
- HNSW vector indexing
- RRF (Reciprocal Rank Fusion)
- Best of keyword + semantic

### 4. RAG Pipeline (Coming Week 5)
- Ollama local LLM
- Context-aware prompts
- Streaming responses
- Gradio chat UI

### 5. Observability (Coming Week 6)
- Langfuse tracing
- Redis caching
- Performance metrics
- Cost tracking

---

## Configuration

All configuration is managed through environment variables with sensible defaults:

```bash
# PostgreSQL
POSTGRES__HOST=localhost
POSTGRES__PORT=5433
POSTGRES__DATABASE=paperalchemy
POSTGRES__USER=paperalchemy
POSTGRES__PASSWORD=paperalchemy_secret

# OpenSearch
OPENSEARCH__HOST=http://localhost:9201
OPENSEARCH__INDEX_NAME=arxiv-papers

# arXiv
ARXIV__MAX_RESULTS=15
ARXIV__SEARCH_CATEGORY=cs.AI
ARXIV__RATE_LIMIT_DELAY=3.0

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
```

See `.env.example` for all available options.

---

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_arxiv_client.py -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker resources (need 8GB+ RAM)
docker system df
docker compose logs
```

**Database connection errors:**
```bash
# Verify PostgreSQL is running
docker compose ps postgres
docker compose logs postgres
```

**OpenSearch not responding:**
```bash
# OpenSearch needs time to initialize
sleep 60
curl http://localhost:9201/_cluster/health
```

**Airflow DAG import errors:**
```bash
# Check Airflow logs
docker compose logs airflow
```

---

## Author

**Nishant Gaurav** - [@nishantgaurav23](https://github.com/nishantgaurav23)

- Email: nishantgaurav23@gmail.com
- GitHub: [github.com/nishantgaurav23](https://github.com/nishantgaurav23)

---

## Acknowledgments

This project structure and architecture is inspired by [arxiv-paper-curator](https://github.com/jamwithai/arxiv-paper-curator) by Jam With AI.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>PaperAlchemy</strong> - Transforming Academic Papers into Knowledge Gold</p>
  <p>Built with ❤️ for researchers and AI enthusiasts</p>
</div>
