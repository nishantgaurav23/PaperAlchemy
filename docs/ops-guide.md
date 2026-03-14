# PaperAlchemy Operations Guide

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker & Docker Compose | v2+ | [docker.com](https://docs.docker.com/get-docker/) |
| Python | 3.12 | [python.org](https://www.python.org/) |
| UV | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 20+ | [nodejs.org](https://nodejs.org/) |
| pnpm | 9+ | `npm install -g pnpm` |

---

## Quick Start (Recommended)

The fastest way to get everything running:

```bash
# 1. Clone and enter the project
cd PaperAlchemy

# 2. Copy environment config
cp .env.example .env
# Edit .env and fill in your API keys (GEMINI__API_KEY, JINA__API_KEY, etc.)

# 3. Install dependencies
make setup                     # Python deps (via UV)
cd frontend && pnpm install    # Node deps
cd ..

# 4. Start everything (Docker infra + backend + frontend)
make dev
```

This starts:
- **PostgreSQL** on `localhost:5433`
- **Redis** on `localhost:6380`
- **OpenSearch** on `localhost:9201`
- **Backend API** on `http://localhost:8002`
- **Frontend** on `http://localhost:3000`

Press `Ctrl+C` to stop backend and frontend. Docker services keep running.

---

## Fixed Ports Reference

Every service has a **fixed port**. These never change.

| Service | Host Port | Container Port | URL |
|---------|-----------|---------------|-----|
| **Frontend (Next.js)** | 3000 | — | `http://localhost:3000` |
| **Backend API (local)** | 8000 | — | `http://localhost:8002` |
| **Backend API (Docker)** | 8001 | 8000 | `http://localhost:8001` |
| PostgreSQL | 5433 | 5432 | `localhost:5433` |
| Redis | 6380 | 6379 | `localhost:6380` |
| OpenSearch | 9201 | 9200 | `http://localhost:9201` |
| OpenSearch Dashboards | 5602 | 5601 | `http://localhost:5602` |
| Ollama (Docker) | 11434 | 11434 | `http://localhost:11434` |
| Airflow | 8080 | 8080 | `http://localhost:8080` |
| Langfuse | 3001 | 3000 | `http://localhost:3001` |
| pgAdmin | 5050 | 80 | `http://localhost:5050` |
| MinIO API | 9100 | 9000 | `http://localhost:9100` |
| MinIO Console | 9101 | 9001 | `http://localhost:9101` |

---

## Environment Setup

### 1. Create `.env` file

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```bash
# Required for cloud LLM
GEMINI__API_KEY=your_key_here

# Required for embeddings
JINA__API_KEY=your_key_here

# Optional: Anthropic Claude
ANTHROPIC__API_KEY=your_key_here
```

### 2. Frontend env

The frontend `.env.local` should already exist. If not:

```bash
cp frontend/.env.local.example frontend/.env.local
```

Contents:
```
NEXT_PUBLIC_API_URL=http://localhost:8002
```

### 3. Install dependencies

```bash
# Python (backend)
make setup

# Node (frontend)
cd frontend && pnpm install && cd ..
```

---

## Starting Services

### Option A: `make dev` (Everything at once)

```bash
make dev
```

This single command:
1. Starts Docker infrastructure (postgres, redis, opensearch)
2. Starts the backend API on port 8000 (with hot-reload)
3. Starts the frontend on port 3000 (with hot-reload)

Press `Ctrl+C` to stop backend + frontend. Docker containers keep running.

### Option B: Start services individually

```bash
# Terminal 1 — Docker infrastructure
make up

# Terminal 2 — Backend API
make dev-api

# Terminal 3 — Frontend
make dev-frontend
```

### Option C: Backend only (no frontend)

```bash
make up         # Start Docker infra
make dev-api    # Start API on :8000
```

Test it: `curl http://localhost:8002/api/v1/ping`

### Option D: Run the API inside Docker too

```bash
# Builds and starts API container + all infra
docker compose --profile full up -d

# API is now on port 8001 (not 8000)
curl http://localhost:8001/api/v1/ping
```

---

## Docker Compose Profiles

Services are organized into profiles. Only core services start by default.

| Profile | Services | Command |
|---------|----------|---------|
| *(default)* | postgres, redis, opensearch | `make up` |
| `full` | + api, ollama, airflow | `docker compose --profile full up -d` |
| `langfuse` | + langfuse stack (web, worker, clickhouse, postgres, redis, minio) | `make up-langfuse` |
| `dev-tools` | + opensearch-dashboards, pgadmin | `docker compose --profile dev-tools up -d` |
| `platform` | + minio, sandbox (code execution) | `make platform` |
| *all* | Everything | `make up-all` |

### Start specific profile combinations

```bash
# Core + Langfuse
make up-langfuse

# Core + dev tools (pgAdmin, OpenSearch Dashboards)
docker compose --profile dev-tools up -d

# Core + platform (MinIO, sandbox)
make platform

# Absolutely everything
make up-all
```

---

## Stopping Services

```bash
# Stop all Docker containers (preserves data)
make down

# Stop all + DELETE all volumes (fresh start)
make down-clean
```

---

## Rebuilding

### Rebuild the backend Docker image

```bash
# Rebuild API image only
make build

# Rebuild and restart
make restart
```

### Rebuild after changing Python dependencies

```bash
# Update lock file and install
uv lock
uv sync

# If running in Docker, rebuild the image
make build
docker compose --profile full up -d
```

### Rebuild after changing Node dependencies

```bash
cd frontend
pnpm install
# Restart frontend
pnpm dev --port 3000
```

### Full clean rebuild (nuclear option)

```bash
# Stop everything and wipe all Docker volumes
make down-clean

# Rebuild from scratch
make build

# Start fresh
make dev
```

---

## Database Migrations (Alembic)

```bash
# Generate a new migration after changing models
make db-migrate msg="add users table"

# Apply all pending migrations
make db-upgrade

# Revert the last migration
make db-downgrade
```

---

## Health Checks

```bash
# Check all services
make health
```

This checks PostgreSQL, OpenSearch, Redis, and the API (both local :8000 and Docker :8001).

### Manual checks

```bash
# Backend API
curl http://localhost:8002/api/v1/ping

# PostgreSQL
docker compose exec postgres pg_isready -U paperalchemy

# Redis
docker compose exec redis redis-cli ping

# OpenSearch
curl http://localhost:9201/_cluster/health
```

---

## Viewing Logs

```bash
# All Docker service logs
make logs

# Specific service logs
make logs s=postgres
make logs s=redis
make logs s=opensearch
make logs s=api          # Only if API runs in Docker
```

Backend API logs (local dev) appear directly in the terminal where you ran `make dev` or `make dev-api`.

---

## Running Tests

```bash
# Backend tests
make test

# Backend tests with coverage
make test-cov

# Frontend tests
cd frontend && pnpm test

# Frontend tests in watch mode
cd frontend && pnpm test:watch
```

---

## Linting & Formatting

```bash
# Python — format
make format

# Python — lint + type check
make lint

# Frontend — lint
cd frontend && pnpm lint
```

---

## Common Issues & Troubleshooting

### "Failed to fetch" in the UI

1. Make sure the backend is running: `curl http://localhost:8002/api/v1/ping`
2. Check `frontend/.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8002`
3. Restart the frontend after changing `.env.local` (env vars are baked at build time)

### Port already in use

```bash
# Find what's using a port
lsof -i :8000
lsof -i :3000

# Kill it
kill -9 <PID>
```

### OpenSearch won't start (memory issues)

OpenSearch needs at least 512MB. On macOS, increase Docker Desktop memory to 4GB+.

Also check:
```bash
# macOS may need this sysctl (inside Docker VM)
docker compose exec opensearch sysctl -w vm.max_map_count=262144
```

### Docker containers stuck or unhealthy

```bash
# Check status
make status

# View logs for a failing service
make logs s=opensearch

# Nuclear reset — wipes ALL data
make down-clean
make up
```

### Backend can't connect to PostgreSQL/Redis/OpenSearch

Make sure Docker services are running and healthy:
```bash
make health
```

The `.env` file must have the correct **host ports** (not container ports):
- PostgreSQL: `POSTGRES__PORT=5433` (not 5432)
- Redis: `REDIS__PORT=6380` (not 6379)
- OpenSearch: `OPENSEARCH__HOST=http://localhost:9201` (not 9200)

### Frontend env changes not taking effect

`NEXT_PUBLIC_*` variables are embedded at build time. After changing `frontend/.env.local`, you must restart the dev server:

```bash
# Kill the frontend process, then:
cd frontend && pnpm dev
# Or just Ctrl+C and re-run: make dev
```

---

## Architecture Summary

```
Browser (:3000)  -->  Next.js Frontend
                           |
                           v
                      Backend API (:8000)
                       /      |       \
                      v       v        v
               PostgreSQL  OpenSearch  Redis
                (:5433)    (:9201)    (:6380)

       Ollama (:11434)  <--  LLM calls
       Gemini (cloud)   <--  LLM calls
       Jina (cloud)     <--  Embeddings
```

---

## Make Command Reference

| Command | Description |
|---------|-------------|
| `make dev` | Start infra + backend + frontend (all-in-one) |
| `make dev-api` | Start backend API only on :8000 |
| `make dev-frontend` | Start frontend only on :3000 |
| `make up` | Start core Docker services (postgres, redis, opensearch) |
| `make up-all` | Start ALL Docker services |
| `make up-langfuse` | Start core + Langfuse observability |
| `make platform` | Start MinIO + sandbox |
| `make down` | Stop all Docker services |
| `make down-clean` | Stop all + delete volumes |
| `make build` | Build API Docker image |
| `make restart` | Stop + rebuild + start all Docker services |
| `make status` | Show Docker container status |
| `make logs` | Tail all logs (`make logs s=api` for specific) |
| `make health` | Health check all services |
| `make setup` | Install Python dependencies |
| `make format` | Format Python code |
| `make lint` | Lint + type check Python |
| `make test` | Run backend tests |
| `make test-cov` | Run backend tests with coverage report |
| `make db-migrate` | Generate Alembic migration |
| `make db-upgrade` | Apply pending migrations |
| `make db-downgrade` | Revert last migration |
| `make clean` | Stop everything, delete volumes + caches |
