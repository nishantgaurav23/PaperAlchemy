# Spec S1.3 — Docker Infrastructure

## Overview
Multi-service Docker Compose setup and multi-stage Dockerfile for PaperAlchemy. Defines containerized infrastructure for all services: PostgreSQL, OpenSearch, Redis, Ollama, Airflow, and the full Langfuse v3 stack (ClickHouse, MinIO, dedicated Postgres/Redis). Includes Docker profiles for selective service startup (core vs full vs langfuse), health checks on all services, named volumes for persistence, and a production-optimized multi-stage Dockerfile using UV.

## Dependencies
- S1.1 (Dependency declaration) — `pyproject.toml` with UV lockfile for Dockerfile build

## Target Location
- `compose.yml` — Docker Compose service definitions (update existing)
- `Dockerfile` — Multi-stage production build (update existing)
- `Makefile` — Docker convenience commands (update existing)

## Functional Requirements

### FR-1: Docker Compose Service Definitions
- **What**: Define all services with proper configuration, health checks, and networking
- **Services** (13 total):
  1. `api` — PaperAlchemy FastAPI app (built from Dockerfile)
  2. `postgres` — PostgreSQL 16 Alpine (main database)
  3. `redis` — Redis 7 Alpine (app cache, LRU eviction, appendonly)
  4. `opensearch` — OpenSearch 2.19 (single-node, security disabled for dev)
  5. `opensearch-dashboards` — OpenSearch Dashboards 2.19 (dev UI)
  6. `ollama` — Ollama LLM server (local inference)
  7. `airflow` — Apache Airflow 2.10 standalone (ingestion DAGs)
  8. `clickhouse` — ClickHouse 24.8 Alpine (Langfuse analytics)
  9. `langfuse-postgres` — PostgreSQL 17 (Langfuse metadata)
  10. `langfuse-redis` — Redis 7 (Langfuse queue, password-protected)
  11. `langfuse-minio` — MinIO (Langfuse object storage)
  12. `langfuse-web` — Langfuse v3 web UI (tracing dashboard)
  13. `langfuse-worker` — Langfuse v3 worker (event ingestion)
  14. `pgadmin` — pgAdmin4 (PostgreSQL web UI)
- **All services**: Must have health checks, restart policies, named volumes, and shared network
- **Edge cases**: OpenSearch needs `memlock` ulimits; Airflow needs `PYTHONPATH` for project imports

### FR-2: Docker Profiles for Selective Startup
- **What**: Group services into profiles so developers can start only what they need
- **Profiles**:
  - (no profile / default): `postgres`, `redis`, `opensearch` — core infra always starts
  - `full`: All services including `api`, `ollama`, `airflow`, dashboards
  - `langfuse`: `clickhouse`, `langfuse-postgres`, `langfuse-redis`, `langfuse-minio`, `langfuse-web`, `langfuse-worker`
  - `dev-tools`: `opensearch-dashboards`, `pgadmin`
- **Usage**: `docker compose up` starts core; `docker compose --profile langfuse up` adds Langfuse
- **Edge cases**: Services with dependencies across profiles still start their deps

### FR-3: Multi-Stage Dockerfile
- **What**: Production-optimized Docker image using UV for fast dependency installation
- **Stages**:
  1. `base` — UV + Python 3.12 (Bookworm), install deps from lockfile
  2. `final` — Python 3.12 slim, copy venv, install system libs (docling deps), run uvicorn
- **Optimizations**: UV bytecode compilation, build cache mount, no-dev deps, slim final image
- **Edge cases**: Docling requires `libxcb1`, `libgl1`, `libglib2.0-0` system libraries

### FR-4: Makefile Docker Commands
- **What**: Convenience Make targets for common Docker operations
- **Targets**:
  - `make up` — Start core services (postgres, redis, opensearch)
  - `make up-all` — Start all services (`--profile full --profile langfuse --profile dev-tools`)
  - `make up-langfuse` — Start core + Langfuse stack
  - `make down` — Stop all services
  - `make down-clean` — Stop + remove volumes
  - `make status` — Show service status
  - `make logs` — Follow logs (all or specific service)
  - `make health` — Check health of all running services
  - `make build` — Build API image
- **Edge cases**: `make logs s=api` for single service logs

### FR-5: Environment Variable Injection
- **What**: Compose services use correct env vars matching `src/config.py` settings
- **API service**: Overrides for container networking (postgres→`postgres`, opensearch→`http://opensearch:9200`, etc.)
- **Airflow**: Needs `PYTHONPATH=/opt/airflow/project` for importing `src/` modules
- **Langfuse**: Pre-seeded org, project, API keys for zero-config dev setup
- **Edge cases**: API reads from `.env` file + environment overrides in compose

### FR-6: Volume Definitions
- **What**: Named volumes for all stateful services
- **Volumes**: `postgres_data`, `redis_data`, `opensearch_data`, `ollama_data`, `airflow_logs`, `clickhouse_data`, `langfuse_postgres_data`, `langfuse_minio_data`
- **Edge cases**: `ollama_data` uses local driver for model persistence

### FR-7: Network Configuration
- **What**: Single bridge network connecting all services
- **Network**: `paperalchemy-network` (bridge driver)
- **All services**: Must be on this network

## Tangible Outcomes
- [ ] `compose.yml` defines all 14 services with health checks
- [ ] Docker profiles work: `docker compose up` starts core (3 services)
- [ ] Docker profiles work: `docker compose --profile langfuse up` starts Langfuse stack
- [ ] Docker profiles work: `docker compose --profile dev-tools up` starts dashboards + pgAdmin
- [ ] `Dockerfile` builds successfully with multi-stage UV pattern
- [ ] Makefile has `up`, `up-all`, `up-langfuse`, `down`, `down-clean`, `build`, `status`, `logs`, `health` targets
- [ ] All services have health checks that report healthy when running
- [ ] Named volumes persist data across restarts
- [ ] API service can connect to postgres, redis, opensearch via container networking
- [ ] No hardcoded secrets in compose.yml (dev defaults are acceptable for local dev)

## Test-Driven Requirements

### Tests to Write First
1. `test_compose_file_valid_yaml`: compose.yml parses as valid YAML
2. `test_compose_all_services_defined`: All 14 expected services exist in compose
3. `test_compose_health_checks`: Every service has a healthcheck block
4. `test_compose_network_assignment`: Every service is on `paperalchemy-network`
5. `test_compose_volume_definitions`: All 8 named volumes are defined
6. `test_compose_profiles`: Services have correct profile assignments
7. `test_compose_api_env_overrides`: API service has correct container networking env vars
8. `test_compose_postgres_config`: Postgres has correct env, port mapping, volume
9. `test_compose_opensearch_config`: OpenSearch has single-node, security disabled, memlock
10. `test_dockerfile_valid`: Dockerfile exists and contains expected stages
11. `test_dockerfile_multi_stage`: Dockerfile has base and final stages
12. `test_dockerfile_uv_pattern`: Dockerfile uses UV for dependency installation
13. `test_makefile_targets`: Makefile contains all expected Docker targets

### Mocking Strategy
- No mocking needed — these are structural/config tests
- Parse YAML to validate compose structure
- Read Dockerfile/Makefile as text to validate content
- No Docker daemon required for tests

### Coverage
- All compose services validated for required fields
- Dockerfile stages and commands validated
- Makefile targets validated for existence
