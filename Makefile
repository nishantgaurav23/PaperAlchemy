.PHONY: help dev dev-api dev-frontend up up-all up-langfuse up-platform down down-clean build start stop restart status logs health setup format lint test test-cov gradio clean db-migrate db-upgrade db-downgrade

# Default target
help:
	@echo "PaperAlchemy - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Docker:"
	@echo "  make up            - Start core services (postgres, redis, opensearch)"
	@echo "  make up-all        - Start ALL services (core + full + langfuse + dev-tools)"
	@echo "  make up-langfuse   - Start core + Langfuse observability stack"
	@echo "  make platform      - Start platform services (MinIO, sandbox)"
	@echo "  make down          - Stop all services"
	@echo "  make down-clean    - Stop all services and remove volumes"
	@echo "  make build         - Build API Docker image"
	@echo "  make start         - Start all services (legacy, same as up-all)"
	@echo "  make stop          - Stop all services (legacy, same as down)"
	@echo "  make restart       - Restart all services"
	@echo "  make status        - Show service status"
	@echo "  make logs          - Show service logs (use s=<service> for specific)"
	@echo "  make health        - Check all services health"
	@echo ""
	@echo "Development:"
	@echo "  make dev             - Start backend + frontend (Docker infra must be running)"
	@echo "  make dev-api         - Start backend API only (http://localhost:8002)"
	@echo "  make dev-frontend    - Start frontend only (http://localhost:3000)"
	@echo "  make setup           - Install Python dependencies"
	@echo "  make format          - Format code with ruff"
	@echo "  make lint            - Lint and type check"
	@echo "  make test            - Run tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make gradio          - Start Gradio web UI (http://localhost:7861)"
	@echo "  make clean           - Clean up everything"
	@echo ""
	@echo "Database Migrations:"
	@echo "  make db-migrate msg=\"description\"  - Generate new migration"
	@echo "  make db-upgrade                    - Apply pending migrations"
	@echo "  make db-downgrade                  - Revert last migration"

# ── Docker commands ──────────────────────────────────────────────────────────

# Core services only (postgres, redis, opensearch)
up:
	docker compose up -d

# All services: core + full + langfuse + dev-tools
up-all:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform up -d

# Platform services (MinIO, sandbox)
platform:
	docker compose --profile platform up -d

# Alias
up-platform:
	docker compose --profile platform up -d

# Core + Langfuse observability stack
up-langfuse:
	docker compose --profile langfuse up -d

# Build API image
build:
	docker compose build api

# Stop services
down:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform down

# Stop + remove volumes
down-clean:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform down -v

# Legacy aliases
start:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform up --build -d

stop:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform down

restart:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform down && docker compose --profile full --profile langfuse --profile dev-tools --profile platform up --build -d

status:
	docker compose ps -a

# Logs (use: make logs or make logs s=api)
logs:
ifdef s
	docker compose logs -f $(s)
else
	docker compose logs -f
endif

# Health check
health:
	@echo "Checking PostgreSQL..."
	@docker compose exec -T postgres pg_isready -U paperalchemy || echo "PostgreSQL not ready"
	@echo "\nChecking OpenSearch..."
	@curl -sf http://localhost:9201/_cluster/health | head -5 || echo "OpenSearch not running"
	@echo "\nChecking Redis..."
	@docker compose exec -T redis redis-cli ping || echo "Redis not running"
	@echo "\nChecking API (local :8002)..."
	@curl -sf http://localhost:8002/api/v1/ping && echo "API running on :8002" || echo "API not running on :8002"
	@echo "\nChecking API (Docker :8001)..."
	@curl -sf http://localhost:8001/api/v1/ping && echo "API running on :8001" || echo "API not running on :8001"

# ── Local Development ────────────────────────────────────────────────────────

# Start both backend API (port 8000) and frontend (port 3000) together.
# Docker infra (postgres, redis, opensearch) must already be running: make up
dev:
	@echo "Starting Docker infra (postgres, redis, opensearch)..."
	docker compose up -d
	@echo ""
	@echo "Starting backend API on http://localhost:8002"
	@echo "Starting frontend on   http://localhost:3000"
	@echo ""
	@trap 'kill 0' EXIT; \
		uv run uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload & \
		cd frontend && pnpm dev & \
		wait

# Backend API only (http://localhost:8002)
dev-api:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload

# Frontend only (http://localhost:3000)
dev-frontend:
	cd frontend && pnpm dev

# ── Development ──────────────────────────────────────────────────────────────

setup:
	uv sync

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy src

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=html

# Gradio UI
gradio:
	uv run python gradio_launcher.py

# ── Database Migrations ──────────────────────────────────────────────────────

# Generate a new migration (usage: make db-migrate msg="add users table")
db-migrate:
	uv run alembic revision --autogenerate -m "$(msg)"

# Apply all pending migrations
db-upgrade:
	uv run alembic upgrade head

# Revert the last migration
db-downgrade:
	uv run alembic downgrade -1

# Cleanup
clean:
	docker compose --profile full --profile langfuse --profile dev-tools --profile platform down -v
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
